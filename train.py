import os
import random
import shutil
import argparse
import warnings
import numpy as np

import torch
from torch import nn
from torch.optim import SGD
from torch.optim import lr_scheduler
from transformers import AutoModel, AutoTokenizer

from network.minenet import MineNet
from network.chdnet342 import CHDNet
from dataloader import get_data_loaders
from utils import print_options, MI_Loss

warnings.filterwarnings("ignore")


def settings(seed, cuda):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CHD")
    parser.add_argument("--exp_name", type=str, default="CHD_1")
    parser.add_argument("--cuda", type=str, default="2")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.8)
    parser.add_argument("--step_size", type=int, default=20)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--js_lambda", type=float, default=0.1)

    parser.add_argument("--info_dim", type=int, default=7)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--data_path", type=str, default="/home/Data/lixiaohui/CHD/CHD3831")
    parser.add_argument("--data_type", type=list, default=["OCT_Center", "OCTA_Center", "data_proj"])
    parser.add_argument("--eye_id", type=list, default=["OD", "OS"])
    parser.add_argument("--train_list", type=str, default="./datasets/train_1.txt")
    parser.add_argument("--test_list", type=str, default="./datasets/test_1.txt")
    parser.add_argument("--clinical_info", type=str, default="./base/dataset_all/")
    parser.add_argument("--clinical_data", type=str, default="./base/data.xlsx")

    parser.add_argument("--pre_train", type=bool, default=True)
    parser.add_argument("--pre_model_path", type=str, default="./premodels/resnet34.pth")
    parser.add_argument(
        "--pre_bert_model_path",
        type=str,
        default="./premodels/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    parser.add_argument(
        "--result_root", type=str, default="/home/Data/lixiaohui/CHD_models/bert_info_image/base/logs_s77"
    )

    opt = parser.parse_args()
    opt.result_dir = os.path.join(opt.result_root, opt.exp_name)
    if not os.path.exists(opt.result_dir):
        os.makedirs(opt.result_dir)
        print_options(parser, opt)
        shutil.copyfile(
            os.path.abspath(__file__),
            os.path.join(opt.result_dir, os.path.basename(__file__)),
        )
    else:
        print("result_dir exists: ", opt.result_dir)

    settings(opt.seed, opt.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main_model = CHDNet(opt).to(device)
    mine_model_1 = MineNet(opt.feature_dim * 2, opt.feature_dim * 4).to(device)
    mine_model_2 = MineNet(opt.feature_dim, opt.feature_dim * 2).to(device)

    auto_model = AutoModel.from_pretrained(opt.pre_bert_model_path)
    tokenizer = AutoTokenizer.from_pretrained(opt.pre_bert_model_path)
    auto_model.to(device)
    for param in auto_model.parameters():
        param.requires_grad = False

    ce_criterion = nn.CrossEntropyLoss().to(device)
    mi_criterion = MI_Loss().to(device)
    optimizer = SGD(
        [
            {"params": main_model.parameters()},
            {"params": mine_model_1.parameters()},
            {"params": mine_model_2.parameters()},
        ],
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

    train_loader = get_data_loaders(opt, tokenizer, "train")
    test_loader = get_data_loaders(opt, tokenizer, "test")

    best_acc = 0.0
    best_pre = 0.0
    best_rec = 0.0
    best_f1 = 0.0
    best_spe = 0.0
    best_acc_epoch = 0
    best_label_list = []
    best_predict_list = []
    best_prob_list = []
    best_name_list = []
    best_confuse_matrix = []
    for epoch in range(1, opt.epoch + 1):
        main_model.train()

        train_correct = 0.0
        train_total = 0.0

        for iter, (img_oct, img_octa, img_octp, ie_input_ids, ie_attention_mask, layer, label, pID_eye) in enumerate(
            train_loader
        ):
            img_oct, img_octa, img_octp, ie_input_ids, ie_attention_mask, layer, label = (
                img_oct.to(device),
                img_octa.to(device),
                img_octp.to(device),
                ie_input_ids.to(device),
                ie_attention_mask.to(device),
                layer.to(device),
                label.to(device),
            )

            auto_output = auto_model(ie_input_ids, attention_mask=ie_attention_mask)
            auto_output_last = auto_output.last_hidden_state
            cls_embedding = auto_output_last[:, 0, :]
            mean_embedding = auto_output_last.mean(dim=1)
            max_embedding, _ = auto_output_last.max(dim=1)
            combined_embedding = torch.cat([cls_embedding, mean_embedding, max_embedding], dim=1)

            att_1, att_2, att_31, att_32, output = main_model(img_oct, img_octp, img_octa, layer, combined_embedding)

            ce_loss = ce_criterion(output, label)
            mi_loss = (
                mi_criterion(att_1, att_2, mine_model_1, img_oct.shape[0])
                + mi_criterion(att_31, att_32, mine_model_2, img_oct.shape[0])
            ) / 2

            loss = ce_loss + opt.js_lambda * mi_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_predict = torch.max(output.data, 1)[1]

            train_correct += (train_predict == label).sum()
            train_total += label.shape[0]
            accuracy = train_correct / train_total * 100.0

            print(
                "Training---->Epoch :%d , Batch : %5d , CE_Loss : %.8f, MI_Loss : %.8f, train_correct:%d, train_total:%d, accuracy:%.6f"
                % (
                    epoch,
                    iter + 1,
                    ce_loss.item(),
                    mi_loss.item(),
                    train_correct,
                    train_total,
                    accuracy,
                )
            )
        scheduler.step()

        accuracy = 0.0
        label_list = []
        predict_list = []
        prob_list = []
        name_list = []
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        with torch.no_grad():
            main_model.eval()
            test_correct = 0.0
            test_total = 0.0
            for j, (img_oct, img_octa, img_octp, ie_input_ids, ie_attention_mask, layer, label, pID_eye) in enumerate(
                test_loader
            ):
                dim = label.shape
                img_oct, img_octa, img_octp, ie_input_ids, ie_attention_mask, layer, label = (
                    img_oct.to(device),
                    img_octa.to(device),
                    img_octp.to(device),
                    ie_input_ids.to(device),
                    ie_attention_mask.to(device),
                    layer.to(device),
                    label.to(device),
                )

                auto_output = auto_model(ie_input_ids, attention_mask=ie_attention_mask)
                auto_output_last = auto_output.last_hidden_state
                cls_embedding = auto_output_last[:, 0, :]
                mean_embedding = auto_output_last.mean(dim=1)
                max_embedding, _ = auto_output_last.max(dim=1)
                combined_embedding = torch.cat([cls_embedding, mean_embedding, max_embedding], dim=1)

                _, _, _, _, output = main_model(img_oct, img_octa, img_octp, layer, combined_embedding)

                predicted = torch.max(output.data, 1)[1]
                prob = [a1[1] for a1 in torch.softmax(output, dim=1).data]

                for ii in range(dim[0]):
                    name_list.append(pID_eye[ii])
                    label_list.append(label[ii].item())
                    predict_list.append(predicted[ii].item())
                    prob_list.append(prob[ii].item())

            for jj in range(len(label_list)):
                if label_list[jj] == 0 and predict_list[jj] == 0:
                    tn += 1
                elif label_list[jj] == 0 and predict_list[jj] == 1:
                    fp += 1
                elif label_list[jj] == 1 and predict_list[jj] == 0:
                    fn += 1
                else:
                    tp += 1
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = 0 if (tp + fp) == 0 else tp / (tp + fp)
            recall = 0 if (tp + fn) == 0 else tp / (tp + fn)
            f1_score = 0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn)
            specificity = 0 if (tn + fp) == 0 else tn / (tn + fp)
            print(accuracy)
            print(precision)
            print(recall)
            print(f1_score)
            print(specificity)
            print("confuse_matrix:" + str([tp, fp, fn, tn]))
            print("label_list:" + str(label_list))
            print("predict_list:" + str(predict_list))
            print("prob_list:" + str(prob_list))

        save_dir = os.path.join(opt.result_dir, "models")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_pre = precision
            best_rec = recall
            best_f1 = f1_score
            best_spe = specificity
            best_acc_epoch = epoch
            best_label_list = label_list
            best_predict_list = predict_list
            best_prob_list = prob_list
            best_name_list = name_list
            best_confuse_matrix = [tp, fp, fn, tn]
            state = {"main_model": main_model.state_dict()}
            torch.save(state, os.path.join(save_dir, "best_model_{}_{}.pth".format(epoch, best_acc)))
    state = {"main_model": main_model.state_dict()}
    torch.save(state, os.path.join(save_dir, "final_model_{}_{}.pth".format(epoch, accuracy)))
    result = [best_acc_epoch, best_acc, best_pre, best_rec, best_f1, best_spe]
    with open(os.path.join(opt.result_root, "result_{}.txt".format(opt.exp_name)), "w") as f:
        for item in result:
            f.write(str(item) + "\n")
        f.write("confuse_matrix:" + str(best_confuse_matrix) + "\n")
        f.write("label_list:" + str(best_label_list) + "\n")
        f.write("predict_list:" + str(best_predict_list) + "\n")
        f.write("prob_list:" + str(best_prob_list) + "\n")
        f.write("name_list:" + str(best_name_list) + "\n")
