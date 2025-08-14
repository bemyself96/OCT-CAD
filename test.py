import os
import random
import argparse
import warnings
import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer

from network.chdnet import CHDNet
from dataloader import get_data_loaders

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
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--info_dim", type=int, default=55)
    parser.add_argument("--max_length", type=int, default=224)
    parser.add_argument("--feature_dim", type=int, default=256)
    parser.add_argument("--data_path", type=str, default="/home/Data/CHD/CHD383")
    parser.add_argument("--data_type", type=list, default=["OCT_Center", "OCTA_Center", "data_proj"])
    parser.add_argument("--eye_id", type=list, default=["OD", "OS"])
    parser.add_argument("--train_list", type=str, default="./datasets/train_1.txt")
    parser.add_argument("--test_list", type=str, default="./datasets/test_1.txt")
    parser.add_argument("--clinical_info", type=str, default="./base/dataset_all/")
    parser.add_argument("--clinical_data", type=str, default="./base/data.xlsx")

    parser.add_argument("--pre_train", type=bool, default=False)
    parser.add_argument(
        "--pre_model_path",
        type=str,
        default="/home/Data/CHD_models/bert_info_image/base/logs_s77/model_100_0.85.pth",
    )
    parser.add_argument(
        "--pre_bert_model_path",
        type=str,
        default="./premodels/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )

    opt = parser.parse_args()

    settings(opt.seed, opt.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    main_model = CHDNet(opt).to(device)
    main_model.load_state_dict(torch.load(opt.pre_model_path)["main_model"])

    auto_model = AutoModel.from_pretrained(opt.pre_bert_model_path)
    tokenizer = AutoTokenizer.from_pretrained(opt.pre_bert_model_path)
    auto_model.to(device)
    for param in auto_model.parameters():
        param.requires_grad = False

    test_loader = get_data_loaders(opt, tokenizer, "test")

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

            _, _, _, _, output = main_model(img_oct, img_octp, img_octa, layer, combined_embedding)

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
        print("name_list:" + str(name_list))
