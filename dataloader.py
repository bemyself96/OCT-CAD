# coding:UTF-8

import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F


def read_path(data_path, file_list, eye_id):

    imlist = []
    with open(file_list, "r", encoding="utf-8") as rf:
        for line in rf.readlines():
            impath = line.strip()
            for eye in eye_id:
                temp = os.path.join(impath, eye + ".bmp")
                if os.path.exists(os.path.join(data_path, "data_proj", temp)):
                    imlist.append(temp)
                # negative/nanjing/1001/OD.bmp
    return imlist


def read_xlsx(path):

    dataframe = pd.read_excel(path, index_col=0)

    return dataframe


def read_txt(path):

    with open(path, "r", encoding="utf-8") as file:
        content = file.read()

    return content


def get_data_loaders(opt, tokenizer, status="train"):

    if status == "train":
        return DataLoader(TaskDataset(opt, tokenizer, "train"), batch_size=opt.batch_size, shuffle=True)

    if status == "test":
        return DataLoader(TaskDataset(opt, tokenizer, "test"), batch_size=1)


class TaskDataset(Dataset):
    def __init__(self, opt, tokenizer, status="train"):

        self.status = status
        self.eye_id = opt.eye_id
        self.tokenizer = tokenizer
        self.data_path = opt.data_path
        self.data_type = opt.data_type
        self.train_list = opt.train_list
        self.test_list = opt.test_list
        self.max_length = opt.max_length
        self.clinical_data = opt.clinical_data
        self.clinical_info = opt.clinical_info

        self.clinical_input = read_xlsx(self.clinical_data)

        if status == "train":
            self.datalist = read_path(self.data_path, self.train_list, self.eye_id)
            print("# train samples: {}".format(len(self.datalist)))

        if status == "test":
            self.datalist = read_path(self.data_path, self.test_list, self.eye_id)
            print("# test samples: {}".format(len(self.datalist)))

    def __len__(self):
        return len(self.datalist)

    def augment(self, oct, octa, octp):
        if np.random.rand() < 0.5:
            oct = F.hflip(oct)
            octa = F.hflip(octa)
            octp = F.hflip(octp)
        return oct, octa, octp

    def __getitem__(self, idx):

        filename = self.datalist[idx]
        oct_filename = os.path.join(self.data_path, self.data_type[0], filename)
        octa_filename = os.path.join(self.data_path, self.data_type[1], filename)
        octp_filename = os.path.join(self.data_path, self.data_type[2], filename)
        # /home/Data/lixiaohui/CHD/negative/nanjing/1001/OD.bmp
        pID_eye = filename.replace("\\", "/").split("/")[-2]
        eye_id = filename.replace("\\", "/").split("/")[-1]
        # 1001_OD

        layer_data = self.clinical_input.loc[int(pID_eye)]
        layer = torch.from_numpy(layer_data.values).float()

        info_txt = [read_txt(os.path.join(self.clinical_info, pID_eye + ".txt"))]
        info_encoding = self.tokenizer.batch_encode_plus(
            info_txt,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ie_input_ids = info_encoding["input_ids"].flatten()
        ie_attention_mask = info_encoding["attention_mask"].flatten()

        if "negative" in filename:
            label = 0
        elif "positive" in filename:
            label = 1

        img_oct = Image.open(oct_filename)
        img_octa = Image.open(octa_filename)
        img_octp = Image.open(octp_filename)

        transformer = transforms.Compose(
            [
                transforms.Resize((400, 400)),
                transforms.ToTensor(),
            ]
        )

        if self.status == "train":
            img_oct, img_octa, img_octp = self.augment(img_oct, img_octa, img_octp)
            transformer1 = transforms.Compose(
                [
                    transforms.Resize((400, 400)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.1),
                    transforms.ToTensor(),
                ]
            )
        elif self.status == "test":
            transformer1 = transforms.Compose(
                [
                    transforms.Resize((400, 400)),
                    transforms.ToTensor(),
                ]
            )
        img_oct = transformer1(img_oct)
        img_octa = transformer(img_octa)
        img_octp = transformer1(img_octp)

        return img_oct, img_octa, img_octp, ie_input_ids, ie_attention_mask, layer, label, pID_eye + "_" + eye_id


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CHD")
    parser.add_argument("--batch_size", type=int, default=16)

    # data option
    parser.add_argument("--data_path", type=str, default="/home/Data/lixiaohui/CHD")
    parser.add_argument("--data_type", type=list, default=["OCT_Center", "OCTA_Center", "data_proj"])
    parser.add_argument("--train_list", type=str, default="./datasets/train_1.txt")
    parser.add_argument("--test_list", type=str, default="./datasets/test_1.txt")
    parser.add_argument("--clinical_info", type=str, default="./datasets/info_data.xlsx")
    parser.add_argument("--eye_id", type=list, default=["OD", "OS"])
    parser.add_argument("--num_workers", type=int, default=2)

    opt = parser.parse_args()

    train_loader = get_data_loaders(opt, "train")

    for i, (img_oct, img_octa, img_octp, layer, label, pID_eye) in enumerate(train_loader):
        print(layer.shape)
        # print(name)
        break
