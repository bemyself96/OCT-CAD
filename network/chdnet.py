import torch
import argparse
from torch import nn
from .resnet import resnet34
from .canet import CrossAttNet


class CHDNet(nn.Module):
    def __init__(self, opt):
        super(CHDNet, self).__init__()
        self.info_dim = opt.info_dim
        self.pre_train = opt.pre_train
        self.num_classes = opt.num_classes
        self.feature_dim = opt.feature_dim
        self.pre_model_path = opt.pre_model_path

        self.resnet_oct = resnet34(in_channels=opt.in_channels, out_channels=self.feature_dim // 2)
        self.resnet_octa = resnet34(in_channels=opt.in_channels, out_channels=self.feature_dim // 2)
        self.resnet_octp = resnet34(in_channels=opt.in_channels, out_channels=self.feature_dim)

        self.mlpnet = MLPNet(self.info_dim, self.feature_dim)
        self.bert_fc = nn.Linear(768 * 3, self.feature_dim)

        self.att_1_2 = CrossAttNet(input_dim=self.feature_dim, num_heads=4)
        self.att_1_3 = CrossAttNet(input_dim=self.feature_dim, num_heads=4)
        self.att_2_1 = CrossAttNet(input_dim=self.feature_dim, num_heads=4)
        self.att_2_3 = CrossAttNet(input_dim=self.feature_dim, num_heads=4)
        self.att_3_1 = CrossAttNet(input_dim=self.feature_dim, num_heads=4)
        self.att_3_2 = CrossAttNet(input_dim=self.feature_dim, num_heads=4)

        self.clsnet = CLSNet(self.feature_dim * 7, self.num_classes)

        if self.pre_train:
            self.load_pre_model(self.resnet_oct, self.pre_model_path)
            self.load_pre_model(self.resnet_octa, self.pre_model_path)
            self.load_pre_model(self.resnet_octp, self.pre_model_path)

    def load_pre_model(self, net, model_path):

        model_dict = torch.load(model_path)
        model_dict["conv1.weight"] = model_dict["conv1.weight"].sum(dim=1, keepdim=True)
        del model_dict["fc.weight"]
        del model_dict["fc.bias"]
        net.load_state_dict(model_dict, strict=False)

    def forward(self, img_oct, img_octa, img_octp, x_numerical, x_language, cls=True):
        oct = self.resnet_oct(img_oct)  # b,128
        octa = self.resnet_octa(img_octa)  # b,128
        oct_octa = torch.cat((oct, octa), dim=1)  # b,256
        octp = self.resnet_octp(img_octp)  # b,256
        layer = self.mlpnet(x_numerical)  # b,256
        x_l = self.bert_fc(x_language)  # b,256

        att_12 = oct_octa + self.att_1_2(oct_octa, octp)  # b,256
        att_13 = oct_octa + self.att_1_3(oct_octa, layer)  # b,256
        att_1 = torch.cat((att_12, att_13), dim=1)  # b,512
        att_21 = octp + self.att_2_1(octp, oct_octa)  # b,256
        att_23 = octp + self.att_2_3(octp, layer)  # b,256
        att_2 = torch.cat((att_21, att_23), dim=1)  # b,512
        att_31 = layer + self.att_3_1(layer, oct_octa)  # b,256
        att_32 = layer + self.att_3_2(layer, octp)  # b,256
        att_3 = torch.cat((att_31, att_32), dim=1)  # b,512

        output = self.clsnet(torch.cat((att_1, att_2, att_3, x_l), dim=1))

        return att_1, att_2, att_31, att_32, output


class MLPNet(nn.Module):
    def __init__(self, input_dim=78, output_dim=256):
        super(MLPNet, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim // 2),
            nn.BatchNorm1d(output_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim * 2, output_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x


class CLSNet(nn.Module):
    def __init__(self, input_dim=1024, num_classes=2):
        super(CLSNet, self).__init__()

        self.cls = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim * 2, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.cls(x)
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="CHD")
    parser.add_argument("--batch_size", type=int, default=16)

    # data option
    parser.add_argument("--in_channels", type=int, default=1)
    parser.add_argument("--info_dim", type=int, default=78)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--pre_train", type=str, default=False)
    parser.add_argument("--pre_model_path", type=str, default="pre_model.pth")

    opt = parser.parse_args()
    net = CHDNet(opt)
    x1 = torch.randn(2, 1, 224, 224)
    x2 = torch.randn(2, 1, 224, 224)
    x3 = torch.randn(2, 1, 224, 224)
    x4 = torch.randn(2, 78)
    y = net(x1, x2, x3, x4)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    print(y[3].shape)
