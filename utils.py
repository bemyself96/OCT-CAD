import os
import torch
import torch.nn.functional as F


def print_options(parser, opt):

    message = "description: " + parser.description + "\n"
    message += "----------------- Options ---------------\n"
    for k, v in sorted(vars(opt).items()):
        comment = ""
        default = parser.get_default(k)
        if v != default:
            comment = "\t[default: %s]" % str(default)
        message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
    message += "----------------- End -------------------"
    print(message)

    # save to the disk
    file_name = os.path.join(opt.result_dir, "opt.txt")
    with open(file_name, "wt") as opt_file:
        opt_file.write(message)
        opt_file.write("\n")


def print_network(net, opt):
    file_name = os.path.join(opt.result_dir, "network.txt")
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    message = str(net)
    with open(file_name, "wt") as opt_file:
        opt_file.write(message)
        opt_file.write("\n")
        opt_file.write("Total number of parameters: %d" % num_params)
        opt_file.write("\n")


def load_premodel(net, model_path):

    model_dict = torch.load(model_path)
    model_dict["conv1.weight"] = model_dict["conv1.weight"].sum(dim=1, keepdim=True)
    del model_dict["fc.weight"]
    del model_dict["fc.bias"]
    net.load_state_dict(model_dict, strict=False)


class JS_DivLoss(torch.nn.Module):
    def __init__(self):
        super(JS_DivLoss, self).__init__()

    def forward(self, p, q):
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(m.log(), p, reduction="batchmean")
        kl_qm = F.kl_div(m.log(), q, reduction="batchmean")
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd


class MI_Loss(torch.nn.Module):
    def __init__(
        self,
    ):
        super(MI_Loss, self).__init__()

    def mutual_information_loss(self, x, y, model, batch_size):
        joint = model(x, y).mean()
        shuffled_y = y[torch.randperm(batch_size)]
        marginal = torch.exp(model(x, shuffled_y)).mean()
        mi_loss = -(joint - torch.log(marginal))
        return mi_loss

    def forward(self, x, y, model, batch_size):
        return self.mutual_information_loss(x, y, model, batch_size)


def yeild_data(dataloader):
    for batch in dataloader:
        yield batch


def loop_iterable(iterable):
    while True:
        yield from iterable
