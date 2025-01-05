import torch
import torch.nn as nn
import torch.optim as optim


# 定义 MINE 模型 (简单的神经网络)
class MineNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MineNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, y):

        xy = torch.cat([x, y], dim=1)
        return self.fc(xy)
