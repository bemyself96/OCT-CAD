import torch
from torch import nn


class CrossAttBlock(nn.Module):
    def __init__(self, input_dim=256, num_heads=4, attn_drop=0.0, proj_drop=0.0):
        super(CrossAttBlock, self).__init__()

        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_linear = nn.Linear(input_dim, input_dim, bias=False)
        self.k_linear = nn.Linear(input_dim, input_dim, bias=False)
        self.v_linear = nn.Linear(input_dim, input_dim, bias=False)

        # self.q_norm = nn.LayerNorm(input_dim)
        # self.k_norm = nn.LayerNorm(input_dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(input_dim, input_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):

        assert x.shape == y.shape
        B, D = x.shape

        q = self.q_linear(x).reshape(B, self.num_heads, self.head_dim)  # B,4,64
        k = self.k_linear(y).reshape(B, self.num_heads, self.head_dim)  # B,4,64
        v = self.v_linear(y).reshape(B, self.num_heads, self.head_dim)  # B,4,64

        # q, k = self.q_norm(q), self.k_norm(k)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # B,4,64 @ B,64,4 -> B,4,4
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v  # B,4,4 @ B,4,64 -> B,4,64

        x = x.reshape(B, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):

    def __init__(self, in_features):
        super(Mlp, self).__init__()

        self.fc1 = nn.Linear(in_features, in_features * 4)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(in_features * 4, in_features)
        self.drop2 = nn.Dropout()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class CrossAttNet(nn.Module):
    def __init__(self, input_dim=256, num_heads=4):
        super(CrossAttNet, self).__init__()
        self.x_norm = nn.LayerNorm(input_dim)
        self.y_norm = nn.LayerNorm(input_dim)
        self.cross_att = CrossAttBlock(input_dim, num_heads)
        self.drop_path1 = nn.Identity()

        self.norm = nn.LayerNorm(input_dim)
        self.mlp = Mlp(in_features=input_dim)
        self.drop_path2 = nn.Identity()

    def forward(self, x, y):
        x = x + self.drop_path1(self.cross_att(self.x_norm(x), self.y_norm(y)))
        x = x + self.drop_path2(self.mlp(self.norm(x)))
        return x
