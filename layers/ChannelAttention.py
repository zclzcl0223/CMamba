import torch
import torch.nn as nn
import math

class ChannelAttention(nn.Module):
    def __init__(self, n_vars, reduction=2, avg_flag=True, max_flag=True):
        super().__init__()
        self.avg_flag = avg_flag
        self.max_flag = max_flag
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Linear(n_vars, n_vars // reduction, bias=True),
                               nn.GELU(),
                               nn.Linear(n_vars // reduction, n_vars, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, n, _, _ = x.shape
        out = torch.zeros_like(x)
        if self.avg_flag:
            out += self.fc(self.avg_pool(x).reshape(b, n)).reshape(b, n, 1, 1)
        if self.max_flag:
            out += self.fc(self.max_pool(x).reshape(b, n)).reshape(b, n, 1, 1)
        return self.sigmoid(out) * x
