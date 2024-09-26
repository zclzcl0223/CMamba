import torch
import torch.nn as nn
import math

class GDDMLP(nn.Module):
    def __init__(self, n_vars, reduction=2, avg_flag=True, max_flag=True):
        super().__init__()
        self.avg_flag = avg_flag
        self.max_flag = max_flag
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
           
        self.fc_sc = nn.Sequential(nn.Linear(n_vars, n_vars // reduction, bias=False),
                               nn.GELU(),
                               nn.Linear(n_vars // reduction, n_vars, bias=False))
        self.fc_sf = nn.Sequential(nn.Linear(n_vars, n_vars // reduction, bias=False),
                               nn.GELU(),
                               nn.Linear(n_vars // reduction, n_vars, bias=False))
        self.sigmoid = nn.Sigmoid()

        #self.initialize_weights()

    def initialize_weights(self):
        for layer in self.fc_sc:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0)

        for layer in self.fc_sf:
            if isinstance(layer, nn.Linear):
                nn.init.constant_(layer.weight, 0)

    def forward(self, x):
        b, n, p, d = x.shape
        scale = torch.zeros_like(x)
        shift = torch.zeros_like(x)
        if self.avg_flag:
            sc = self.fc_sc(self.avg_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            sf = self.fc_sf(self.avg_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            scale += sc.unsqueeze(-1)
            shift += sf.unsqueeze(-1)
        if self.max_flag:
            sc = self.fc_sc(self.max_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            sf = self.fc_sf(self.max_pool(x.reshape(b*n, p, d)).reshape(b, n, p).permute(0, 2, 1)).permute(0, 2, 1)
            scale += sc.unsqueeze(-1)
            shift += sf.unsqueeze(-1)
        return self.sigmoid(scale) * x + self.sigmoid(shift)
