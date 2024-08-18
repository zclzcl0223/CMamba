import torch
from torch import nn
from layers.Embed import PatchEmbedding
from layers.CMambaEncoder import CMambaEncoder

class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs, nvars, d_model, patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class Model(nn.Module):

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride
        # patching and embedding
        configs.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.head_dropout)
        # Encoder
        self.encoder = CMambaEncoder(configs)
        # Prediction Head
        self.head_nf = configs.d_model * configs.patch_num
        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.head_dropout)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Instance Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)
        enc_out = self.encoder(enc_out)
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # enc_out: [bs, nvar, d_model, patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)
        # Decoder
        dec_out = self.head(enc_out)  # dec_out: [bs, nvars, target_window]
        dec_out = dec_out.permute(0, 2, 1)
        # De-Normalization
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out