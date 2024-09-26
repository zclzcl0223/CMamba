import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN
from layers.GDDMLP import GDDMLP

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.configs = configs
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len        
        self.Linear = nn.ModuleList([
            nn.Linear(self.seq_len, self.pred_len) for _ in range(configs.enc_in)
        ]) if configs.individual else nn.Linear(self.seq_len, self.pred_len)
        self.gddmlp = configs.gddmlp
        if self.gddmlp:
            print("Insert GDDMLP")
            self.GDDMLP = GDDMLP(configs.c_out, configs.reduction, 
                                                 configs.avg, configs.max)
        self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.c_out) if configs.rev else None
        self.individual = configs.individual
        if self.task_name == 'classification':
            self.act = F.gelu
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)


    def encoder(self, x):
        # x: [B, L, D]
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros([x.size(0), x.size(1), self.pred_len],
                                dtype=x.dtype).to(x.device)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            if self.gddmlp:
                x = x.transpose(1, 2)
                x = self.GDDMLP(x.unsqueeze(-2))
                x = x.squeeze(-2)
                x = x.transpose(1, 2)
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred

    def forecast(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        enc_out = self.encoder(x_enc)
        # Output
        # (batch_size, seq_length * d_model)
        output = enc_out.reshape(enc_out.shape[0], -1)
        # (batch_size, num_classes)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
            if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
                dec_out = self.forecast(x_enc)
                return dec_out[:, -self.pred_len:, :]  # [B, L, D]
            if self.task_name == 'imputation':
                dec_out = self.imputation(x_enc)
                return dec_out  # [B, L, D]
            if self.task_name == 'anomaly_detection':
                dec_out = self.anomaly_detection(x_enc)
                return dec_out  # [B, L, D]
            if self.task_name == 'classification':
                dec_out = self.classification(x_enc)
                return dec_out  # [B, N]
            return None