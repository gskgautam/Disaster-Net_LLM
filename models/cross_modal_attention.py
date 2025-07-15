import torch
from torch import nn

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim] (seq_len=3 for [text, image, geo])
        x = x.permute(1, 0, 2)  # [seq_len, batch, embed_dim]
        out = self.transformer(x)
        out = out.permute(1, 0, 2)  # [batch, seq_len, embed_dim]
        # Optionally, pool across seq_len
        pooled = out.mean(dim=1)
        return pooled 