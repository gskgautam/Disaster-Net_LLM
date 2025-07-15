import torch
from torch import nn

class AdaptiveTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, context):
        # x: [batch, embed_dim], context: [batch, embed_dim]
        gate_input = torch.cat([x, context], dim=1)
        gate = self.gate(gate_input)
        gated_x = x * gate + context * (1 - gate)
        # Add sequence dimension for transformer
        out = self.transformer(gated_x.unsqueeze(1)).squeeze(1)
        return out 