import torch
from torch import nn

class MultimodalFusion(nn.Module):
    def __init__(self, text_dim, image_dim, geo_dim, fusion_dim=512):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + image_dim + geo_dim, fusion_dim),
            nn.ReLU(),
        )

    def forward(self, text_emb, image_emb, geo_emb):
        # text_emb, image_emb, geo_emb: [batch, dim]
        x = torch.cat([text_emb, image_emb, geo_emb], dim=1)
        return self.fusion(x) 