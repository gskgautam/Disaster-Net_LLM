import torch
from torch import nn
from .text_encoder import TextEncoder
from .image_encoder import ImageEncoder
from .geo_encoder import GeoEncoder
from .fusion import MultimodalFusion
from .cross_modal_attention import CrossModalAttention
from .adaptive_transformer import AdaptiveTransformer
from .classifier import Classifier

class DisasterNetLLM(nn.Module):
    def __init__(self, num_classes,
                 text_dim=768, image_dim=512, geo_dim=256, fusion_dim=512, attn_heads=4, attn_layers=2, dropout=0.2):
        super().__init__()
        self.text_encoder = TextEncoder(output_dim=text_dim)
        self.image_encoder = ImageEncoder(output_dim=image_dim)
        self.geo_encoder = GeoEncoder(output_dim=geo_dim)
        self.fusion = MultimodalFusion(text_dim, image_dim, geo_dim, fusion_dim)
        self.cross_modal_attn = CrossModalAttention(fusion_dim, num_heads=attn_heads, num_layers=attn_layers)
        self.adaptive_transformer = AdaptiveTransformer(fusion_dim, num_heads=attn_heads, num_layers=attn_layers)
        self.classifier = Classifier(fusion_dim, num_classes, dropout=dropout)

    def forward(self, texts, images, geos, context=None):
        # texts: list of str, images: list of PIL or tensor, geos: tensor [batch, vars, H, W]
        text_emb = self.text_encoder(texts)
        image_emb = self.image_encoder(images)
        geo_emb = self.geo_encoder(geos)
        # Fusion
        fused = self.fusion(text_emb, image_emb, geo_emb)
        # Cross-modal attention (treat as sequence of 3 modalities)
        seq = torch.stack([text_emb, image_emb, geo_emb], dim=1)  # [batch, 3, dim]
        attn_out = self.cross_modal_attn(seq)
        # Adaptive transformer (use context if provided, else self-fusion)
        if context is None:
            context = fused
        adapted = self.adaptive_transformer(attn_out, context)
        # Classifier
        probs = self.classifier(adapted)
        return probs 