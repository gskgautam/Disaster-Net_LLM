import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel

class ImageEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch32', output_dim=512):
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name)
        self.output_dim = output_dim
        self.proj = nn.Linear(self.model.config.projection_dim, output_dim)

    def forward(self, images):
        # images: list of PIL Images or batched tensor
        inputs = self.processor(images=images, return_tensors='pt', padding=True)
        outputs = self.model.get_image_features(**inputs)
        return self.proj(outputs) 