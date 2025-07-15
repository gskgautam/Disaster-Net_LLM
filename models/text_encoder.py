import torch
from torch import nn
from transformers import GPT2Tokenizer, GPT2Model

class TextEncoder(nn.Module):
    def __init__(self, model_name='openai-gpt', output_dim=768):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.output_dim = output_dim
        self.proj = nn.Linear(self.model.config.hidden_size, output_dim)

    def forward(self, texts):
        # texts: list of strings
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        # Use the last hidden state of the first token (CLS-like)
        pooled = outputs.last_hidden_state[:, 0, :]
        return self.proj(pooled) 