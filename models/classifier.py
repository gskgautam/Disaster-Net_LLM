import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes, dropout=0.2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        logits = self.fc(x)
        probs = self.softmax(logits)
        return probs 