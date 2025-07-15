import torch
from torch import nn

class GeoEncoder(nn.Module):
    def __init__(self, in_channels=4, output_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        # x: [batch, variables, H, W]
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        return self.fc(features) 