"""
Simple ConvNet for MNIST
"""
import torch
from torch import nn

from model.module import MLP_Block, Encoder

class SimpleConvNet(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.encoder = Encoder(kernel_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = MLP_Block(hidden_size=128, output_size=128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        feats = self.encoder(x)
        logits = self.avgpool(feats)
        logits = torch.flatten(logits, 1)
        logits = self.mlp(logits)
        logits = self.fc(logits)

        return logits

    def _get_params(self):
        return list(self.parameters())

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

