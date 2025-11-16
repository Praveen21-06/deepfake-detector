# baseline_model.py

import torch
import torch.nn as nn
from torchvision import models

class BaselineDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(BaselineDeepfakeDetector, self).__init__()
        # EfficientNet-B0 backbone
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        # Replace classifier head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),          # remove inplace=True
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):
        return self.backbone(x)
