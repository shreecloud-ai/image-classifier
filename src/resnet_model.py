import torch.nn as nn
from torchvision import models
import yaml
from loguru import logger

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

class ResNet18Transfer(nn.Module):
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("✅ Backbone frozen (only final layer training)")
        else:
            logger.info("✅ All layers unfrozen for fine-tuning")

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        logger.info(f"✅ ResNet-18 ready: {num_features} → {num_classes} classes")

    def forward(self, x):
        return self.model(x)

    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("🔓 All layers unfrozen")
