# src/resnet_model.py
"""
ResNet-18 with transfer learning for CIFAR-10.
We use a pre-trained model and adapt it for our 10 classes.
"""

import torch
import torch.nn as nn
from torchvision import models
import yaml
from loguru import logger

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


class ResNet18Transfer(nn.Module):
    """ResNet-18 with transfer learning for CIFAR-10."""
    
    def __init__(self, num_classes=10, freeze_backbone=True):
        super().__init__()
        
        # Load pre-trained ResNet-18 from ImageNet
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Freeze or unfreeze the backbone
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("✅ Backbone frozen - only final layer will be trained")
        else:
            logger.info("✅ All layers unfrozen for full fine-tuning")
        
        # Replace the final fully connected layer for our 10 classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        logger.info(f"✅ ResNet18Transfer created ({'frozen' if freeze_backbone else 'unfrozen'} backbone)")
        logger.info(f"   Final layer changed: {num_features} features → {num_classes} classes")
    
    def forward(self, x):
        """Forward pass."""
        return self.model(x)
    
    def unfreeze(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("🔓 All layers unfrozen - full fine-tuning enabled")


# Quick test
if __name__ == "__main__":
    model = ResNet18Transfer(freeze_backbone=True)
    print(model)
    
    # Test with dummy input (ResNet expects 224x224 images)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"✅ Output shape: {output.shape}")  # Should be [1, 10]
    print("✅ ResNet-18 model is ready!")