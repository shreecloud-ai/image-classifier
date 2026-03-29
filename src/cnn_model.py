# src/cnn_model.py
"""
Custom CNN model for CIFAR-10 classification.
Built from scratch to learn image features.
"""

import torch
import torch.nn as nn
import yaml
from loguru import logger

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


class CustomCNN(nn.Module):
    """Simple CNN architecture for 32x32 color images."""
    
    def __init__(self, num_classes=10):
        super().__init__()   # Initialize the parent class
        
        # === Feature Extractor (Convolutional Layers) ===
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),  # 3=R,G,B channels
            nn.ReLU(),                                                            # Activation function
            nn.MaxPool2d(kernel_size=2, stride=2),                                # Reduce size 32→16
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                                   # Reduce size 16→8
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                                                   # Reduce size 8→4
        )
        
        # === Classifier (Fully Connected Layers) ===
        self.classifier = nn.Sequential(
            nn.Flatten(),                                                         # Convert 4x4x128 into 1D vector
            nn.Linear(128 * 4 * 4, 512),                                          # Fully connected layer
            nn.ReLU(),
            nn.Dropout(p=config['model']['dropout_rate']),                        # Prevent overfitting
            nn.Linear(512, num_classes)                                           # Output 10 classes
        )
        
        logger.info("✅ CustomCNN model created successfully!")
    
    def forward(self, x):
        """Forward pass: how data flows through the network."""
        x = self.features(x)      # Extract features using conv layers
        x = self.classifier(x)    # Make final prediction
        return x


# Quick test when running this file directly
if __name__ == "__main__":
    model = CustomCNN()
    print(model)                  # Print model architecture
    
    # Test with dummy data
    dummy_input = torch.randn(1, 3, 32, 32)   # Batch of 1 image
    output = model(dummy_input)
    print(f"✅ Output shape: {output.shape}")  # Should be [1, 10]
    print("✅ Custom CNN is ready to be trained!")