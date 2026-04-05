# src/train.py
"""
Training loop that works for BOTH Custom CNN and ResNet-18.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))   

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
from loguru import logger
from datetime import datetime

logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)


def get_model(model_name: str):
    """Factory function to create either CustomCNN or ResNet18."""
    if model_name.lower() == "custom_cnn":
        from src.cnn_model import CustomCNN
        return CustomCNN()
    elif model_name.lower() == "resnet18":
        from src.resnet_model import ResNet18Transfer
        return ResNet18Transfer(freeze_backbone=True)   # Start with frozen backbone
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(test_loader), 100. * correct / total


def train_model(model_name: str = "custom_cnn", num_epochs=None):
    """Main training function - works for both models."""
    if num_epochs is None:
        num_epochs = config['training']['epochs']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Training on device: {device} | Model: {model_name.upper()}")

    model = get_model(model_name)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    best_acc = 0.0
    best_model_path = None

    for epoch in range(num_epochs):
        logger.info(f"\n📅 Epoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)   # Note: train_loader not defined yet
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            best_model_path = f"{config['paths']['checkpoint_dir']}/best_{model_name}_{timestamp}_acc{best_acc:.1f}.pth"
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': best_acc,
                'model_name': model_name
            }, best_model_path)
            
            logger.info(f"💾 New best model saved → Accuracy: {best_acc:.2f}%")

    logger.info(f"✅ Training completed! Best accuracy: {best_acc:.2f}%")
    return best_model_path


# ====================== Main Execution ======================
if __name__ == "__main__":
    from src.dataset import get_data_loaders

    logger.info("Starting training...")
    train_loader, test_loader, _ = get_data_loaders()

    # Change this line to switch models:
    # model_name = "custom_cnn"     # or "resnet18"
    model_name = "resnet18"         # ← We start with ResNet-18

    best_path = train_model(model_name=model_name, num_epochs=5)   # Start with 5 epochs for testing

    logger.info(f"✅ Training finished! Best model: {best_path}")