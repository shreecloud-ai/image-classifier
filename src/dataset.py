# src/dataset.py
"""
Handles loading, transforming, and creating DataLoaders for CIFAR-10.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

# Load config once
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


def get_transforms(train: bool = True):
    """Return transforms for train or test set with augmentation and resizing for ResNet."""
    if train and config['augmentation']['use_augmentation']:
        transform_list = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((224, 224)),          # Important for ResNet-18
            transforms.ToTensor(),
            transforms.Normalize(mean=config['dataset']['mean'],
                               std=config['dataset']['std'])
        ]
    else:
        transform_list = [
            transforms.Resize((224, 224)),          # Always resize for ResNet
            transforms.ToTensor(),
            transforms.Normalize(mean=config['dataset']['mean'],
                               std=config['dataset']['std'])
        ]
    
    # For Custom CNN we can keep 32x32, but for simplicity we resize both for now
    return transforms.Compose(transform_list)


def get_data_loaders():
    """Create and return train and test DataLoaders."""
    
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=config['paths']['data_dir'],
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=config['paths']['data_dir'],
        train=False,
        download=True,
        transform=test_transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True  # Faster data transfer to GPU (if we use one later)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['test_batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    logger.info(f"✅ CIFAR-10 Data Pipeline Ready!")
    logger.info(f"   Training images : {len(train_dataset)}")
    logger.info(f"   Test images     : {len(test_dataset)}")
    logger.info(f"   Batch size      : {config['training']['batch_size']}")

    return train_loader, test_loader, train_dataset.classes


def visualize_samples(loader, classes, num_images=10):
    """Visualize one sample from each class."""
    dataiter = iter(loader)
    images, labels = next(dataiter)
    
    fig = plt.figure(figsize=(15, 6))
    found = set()
    idx = 0

    while len(found) < num_images and idx < len(images):
        label = labels[idx].item()
        if label not in found:
            found.add(label)
            img = images[idx].numpy().transpose((1, 2, 0))
            mean = np.array(config['dataset']['mean'])
            std = np.array(config['dataset']['std'])
            img = img * std + mean
            img = np.clip(img, 0, 1)

            plt.subplot(2, 5, len(found))
            plt.imshow(img)
            plt.title(classes[label])
            plt.axis('off')
        idx += 1

    plt.suptitle("CIFAR-10 Sample Images", fontsize=16)
    plt.tight_layout()
    plt.show()

def visualize_augmentation_comparison(classes):
    """Show original vs augmented version of the same images."""
    # Create two different transforms
    plain_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config['dataset']['mean'],
                           std=config['dataset']['std'])
    ])
    
    aug_transform = get_transforms(train=True)   # This has augmentation
    
    # Load the same dataset twice with different transforms
    dataset_plain = datasets.CIFAR10(
        root=config['paths']['data_dir'], train=True, download=False, transform=plain_transform
    )
    dataset_aug = datasets.CIFAR10(
        root=config['paths']['data_dir'], train=True, download=False, transform=aug_transform
    )
    
    fig = plt.figure(figsize=(12, 8))
    
    for i in range(6):  # Show 3 pairs
        idx = torch.randint(0, len(dataset_plain), (1,)).item()
        img_original, label = dataset_plain[idx]
        img_aug, _ = dataset_aug[idx]
        
        # Original
        plt.subplot(3, 4, i*2 + 1)
        img_orig = img_original.numpy().transpose(1, 2, 0)
        img_orig = img_orig * np.array(config['dataset']['std']) + np.array(config['dataset']['mean'])
        plt.imshow(np.clip(img_orig, 0, 1))
        plt.title(f"Original\n{classes[label]}")      # Fixed here
        plt.axis('off')
        
        # Augmented
        plt.subplot(3, 4, i*2 + 2)
        img_a = img_aug.numpy().transpose(1, 2, 0)
        img_a = img_a * np.array(config['dataset']['std']) + np.array(config['dataset']['mean'])
        plt.imshow(np.clip(img_a, 0, 1))
        plt.title("Augmented Version")
        plt.axis('off')
    
    plt.suptitle("Original vs Augmented Images", fontsize=14)
    plt.tight_layout()
    plt.show()

# Test when running file directly
if __name__ == "__main__":
    train_loader, test_loader, classes = get_data_loaders()
    
    print("Showing normal samples...")
    visualize_samples(train_loader, classes)
    
    print("\n🔍 Showing Original vs Augmented comparison...")
    visualize_augmentation_comparison(classes)   # Pass classes here
    
    print("✅ Augmentation verification complete!")