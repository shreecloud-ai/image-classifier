import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml
from loguru import logger

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def get_transforms(train: bool = True):
    if train and config['augmentation']['use_augmentation']:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.Resize((224, 224)),   # Required for ResNet
            transforms.ToTensor(),
            transforms.Normalize(mean=config['dataset']['mean'], std=config['dataset']['std'])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config['dataset']['mean'], std=config['dataset']['std'])
        ])

def get_data_loaders():
    train_transform = get_transforms(train=True)
    test_transform = get_transforms(train=False)

    train_dataset = datasets.CIFAR10(root=config['paths']['data_dir'], train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=config['paths']['data_dir'], train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training']['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=config['training']['num_workers'], pin_memory=True)

    logger.info(f"✅ CIFAR-10 loaded | Train: {len(train_dataset)} | Test: {len(test_dataset)} | Aug: {config['augmentation']['use_augmentation']}")
    return train_loader, test_loader, train_dataset.classes
