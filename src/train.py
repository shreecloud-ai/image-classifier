# src/train.py
"""
Training loop for both Custom CNN and ResNet-18.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import yaml
from loguru import logger
import os
from datetime import datetime
import sys

# ====================== SETUP LOGGING ======================
logger.remove()  # Remove default handler
logger.add(sys.stdout, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# Load config
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

os.makedirs(config['paths']['checkpoint_dir'], exist_ok=True)

# ... (rest of your file remains the same) ...