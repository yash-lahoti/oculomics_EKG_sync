"""
Configuration file for RetinalHydraNet
Defines global hyperparameters and paths
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMG_DIR = DATA_DIR / "images"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"
LOGS_DIR = EXPERIMENTS_DIR / "logs"
SALIENCY_DIR = EXPERIMENTS_DIR / "saliency_maps"

# Data Parameters
IMG_SIZE = 224
BATCH_SIZE = 16
N_FOLDS = 5

# Model Parameters
BACKBONE = "vit_base_patch16_224"  # Options: vit_base_patch16_224, resnet50, efficientnet_b0
PRETRAINED = True
DROPOUT_RATE = 0.3

# Training Parameters
LR = 1e-4
WEIGHT_DECAY = 1e-5
EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Loss Weights
LAMBDA_REG = 1.0  # Weight for regression loss (MSE)
LAMBDA_BIN = 1.0  # Weight for binary classification loss (BCE)

# Reproducibility
SEED = 42

# Device
DEVICE = "cuda"  # Will be auto-detected in train.py

# Image Normalization (ImageNet stats - update if using domain-specific pretrained model)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data Columns (to be overridden by command-line args if needed)
PATIENT_ID_COL = "patient_id"
IMAGE_PATH_COL = "image_path"  # Or "image_filename"

# Augmentation Parameters
AUG_ROTATE_LIMIT = 15
AUG_BRIGHTNESS_LIMIT = 0.2
AUG_CONTRAST_LIMIT = 0.2

# Validation
MIN_SAMPLES_PER_FOLD = 10  # Minimum samples required per fold

def create_directories():
    """Create necessary directories if they don't exist"""
    for dir_path in [DATA_DIR, IMG_DIR, EXPERIMENTS_DIR,
                     CHECKPOINTS_DIR, LOGS_DIR, SALIENCY_DIR]:
        os.makedirs(dir_path, exist_ok=True)

if __name__ == "__main__":
    print("=== RetinalHydraNet Configuration ===")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")
    print(f"N Folds: {N_FOLDS}")
    print(f"Backbone: {BACKBONE}")
    print(f"Seed: {SEED}")
    create_directories()
    print("\nDirectories created successfully!")
