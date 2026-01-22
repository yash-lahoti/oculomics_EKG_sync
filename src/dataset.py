"""
PyTorch Dataset for RetinalHydraNet
Handles image loading, augmentation, and multi-task target preparation
"""

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional, Tuple, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config


class RetinalDataset(Dataset):
    """
    Multi-task dataset for retinal fundus images
    Supports simultaneous regression and binary classification targets
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        reg_columns: List[str],
        bin_columns: List[str],
        image_path_col: str = config.IMAGE_PATH_COL,
        transform: Optional[A.Compose] = None,
        is_training: bool = True
    ):
        """
        Args:
            df: DataFrame with image paths and labels
            img_dir: Directory containing images
            reg_columns: List of regression target column names
            bin_columns: List of binary classification target column names
            image_path_col: Name of column containing image paths/filenames
            transform: Albumentations transform pipeline
            is_training: Whether this is training data (for debugging)
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.reg_columns = reg_columns
        self.bin_columns = bin_columns
        self.image_path_col = image_path_col
        self.transform = transform
        self.is_training = is_training

        # Verify columns exist
        missing_cols = []
        for col in reg_columns + bin_columns + [image_path_col]:
            if col not in self.df.columns:
                missing_cols.append(col)

        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        print(f"Dataset initialized: {len(self.df)} samples, "
              f"{len(reg_columns)} regression targets, {len(bin_columns)} binary targets")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Get a single sample

        Returns:
            image: Tensor of shape (C, H, W)
            targets: Dictionary with keys 'regression' and 'binary'
        """
        row = self.df.iloc[idx]

        # Load image
        image_path = row[self.image_path_col]

        # Handle both absolute and relative paths
        if not os.path.isabs(image_path):
            image_path = os.path.join(self.img_dir, image_path)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read image with OpenCV (BGR by default)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']

        # Extract regression targets
        reg_targets = []
        for col in self.reg_columns:
            value = row[col]
            # Handle missing values (replace with 0, will be masked in loss)
            if pd.isna(value):
                value = 0.0
            reg_targets.append(float(value))

        reg_targets = torch.tensor(reg_targets, dtype=torch.float32)

        # Extract binary targets
        bin_targets = []
        for col in self.bin_columns:
            value = row[col]
            # Handle missing values
            if pd.isna(value):
                value = -1.0  # Special value to indicate missing (will be masked in loss)
            bin_targets.append(float(value))

        bin_targets = torch.tensor(bin_targets, dtype=torch.float32)

        targets = {
            'regression': reg_targets,
            'binary': bin_targets
        }

        return image, targets


def get_train_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    """
    Get augmentation pipeline for training
    Uses medical-grade transformations that preserve anatomical structure

    Args:
        img_size: Target image size

    Returns:
        Albumentations composition
    """
    return A.Compose([
        # Resize
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),

        # Geometric augmentations (preserve structure)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=config.AUG_ROTATE_LIMIT, p=0.5, border_mode=cv2.BORDER_CONSTANT),

        # Color augmentations (mild to preserve clinical features)
        A.RandomBrightnessContrast(
            brightness_limit=config.AUG_BRIGHTNESS_LIMIT,
            contrast_limit=config.AUG_CONTRAST_LIMIT,
            p=0.5
        ),

        # Optional: Add slight blur to simulate focus variations
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),

        # Normalize using ImageNet stats (or domain-specific if available)
        A.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
            max_pixel_value=255.0
        ),

        # Convert to PyTorch tensor
        ToTensorV2()
    ])


def get_val_transforms(img_size: int = config.IMG_SIZE) -> A.Compose:
    """
    Get augmentation pipeline for validation/inference
    Only applies necessary preprocessing without data augmentation

    Args:
        img_size: Target image size

    Returns:
        Albumentations composition
    """
    return A.Compose([
        A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    img_dir: str,
    reg_columns: List[str],
    bin_columns: List[str],
    batch_size: int = config.BATCH_SIZE,
    num_workers: int = 4,
    image_path_col: str = config.IMAGE_PATH_COL
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders

    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        img_dir: Image directory
        reg_columns: Regression target columns
        bin_columns: Binary target columns
        batch_size: Batch size
        num_workers: Number of workers for data loading
        image_path_col: Column name for image paths

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = RetinalDataset(
        df=train_df,
        img_dir=img_dir,
        reg_columns=reg_columns,
        bin_columns=bin_columns,
        image_path_col=image_path_col,
        transform=get_train_transforms(),
        is_training=True
    )

    val_dataset = RetinalDataset(
        df=val_df,
        img_dir=img_dir,
        reg_columns=reg_columns,
        bin_columns=bin_columns,
        image_path_col=image_path_col,
        transform=get_val_transforms(),
        is_training=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"Train DataLoader: {len(train_loader)} batches")
    print(f"Val DataLoader: {len(val_loader)} batches")

    return train_loader, val_loader


if __name__ == "__main__":
    """
    Test the dataset implementation with synthetic data
    """
    print("=== Testing RetinalDataset ===\n")

    # Create a dummy image for testing
    dummy_img_path = config.IMG_DIR / "test_image.jpg"
    os.makedirs(config.IMG_DIR, exist_ok=True)

    # Generate a synthetic retinal-like image
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    # Add a circular pattern to simulate fundus
    center = (256, 256)
    radius = 200
    cv2.circle(dummy_image, center, radius, (255, 180, 120), -1)
    cv2.circle(dummy_image, center, radius, (200, 100, 50), 3)

    cv2.imwrite(str(dummy_img_path), dummy_image)
    print(f"Created dummy image: {dummy_img_path}")

    # Create synthetic metadata
    data = {
        'patient_id': ['P001', 'P001', 'P002', 'P002', 'P003'],
        'image_path': ['test_image.jpg'] * 5,
        'grs_cad': [0.5, 0.8, -0.3, 0.2, 1.1],
        'grs_diabetes': [-0.2, 0.4, 0.9, -0.5, 0.3],
        'has_hypertension': [1, 1, 0, 0, 1],
        'has_glaucoma': [0, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)

    print(f"\nDataFrame:\n{df}\n")

    # Define columns
    reg_columns = ['grs_cad', 'grs_diabetes']
    bin_columns = ['has_hypertension', 'has_glaucoma']

    # Test dataset creation
    train_dataset = RetinalDataset(
        df=df,
        img_dir=str(config.IMG_DIR),
        reg_columns=reg_columns,
        bin_columns=bin_columns,
        transform=get_train_transforms(),
        is_training=True
    )

    print(f"\nDataset length: {len(train_dataset)}")

    # Load one sample
    image, targets = train_dataset[0]

    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Regression targets: {targets['regression']}")
    print(f"  Binary targets: {targets['binary']}")

    # Test DataLoader
    print("\n=== Testing DataLoader ===")
    train_df = df.iloc[:3]
    val_df = df.iloc[3:]

    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        img_dir=str(config.IMG_DIR),
        reg_columns=reg_columns,
        bin_columns=bin_columns,
        batch_size=2,
        num_workers=0  # Use 0 for testing to avoid multiprocessing issues
    )

    # Load one batch
    for batch_idx, (images, targets) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Regression targets shape: {targets['regression'].shape}")
        print(f"  Binary targets shape: {targets['binary'].shape}")

        if batch_idx == 0:  # Only print first batch
            break

    # Test augmentation differences
    print("\n=== Testing Augmentation Variability ===")
    aug_samples = [train_dataset[0][0] for _ in range(3)]
    print("Loading same image 3 times with augmentation:")
    for i, img in enumerate(aug_samples):
        print(f"  Sample {i}: mean={img.mean():.3f}, std={img.std():.3f}")

    print("\n=== All Dataset Tests Passed! ===")

    # Cleanup
    os.remove(str(dummy_img_path))
    print(f"\nCleaned up test image: {dummy_img_path}")
