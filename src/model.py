"""
RetinalHydraNet Model Architecture
Multi-task model with shared backbone and task-specific heads
"""

import torch
import torch.nn as nn
import timm
from typing import Tuple, Dict
import config


class RetinalHydraNet(nn.Module):
    """
    Multi-task deep learning model for retinal image analysis

    Architecture:
    - Shared backbone (pretrained ViT/ResNet/EfficientNet from timm)
    - Regression head (for continuous genetic risk scores)
    - Binary classification head (for binary clinical outcomes)
    """

    def __init__(
        self,
        num_reg_targets: int,
        num_bin_targets: int,
        backbone: str = config.BACKBONE,
        pretrained: bool = config.PRETRAINED,
        dropout_rate: float = config.DROPOUT_RATE
    ):
        """
        Args:
            num_reg_targets: Number of regression outputs (e.g., 2 for GRS_CAD, GRS_Diabetes)
            num_bin_targets: Number of binary classification outputs (e.g., 2 for hypertension, glaucoma)
            backbone: Model architecture from timm (e.g., 'vit_base_patch16_224', 'resnet50')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        self.num_reg_targets = num_reg_targets
        self.num_bin_targets = num_bin_targets
        self.backbone_name = backbone

        # Load backbone from timm
        print(f"Loading backbone: {backbone} (pretrained={pretrained})")
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove default classifier
            global_pool=''   # We'll add our own pooling
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
            features = self.backbone(dummy_input)

            # Handle different output formats
            if isinstance(features, tuple):
                features = features[0]

            if len(features.shape) == 4:  # CNN output (B, C, H, W)
                self.feature_dim = features.shape[1]
                self.use_adaptive_pool = True
            elif len(features.shape) == 3:  # ViT output (B, N, D)
                self.feature_dim = features.shape[-1]
                self.use_adaptive_pool = False
            else:
                self.feature_dim = features.shape[-1]
                self.use_adaptive_pool = False

        print(f"Feature dimension: {self.feature_dim}")

        # Global pooling for CNNs
        if self.use_adaptive_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        else:
            self.global_pool = None

        # Regression head (for continuous targets like genetic risk scores)
        self.reg_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_reg_targets)
        )

        # Binary classification head (for binary outcomes)
        self.bin_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_bin_targets)
        )

        print(f"Model initialized:")
        print(f"  - Regression outputs: {num_reg_targets}")
        print(f"  - Binary outputs: {num_bin_targets}")
        print(f"  - Total parameters: {self.count_parameters():,}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Tuple of (regression_outputs, binary_outputs)
            - regression_outputs: (B, num_reg_targets) - raw values
            - binary_outputs: (B, num_bin_targets) - logits (pre-sigmoid)
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Handle different output formats
        if isinstance(features, tuple):
            features = features[0]

        # Pool features if needed
        if self.use_adaptive_pool:
            # CNN: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
            features = self.global_pool(features)
            features = features.flatten(1)
        else:
            # ViT: (B, N, D) -> (B, D) using CLS token or mean pooling
            if len(features.shape) == 3:
                # Use CLS token (first token) for ViT
                features = features[:, 0]

        # Generate predictions from both heads
        reg_outputs = self.reg_head(features)
        bin_outputs = self.bin_head(features)

        return reg_outputs, bin_outputs

    def count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")

    def get_last_conv_layer(self):
        """
        Get the last convolutional/attention layer for GradCAM
        Returns the appropriate layer based on architecture
        """
        if 'vit' in self.backbone_name.lower():
            # For Vision Transformers, use the last attention block
            return self.backbone.blocks[-1].norm1
        elif 'resnet' in self.backbone_name.lower():
            # For ResNet, use layer4
            return self.backbone.layer4
        elif 'efficientnet' in self.backbone_name.lower():
            # For EfficientNet, use the last conv layer
            return self.backbone.conv_head
        else:
            # Default: try to find the last layer with spatial dimensions
            for name, module in reversed(list(self.backbone.named_modules())):
                if isinstance(module, (nn.Conv2d, nn.LayerNorm)):
                    return module
            # Fallback
            return list(self.backbone.children())[-1]


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning
    Weighted sum of MSE (regression) and BCE (binary classification)
    """

    def __init__(
        self,
        lambda_reg: float = config.LAMBDA_REG,
        lambda_bin: float = config.LAMBDA_BIN,
        bin_pos_weights: Dict[str, torch.Tensor] = None
    ):
        """
        Args:
            lambda_reg: Weight for regression loss
            lambda_bin: Weight for binary classification loss
            bin_pos_weights: Dictionary of pos_weight tensors for each binary target
        """
        super().__init__()

        self.lambda_reg = lambda_reg
        self.lambda_bin = lambda_bin

        # Regression loss (MSE)
        self.mse_loss = nn.MSELoss(reduction='mean')

        # Binary classification loss (BCE with logits)
        # pos_weight will be set per-target if provided
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')  # We'll handle weighting manually

        self.bin_pos_weights = bin_pos_weights or {}

    def forward(
        self,
        reg_pred: torch.Tensor,
        bin_pred: torch.Tensor,
        reg_target: torch.Tensor,
        bin_target: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss

        Args:
            reg_pred: Regression predictions (B, num_reg)
            bin_pred: Binary predictions as logits (B, num_bin)
            reg_target: Regression targets (B, num_reg)
            bin_target: Binary targets (B, num_bin), use -1 for missing values

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Regression loss
        reg_loss = self.mse_loss(reg_pred, reg_target)

        # Binary classification loss (handle missing values)
        bin_losses = []
        for i in range(bin_pred.shape[1]):
            # Mask out missing values (marked as -1)
            valid_mask = bin_target[:, i] != -1

            if valid_mask.sum() == 0:
                # No valid samples for this target
                bin_losses.append(torch.tensor(0.0, device=bin_pred.device))
                continue

            pred_i = bin_pred[valid_mask, i]
            target_i = bin_target[valid_mask, i]

            # Compute BCE loss
            loss_i = self.bce_loss(pred_i.unsqueeze(1), target_i.unsqueeze(1))

            # Apply pos_weight if available
            if i in self.bin_pos_weights:
                pos_weight = self.bin_pos_weights[i].to(bin_pred.device)
                # Weight positive samples
                weights = torch.where(target_i == 1, pos_weight, torch.tensor(1.0, device=bin_pred.device))
                loss_i = (loss_i.squeeze() * weights).mean()
            else:
                loss_i = loss_i.mean()

            bin_losses.append(loss_i)

        bin_loss = torch.stack(bin_losses).mean() if bin_losses else torch.tensor(0.0, device=bin_pred.device)

        # Total weighted loss
        total_loss = self.lambda_reg * reg_loss + self.lambda_bin * bin_loss

        # Return detailed losses for logging
        loss_dict = {
            'total': total_loss.item(),
            'regression': reg_loss.item(),
            'binary': bin_loss.item()
        }

        return total_loss, loss_dict


if __name__ == "__main__":
    """Test the model architecture"""
    print("=== Testing RetinalHydraNet ===\n")

    # Model parameters
    num_reg = 2  # GRS_CAD, GRS_Diabetes
    num_bin = 2  # Hypertension, Glaucoma

    # Create model
    print("Creating model...")
    model = RetinalHydraNet(
        num_reg_targets=num_reg,
        num_bin_targets=num_bin,
        backbone='resnet50',  # Use ResNet for faster testing
        pretrained=True
    )

    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, config.IMG_SIZE, config.IMG_SIZE)

    with torch.no_grad():
        reg_out, bin_out = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Regression output shape: {reg_out.shape}")
    print(f"Binary output shape: {bin_out.shape}")

    assert reg_out.shape == (batch_size, num_reg), f"Unexpected regression output shape"
    assert bin_out.shape == (batch_size, num_bin), f"Unexpected binary output shape"

    # Test loss function
    print("\nTesting MultiTaskLoss...")

    # Create dummy targets
    reg_target = torch.randn(batch_size, num_reg)
    bin_target = torch.randint(0, 2, (batch_size, num_bin)).float()

    # Create loss with pos_weights
    pos_weights = {0: torch.tensor([2.0]), 1: torch.tensor([1.5])}
    criterion = MultiTaskLoss(bin_pos_weights=pos_weights)

    # Compute loss
    total_loss, loss_dict = criterion(reg_out, bin_out, reg_target, bin_target)

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss breakdown: {loss_dict}")

    # Test GradCAM layer retrieval
    print("\nTesting GradCAM layer retrieval...")
    target_layer = model.get_last_conv_layer()
    print(f"Target layer for GradCAM: {target_layer.__class__.__name__}")

    # Test freezing/unfreezing
    print("\nTesting freeze/unfreeze...")
    model.freeze_backbone()
    model.unfreeze_backbone()

    print("\n=== All Model Tests Passed! ===")
