"""
Training script for RetinalHydraNet
Implements cross-validation with patient-level stratification
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error, roc_auc_score
from tqdm import tqdm
from pathlib import Path
import json

import config
from utils import DataSplitter, TargetNormalizer, set_seed, compute_class_weights
from dataset import create_dataloaders
from model import RetinalHydraNet, MultiTaskLoss


class Trainer:
    """Handles training and validation for one fold"""

    def __init__(
        self,
        model: RetinalHydraNet,
        criterion: MultiTaskLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
        fold: int,
        writer: SummaryWriter
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.fold = fold
        self.writer = writer

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()

        running_loss = 0.0
        running_reg_loss = 0.0
        running_bin_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Fold {self.fold} Epoch {epoch} [Train]")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            reg_targets = targets['regression'].to(self.device)
            bin_targets = targets['binary'].to(self.device)

            # Forward pass
            reg_pred, bin_pred = self.model(images)

            # Compute loss
            loss, loss_dict = self.criterion(reg_pred, bin_pred, reg_targets, bin_targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track losses
            running_loss += loss_dict['total']
            running_reg_loss += loss_dict['regression']
            running_bin_loss += loss_dict['binary']

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'reg': f"{loss_dict['regression']:.4f}",
                'bin': f"{loss_dict['binary']:.4f}"
            })

        # Calculate epoch averages
        n_batches = len(train_loader)
        avg_loss = running_loss / n_batches
        avg_reg_loss = running_reg_loss / n_batches
        avg_bin_loss = running_bin_loss / n_batches

        return {
            'loss': avg_loss,
            'reg_loss': avg_reg_loss,
            'bin_loss': avg_bin_loss
        }

    @torch.no_grad()
    def validate_epoch(self, val_loader, epoch):
        """Validate for one epoch"""
        self.model.eval()

        running_loss = 0.0
        running_reg_loss = 0.0
        running_bin_loss = 0.0

        # Collect predictions and targets for metrics
        all_reg_preds = []
        all_reg_targets = []
        all_bin_preds = []
        all_bin_targets = []

        pbar = tqdm(val_loader, desc=f"Fold {self.fold} Epoch {epoch} [Val]")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            reg_targets = targets['regression'].to(self.device)
            bin_targets = targets['binary'].to(self.device)

            # Forward pass
            reg_pred, bin_pred = self.model(images)

            # Compute loss
            loss, loss_dict = self.criterion(reg_pred, bin_pred, reg_targets, bin_targets)

            # Track losses
            running_loss += loss_dict['total']
            running_reg_loss += loss_dict['regression']
            running_bin_loss += loss_dict['binary']

            # Store predictions and targets
            all_reg_preds.append(reg_pred.cpu().numpy())
            all_reg_targets.append(reg_targets.cpu().numpy())
            all_bin_preds.append(torch.sigmoid(bin_pred).cpu().numpy())  # Convert logits to probabilities
            all_bin_targets.append(bin_targets.cpu().numpy())

            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

        # Calculate epoch averages
        n_batches = len(val_loader)
        avg_loss = running_loss / n_batches
        avg_reg_loss = running_reg_loss / n_batches
        avg_bin_loss = running_bin_loss / n_batches

        # Concatenate predictions
        all_reg_preds = np.concatenate(all_reg_preds, axis=0)
        all_reg_targets = np.concatenate(all_reg_targets, axis=0)
        all_bin_preds = np.concatenate(all_bin_preds, axis=0)
        all_bin_targets = np.concatenate(all_bin_targets, axis=0)

        # Calculate metrics
        # MAE for regression
        mae_scores = []
        for i in range(all_reg_preds.shape[1]):
            mae = mean_absolute_error(all_reg_targets[:, i], all_reg_preds[:, i])
            mae_scores.append(mae)
        avg_mae = np.mean(mae_scores)

        # AUC for binary classification
        auc_scores = []
        for i in range(all_bin_preds.shape[1]):
            # Filter out missing values (-1)
            valid_mask = all_bin_targets[:, i] != -1
            if valid_mask.sum() > 0 and len(np.unique(all_bin_targets[valid_mask, i])) > 1:
                try:
                    auc = roc_auc_score(all_bin_targets[valid_mask, i], all_bin_preds[valid_mask, i])
                    auc_scores.append(auc)
                except:
                    pass  # Skip if AUC can't be computed

        avg_auc = np.mean(auc_scores) if auc_scores else 0.0

        return {
            'loss': avg_loss,
            'reg_loss': avg_reg_loss,
            'bin_loss': avg_bin_loss,
            'mae': avg_mae,
            'auc': avg_auc,
            'reg_preds': all_reg_preds,
            'reg_targets': all_reg_targets,
            'bin_preds': all_bin_preds,
            'bin_targets': all_bin_targets
        }

    def fit(self, train_loader, val_loader, epochs, checkpoint_dir):
        """Train for multiple epochs with early stopping"""

        for epoch in range(1, epochs + 1):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate_epoch(val_loader, epoch)

            # Log to TensorBoard
            self.writer.add_scalar(f'Fold_{self.fold}/Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar(f'Fold_{self.fold}/Loss/Val', val_metrics['loss'], epoch)
            self.writer.add_scalar(f'Fold_{self.fold}/RegLoss/Train', train_metrics['reg_loss'], epoch)
            self.writer.add_scalar(f'Fold_{self.fold}/RegLoss/Val', val_metrics['reg_loss'], epoch)
            self.writer.add_scalar(f'Fold_{self.fold}/BinLoss/Train', train_metrics['bin_loss'], epoch)
            self.writer.add_scalar(f'Fold_{self.fold}/BinLoss/Val', val_metrics['bin_loss'], epoch)
            self.writer.add_scalar(f'Fold_{self.fold}/MAE/Val', val_metrics['mae'], epoch)
            self.writer.add_scalar(f'Fold_{self.fold}/AUC/Val', val_metrics['auc'], epoch)
            self.writer.add_scalar(f'Fold_{self.fold}/LR', self.optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{epochs} - Fold {self.fold}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} (Reg: {train_metrics['reg_loss']:.4f}, Bin: {train_metrics['bin_loss']:.4f})")
            print(f"  Val Loss: {val_metrics['loss']:.4f} (Reg: {val_metrics['reg_loss']:.4f}, Bin: {val_metrics['bin_loss']:.4f})")
            print(f"  Val MAE: {val_metrics['mae']:.4f}, Val AUC: {val_metrics['auc']:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])

            # Early stopping and checkpointing
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.epochs_no_improve = 0

                # Save best model
                checkpoint_path = checkpoint_dir / f"best_model_fold_{self.fold}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_mae': val_metrics['mae'],
                    'val_auc': val_metrics['auc'],
                }, checkpoint_path)
                print(f"  ✓ Saved best model (Val Loss: {val_metrics['loss']:.4f})")

            else:
                self.epochs_no_improve += 1
                print(f"  No improvement for {self.epochs_no_improve} epoch(s)")

                if self.epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
                    print(f"  Early stopping triggered!")
                    break

        return val_metrics


def train_cross_validation(args):
    """Main training function with cross-validation"""

    # Set random seed
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from: {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} samples")

    # Parse column names
    reg_columns = args.reg_columns.split(',') if args.reg_columns else []
    bin_columns = args.bin_columns.split(',') if args.bin_columns else []

    print(f"Regression columns: {reg_columns}")
    print(f"Binary columns: {bin_columns}")

    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create data splitter
    splitter = DataSplitter(n_folds=args.n_folds, seed=args.seed)
    folds = splitter.get_folds(df, patient_id_col=args.patient_id_col)

    # Store results from all folds
    all_results = []
    fold_metrics = []

    # Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print("\n" + "="*80)
        print(f"FOLD {fold_idx + 1}/{args.n_folds}")
        print("="*80)

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        # Normalize regression targets (fit on train only!)
        normalizer = TargetNormalizer()
        normalizer.fit(train_df, reg_columns)

        train_df_norm = normalizer.transform(train_df)
        val_df_norm = normalizer.transform(val_df)

        # Compute class weights for binary targets
        class_weights = compute_class_weights(train_df, bin_columns)
        bin_pos_weights = {i: class_weights[col] for i, col in enumerate(bin_columns)}

        # Create dataloaders
        train_loader, val_loader = create_dataloaders(
            train_df=train_df_norm,
            val_df=val_df_norm,
            img_dir=args.img_dir,
            reg_columns=reg_columns,
            bin_columns=bin_columns,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_path_col=args.image_path_col
        )

        # Create model
        model = RetinalHydraNet(
            num_reg_targets=len(reg_columns),
            num_bin_targets=len(bin_columns),
            backbone=args.backbone,
            pretrained=args.pretrained,
            dropout_rate=args.dropout
        ).to(device)

        # Create loss function
        criterion = MultiTaskLoss(
            lambda_reg=args.lambda_reg,
            lambda_bin=args.lambda_bin,
            bin_pos_weights=bin_pos_weights
        )

        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Create scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )

        # Create TensorBoard writer
        writer = SummaryWriter(log_dir / f'fold_{fold_idx}')

        # Create trainer
        trainer = Trainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            fold=fold_idx,
            writer=writer
        )

        # Train
        final_metrics = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            checkpoint_dir=checkpoint_dir
        )

        # Save fold results
        fold_result = {
            'fold': fold_idx,
            'val_loss': final_metrics['loss'],
            'val_mae': final_metrics['mae'],
            'val_auc': final_metrics['auc'],
            'best_val_loss': trainer.best_val_loss
        }
        fold_metrics.append(fold_result)

        # Save predictions for this fold
        val_results = pd.DataFrame({
            'fold': fold_idx,
            'patient_id': val_df[args.patient_id_col].values,
            'image_path': val_df[args.image_path_col].values,
        })

        # Add regression predictions and targets
        for i, col in enumerate(reg_columns):
            val_results[f'{col}_pred'] = final_metrics['reg_preds'][:, i]
            val_results[f'{col}_true'] = final_metrics['reg_targets'][:, i]

        # Add binary predictions and targets
        for i, col in enumerate(bin_columns):
            val_results[f'{col}_pred'] = final_metrics['bin_preds'][:, i]
            val_results[f'{col}_true'] = final_metrics['bin_targets'][:, i]

        all_results.append(val_results)

        writer.close()

    # Combine all fold results
    results_df = pd.concat(all_results, ignore_index=True)
    results_path = checkpoint_dir.parent / 'results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")

    # Print cross-validation summary
    print("\n" + "="*80)
    print("CROSS-VALIDATION SUMMARY")
    print("="*80)

    metrics_df = pd.DataFrame(fold_metrics)
    print(metrics_df.to_string(index=False))

    print(f"\nMean Val Loss: {metrics_df['val_loss'].mean():.4f} ± {metrics_df['val_loss'].std():.4f}")
    print(f"Mean Val MAE: {metrics_df['val_mae'].mean():.4f} ± {metrics_df['val_mae'].std():.4f}")
    print(f"Mean Val AUC: {metrics_df['val_auc'].mean():.4f} ± {metrics_df['val_auc'].std():.4f}")

    # Save summary
    summary = {
        'mean_val_loss': float(metrics_df['val_loss'].mean()),
        'std_val_loss': float(metrics_df['val_loss'].std()),
        'mean_val_mae': float(metrics_df['val_mae'].mean()),
        'std_val_mae': float(metrics_df['val_mae'].std()),
        'mean_val_auc': float(metrics_df['val_auc'].mean()),
        'std_val_auc': float(metrics_df['val_auc'].std()),
        'fold_results': fold_metrics
    }

    summary_path = checkpoint_dir.parent / 'cv_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Train RetinalHydraNet with Cross-Validation')

    # Data arguments
    parser.add_argument('--csv', type=str, required=True, help='Path to CSV file with labels')
    parser.add_argument('--img_dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--patient_id_col', type=str, default=config.PATIENT_ID_COL, help='Patient ID column name')
    parser.add_argument('--image_path_col', type=str, default=config.IMAGE_PATH_COL, help='Image path column name')
    parser.add_argument('--reg_columns', type=str, required=True, help='Comma-separated regression column names')
    parser.add_argument('--bin_columns', type=str, required=True, help='Comma-separated binary column names')

    # Model arguments
    parser.add_argument('--backbone', type=str, default=config.BACKBONE, help='Model backbone from timm')
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use pretrained weights')
    parser.add_argument('--dropout', type=float, default=config.DROPOUT_RATE, help='Dropout rate')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--lr', type=float, default=config.LR, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=config.WEIGHT_DECAY, help='Weight decay')
    parser.add_argument('--lambda_reg', type=float, default=config.LAMBDA_REG, help='Weight for regression loss')
    parser.add_argument('--lambda_bin', type=float, default=config.LAMBDA_BIN, help='Weight for binary loss')

    # Cross-validation arguments
    parser.add_argument('--n_folds', type=int, default=config.N_FOLDS, help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=config.SEED, help='Random seed')

    # System arguments
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--checkpoint_dir', type=str, default=str(config.CHECKPOINTS_DIR), help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default=str(config.LOGS_DIR), help='TensorBoard log directory')

    args = parser.parse_args()

    # Run training
    train_cross_validation(args)


if __name__ == "__main__":
    main()
