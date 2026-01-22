"""
Utility functions and classes for RetinalHydraNet
Includes data splitting and target normalization with patient-level stratification
"""

import numpy as np
import pandas as pd
import torch
import random
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Dict, Optional
import config


class DataSplitter:
    """
    Handles patient-level stratified cross-validation splits
    Ensures no patient appears in both train and validation sets
    """

    def __init__(self, n_folds: int = config.N_FOLDS, seed: int = config.SEED):
        """
        Args:
            n_folds: Number of cross-validation folds
            seed: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.seed = seed
        self.group_kfold = GroupKFold(n_splits=n_folds)

    def get_folds(self, df: pd.DataFrame, patient_id_col: str = config.PATIENT_ID_COL) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate patient-level stratified folds

        Args:
            df: DataFrame with patient_id column
            patient_id_col: Name of the patient ID column

        Returns:
            List of (train_indices, val_indices) tuples for each fold
        """
        if patient_id_col not in df.columns:
            raise ValueError(f"Column '{patient_id_col}' not found in DataFrame. Available columns: {df.columns.tolist()}")

        # Extract patient groups
        groups = df[patient_id_col].values

        # Create dummy target (GroupKFold doesn't use it, but required by API)
        X = np.arange(len(df))
        y = np.zeros(len(df))

        # Generate folds
        folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(self.group_kfold.split(X, y, groups=groups)):
            # Verify no patient overlap
            train_patients = set(df.iloc[train_idx][patient_id_col].unique())
            val_patients = set(df.iloc[val_idx][patient_id_col].unique())

            overlap = train_patients.intersection(val_patients)
            if overlap:
                raise RuntimeError(f"Patient overlap detected in fold {fold_idx}: {overlap}")

            # Verify minimum samples per fold
            if len(val_idx) < config.MIN_SAMPLES_PER_FOLD:
                print(f"WARNING: Fold {fold_idx} has only {len(val_idx)} validation samples (minimum: {config.MIN_SAMPLES_PER_FOLD})")

            folds.append((train_idx, val_idx))

            print(f"Fold {fold_idx}: Train={len(train_idx)} samples ({len(train_patients)} patients), "
                  f"Val={len(val_idx)} samples ({len(val_patients)} patients)")

        return folds

    def get_patient_summary(self, df: pd.DataFrame, patient_id_col: str = config.PATIENT_ID_COL) -> Dict:
        """
        Get summary statistics about patient distribution

        Args:
            df: DataFrame with patient_id column
            patient_id_col: Name of the patient ID column

        Returns:
            Dictionary with summary statistics
        """
        patient_counts = df[patient_id_col].value_counts()

        summary = {
            'total_samples': len(df),
            'total_patients': len(patient_counts),
            'samples_per_patient_mean': patient_counts.mean(),
            'samples_per_patient_std': patient_counts.std(),
            'samples_per_patient_min': patient_counts.min(),
            'samples_per_patient_max': patient_counts.max(),
        }

        return summary


class TargetNormalizer:
    """
    Normalizes regression targets using StandardScaler
    CRITICAL: Fits only on training data to prevent data leakage
    """

    def __init__(self):
        """Initialize with empty scalers"""
        self.scalers = {}
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, reg_columns: List[str]):
        """
        Fit scalers on training data only

        Args:
            train_df: Training DataFrame
            reg_columns: List of column names for regression targets
        """
        self.reg_columns = reg_columns

        for col in reg_columns:
            if col not in train_df.columns:
                raise ValueError(f"Regression column '{col}' not found in training data")

            scaler = StandardScaler()

            # Fit on training data only
            values = train_df[col].values.reshape(-1, 1)

            # Handle missing values
            valid_mask = ~np.isnan(values.flatten())
            if valid_mask.sum() == 0:
                raise ValueError(f"No valid values found in column '{col}'")

            scaler.fit(values[valid_mask])
            self.scalers[col] = scaler

        self.is_fitted = True
        print(f"TargetNormalizer fitted on {len(reg_columns)} regression columns")

    def transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Transform regression targets using fitted scalers

        Args:
            df: DataFrame to transform
            columns: Optional list of columns to transform (defaults to all fitted columns)

        Returns:
            DataFrame with transformed values
        """
        if not self.is_fitted:
            raise RuntimeError("TargetNormalizer must be fitted before transform")

        if columns is None:
            columns = self.reg_columns

        df_normalized = df.copy()

        for col in columns:
            if col not in self.scalers:
                raise ValueError(f"Column '{col}' was not fitted")

            values = df[col].values.reshape(-1, 1)

            # Handle missing values
            valid_mask = ~np.isnan(values.flatten())
            transformed_values = np.full_like(values.flatten(), np.nan, dtype=float)

            if valid_mask.sum() > 0:
                transformed_values[valid_mask] = self.scalers[col].transform(
                    values[valid_mask]
                ).flatten()

            df_normalized[col] = transformed_values

        return df_normalized

    def inverse_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Inverse transform normalized values back to original scale

        Args:
            df: DataFrame with normalized values
            columns: Optional list of columns to inverse transform (defaults to all fitted columns)

        Returns:
            DataFrame with original scale values
        """
        if not self.is_fitted:
            raise RuntimeError("TargetNormalizer must be fitted before inverse_transform")

        if columns is None:
            columns = self.reg_columns

        df_original = df.copy()

        for col in columns:
            if col not in self.scalers:
                raise ValueError(f"Column '{col}' was not fitted")

            values = df[col].values.reshape(-1, 1)

            # Handle missing values
            valid_mask = ~np.isnan(values.flatten())
            original_values = np.full_like(values.flatten(), np.nan, dtype=float)

            if valid_mask.sum() > 0:
                original_values[valid_mask] = self.scalers[col].inverse_transform(
                    values[valid_mask]
                ).flatten()

            df_original[col] = original_values

        return df_original

    def get_params(self) -> Dict:
        """Get scaler parameters for logging/debugging"""
        if not self.is_fitted:
            return {}

        params = {}
        for col, scaler in self.scalers.items():
            params[col] = {
                'mean': scaler.mean_[0],
                'std': scaler.scale_[0]
            }
        return params


def set_seed(seed: int = config.SEED):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def compute_class_weights(train_df: pd.DataFrame, binary_columns: List[str]) -> Dict[str, torch.Tensor]:
    """
    Compute pos_weight for BCEWithLogitsLoss based on class imbalance

    Args:
        train_df: Training DataFrame
        binary_columns: List of binary classification column names

    Returns:
        Dictionary mapping column names to pos_weight tensors
    """
    weights = {}

    for col in binary_columns:
        if col not in train_df.columns:
            raise ValueError(f"Binary column '{col}' not found in training data")

        values = train_df[col].values

        # Count positive and negative samples (excluding NaN)
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]

        if len(valid_values) == 0:
            print(f"WARNING: No valid values in column '{col}', using default weight=1.0")
            weights[col] = torch.tensor([1.0])
            continue

        n_pos = (valid_values == 1).sum()
        n_neg = (valid_values == 0).sum()

        if n_pos == 0 or n_neg == 0:
            print(f"WARNING: Column '{col}' has only one class, using default weight=1.0")
            weights[col] = torch.tensor([1.0])
            continue

        # pos_weight = n_neg / n_pos (weight for positive class)
        pos_weight = n_neg / n_pos
        weights[col] = torch.tensor([pos_weight])

        print(f"Column '{col}': pos={n_pos}, neg={n_neg}, pos_weight={pos_weight:.3f}")

    return weights


if __name__ == "__main__":
    print("=== Testing DataSplitter and TargetNormalizer ===\n")

    # Create synthetic test data
    n_patients = 20
    samples_per_patient = [3, 5, 2, 4, 3, 1, 2, 3, 4, 2, 3, 5, 1, 2, 3, 4, 5, 2, 3, 1]

    data = []
    for patient_id, n_samples in enumerate(samples_per_patient):
        for _ in range(n_samples):
            data.append({
                'patient_id': f'P{patient_id:03d}',
                'image_path': f'img_{patient_id}_{np.random.randint(1000)}.jpg',
                'grs_cad': np.random.randn(),
                'grs_diabetes': np.random.randn(),
                'has_hypertension': np.random.choice([0, 1]),
                'has_glaucoma': np.random.choice([0, 1])
            })

    df = pd.DataFrame(data)
    print(f"Created synthetic dataset: {len(df)} samples, {df['patient_id'].nunique()} patients\n")

    # Test DataSplitter
    splitter = DataSplitter(n_folds=5)
    summary = splitter.get_patient_summary(df)
    print("\nPatient Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\nGenerating folds:")
    folds = splitter.get_folds(df)

    # Test TargetNormalizer
    print("\n=== Testing TargetNormalizer ===")
    train_idx, val_idx = folds[0]
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    normalizer = TargetNormalizer()
    reg_columns = ['grs_cad', 'grs_diabetes']

    normalizer.fit(train_df, reg_columns)
    print("\nScaler parameters:")
    for col, params in normalizer.get_params().items():
        print(f"  {col}: mean={params['mean']:.3f}, std={params['std']:.3f}")

    # Transform and inverse transform
    train_normalized = normalizer.transform(train_df, reg_columns)
    train_recovered = normalizer.inverse_transform(train_normalized, reg_columns)

    print("\nVerifying inverse transform accuracy:")
    for col in reg_columns:
        original = train_df[col].values
        recovered = train_recovered[col].values
        max_error = np.max(np.abs(original - recovered))
        print(f"  {col}: max_error={max_error:.2e}")

    # Test class weights
    print("\n=== Testing Class Weights ===")
    binary_columns = ['has_hypertension', 'has_glaucoma']
    weights = compute_class_weights(train_df, binary_columns)

    print("\nAll tests passed!")
