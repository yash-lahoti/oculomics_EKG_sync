"""
Inference and Explainability for RetinalHydraNet
Includes GradCAM visualization for model interpretation
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch-grad-cam not available. GradCAM visualization will be disabled.")

import config
from model import RetinalHydraNet
from dataset import get_val_transforms
from utils import TargetNormalizer


class RetinalPredictor:
    """Handles inference and explainability for retinal images"""

    def __init__(
        self,
        model_path: str,
        normalizer_path: Optional[str] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            model_path: Path to saved model checkpoint
            normalizer_path: Path to saved TargetNormalizer (pickle)
            device: Device to run inference on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        # Extract model info from checkpoint if available
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Infer model configuration from state dict
        # This is a simplified approach - in production, save config with model
        self.model = self._build_model_from_checkpoint(state_dict)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")

        # Load normalizer if provided
        self.normalizer = None
        if normalizer_path and os.path.exists(normalizer_path):
            with open(normalizer_path, 'rb') as f:
                self.normalizer = pickle.load(f)
            print(f"Normalizer loaded from: {normalizer_path}")

        # Setup transforms
        self.transform = get_val_transforms()

        # Setup GradCAM if available
        self.gradcam = None
        if GRADCAM_AVAILABLE:
            target_layer = self.model.get_last_conv_layer()
            self.gradcam = GradCAM(model=self.model, target_layers=[target_layer])
            print(f"GradCAM initialized with layer: {target_layer.__class__.__name__}")

    def _build_model_from_checkpoint(self, state_dict: Dict) -> RetinalHydraNet:
        """Infer model architecture from checkpoint"""
        # Count output dimensions from final layers
        reg_head_key = 'reg_head.6.weight'  # Last linear layer of reg_head
        bin_head_key = 'bin_head.6.weight'  # Last linear layer of bin_head

        num_reg = state_dict[reg_head_key].shape[0] if reg_head_key in state_dict else 2
        num_bin = state_dict[bin_head_key].shape[0] if bin_head_key in state_dict else 2

        # Create model with inferred dimensions
        model = RetinalHydraNet(
            num_reg_targets=num_reg,
            num_bin_targets=num_bin,
            backbone=config.BACKBONE,
            pretrained=False  # Weights will be loaded from checkpoint
        )

        return model

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Load and preprocess an image

        Args:
            image_path: Path to image file

        Returns:
            Tuple of (preprocessed_tensor, original_rgb_image)
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transformed = self.transform(image=image_rgb)
        image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

        return image_tensor, image_rgb

    @torch.no_grad()
    def predict(self, image_path: str) -> Dict:
        """
        Make prediction on a single image

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with predictions
        """
        # Preprocess image
        image_tensor, image_rgb = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        # Forward pass
        reg_pred, bin_pred = self.model(image_tensor)

        # Convert to numpy
        reg_pred = reg_pred.cpu().numpy()[0]
        bin_pred = torch.sigmoid(torch.tensor(bin_pred.cpu().numpy()[0])).numpy()

        # Denormalize regression predictions if normalizer is available
        if self.normalizer:
            # Create temporary dataframe for inverse transform
            temp_df = pd.DataFrame([reg_pred], columns=self.normalizer.reg_columns)
            reg_pred_denorm = self.normalizer.inverse_transform(temp_df)
            reg_pred = reg_pred_denorm.iloc[0].values

        results = {
            'regression': reg_pred,
            'binary': bin_pred,
            'image_path': image_path
        }

        return results

    def generate_gradcam(
        self,
        image_path: str,
        target_index: int = 0,
        task: str = 'regression',
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Generate GradCAM visualization

        Args:
            image_path: Path to image file
            target_index: Index of the target output to visualize
            task: 'regression' or 'binary'
            save_path: Optional path to save visualization

        Returns:
            GradCAM visualization as numpy array
        """
        if not GRADCAM_AVAILABLE:
            raise RuntimeError("pytorch-grad-cam is not installed. Install with: pip install grad-cam")

        if self.gradcam is None:
            raise RuntimeError("GradCAM not initialized")

        # Preprocess image
        image_tensor, image_rgb = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)

        # Resize original image for overlay
        image_resized = cv2.resize(image_rgb, (config.IMG_SIZE, config.IMG_SIZE))
        image_resized_float = image_resized.astype(np.float32) / 255.0

        # Define target for GradCAM
        class ModelOutputTarget:
            def __init__(self, task, target_index):
                self.task = task
                self.target_index = target_index

            def __call__(self, model_output):
                if self.task == 'regression':
                    return model_output[0][:, self.target_index]  # reg_pred
                else:  # binary
                    return model_output[1][:, self.target_index]  # bin_pred

        targets = [ModelOutputTarget(task, target_index)]

        # Generate GradCAM
        grayscale_cam = self.gradcam(input_tensor=image_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Create visualization
        visualization = show_cam_on_image(image_resized_float, grayscale_cam, use_rgb=True)

        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Create a comprehensive visualization
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(image_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')

            axes[1].imshow(grayscale_cam, cmap='jet')
            axes[1].set_title(f'GradCAM Heatmap ({task} #{target_index})')
            axes[1].axis('off')

            axes[2].imshow(visualization)
            axes[2].set_title('Overlay')
            axes[2].axis('off')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"GradCAM visualization saved to: {save_path}")

        return visualization

    def predict_with_explanation(
        self,
        image_path: str,
        save_dir: Optional[str] = None
    ) -> Dict:
        """
        Make prediction and generate GradCAM visualizations for all outputs

        Args:
            image_path: Path to image file
            save_dir: Directory to save visualizations

        Returns:
            Dictionary with predictions and visualization paths
        """
        # Make prediction
        results = self.predict(image_path)

        # Generate GradCAM visualizations if available
        if GRADCAM_AVAILABLE and self.gradcam is not None and save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            base_name = Path(image_path).stem

            # Generate for each regression output
            for i in range(len(results['regression'])):
                save_path = save_dir / f"{base_name}_gradcam_reg_{i}.png"
                self.generate_gradcam(image_path, target_index=i, task='regression', save_path=str(save_path))

            # Generate for each binary output
            for i in range(len(results['binary'])):
                save_path = save_dir / f"{base_name}_gradcam_bin_{i}.png"
                self.generate_gradcam(image_path, target_index=i, task='binary', save_path=str(save_path))

        return results


def batch_inference(
    csv_path: str,
    img_dir: str,
    model_path: str,
    output_path: str,
    normalizer_path: Optional[str] = None,
    generate_gradcam: bool = False,
    gradcam_dir: Optional[str] = None,
    image_path_col: str = config.IMAGE_PATH_COL
):
    """
    Run inference on multiple images from a CSV file

    Args:
        csv_path: Path to CSV with image paths
        img_dir: Directory containing images
        model_path: Path to model checkpoint
        output_path: Path to save results CSV
        normalizer_path: Path to saved normalizer
        generate_gradcam: Whether to generate GradCAM visualizations
        gradcam_dir: Directory to save GradCAM outputs
        image_path_col: Column name for image paths
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} samples from {csv_path}")

    # Create predictor
    predictor = RetinalPredictor(model_path, normalizer_path)

    # Run predictions
    results = []

    for idx, row in df.iterrows():
        image_path = row[image_path_col]

        # Handle relative paths
        if not os.path.isabs(image_path):
            image_path = os.path.join(img_dir, image_path)

        print(f"Processing {idx+1}/{len(df)}: {image_path}")

        try:
            # Make prediction
            pred = predictor.predict(image_path)

            # Generate GradCAM if requested
            if generate_gradcam and gradcam_dir:
                predictor.predict_with_explanation(image_path, save_dir=gradcam_dir)

            # Store results
            result_row = {
                'image_path': row[image_path_col],
            }

            # Add regression predictions
            for i, val in enumerate(pred['regression']):
                result_row[f'reg_pred_{i}'] = val

            # Add binary predictions
            for i, val in enumerate(pred['binary']):
                result_row[f'bin_pred_{i}'] = val

            results.append(result_row)

        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            continue

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='RetinalHydraNet Inference')

    parser.add_argument('--mode', type=str, choices=['single', 'batch'], default='single',
                        help='Inference mode')
    parser.add_argument('--image', type=str, help='Path to single image (for single mode)')
    parser.add_argument('--csv', type=str, help='Path to CSV file (for batch mode)')
    parser.add_argument('--img_dir', type=str, help='Image directory (for batch mode)')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--normalizer', type=str, help='Path to normalizer pickle file')
    parser.add_argument('--output', type=str, help='Output path for results')
    parser.add_argument('--gradcam', action='store_true', help='Generate GradCAM visualizations')
    parser.add_argument('--gradcam_dir', type=str, default=str(config.SALIENCY_DIR),
                        help='Directory for GradCAM outputs')
    parser.add_argument('--image_path_col', type=str, default=config.IMAGE_PATH_COL,
                        help='Image path column name in CSV')

    args = parser.parse_args()

    if args.mode == 'single':
        if not args.image:
            parser.error("--image is required for single mode")

        # Single image inference
        predictor = RetinalPredictor(args.model, args.normalizer)

        results = predictor.predict_with_explanation(
            image_path=args.image,
            save_dir=args.gradcam_dir if args.gradcam else None
        )

        print("\n=== Prediction Results ===")
        print(f"Image: {args.image}")
        print(f"Regression outputs: {results['regression']}")
        print(f"Binary outputs: {results['binary']}")

    elif args.mode == 'batch':
        if not args.csv or not args.img_dir or not args.output:
            parser.error("--csv, --img_dir, and --output are required for batch mode")

        # Batch inference
        batch_inference(
            csv_path=args.csv,
            img_dir=args.img_dir,
            model_path=args.model,
            output_path=args.output,
            normalizer_path=args.normalizer,
            generate_gradcam=args.gradcam,
            gradcam_dir=args.gradcam_dir,
            image_path_col=args.image_path_col
        )


if __name__ == "__main__":
    main()
