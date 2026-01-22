#!/bin/bash

################################################################################
# RetinalHydraNet Training Script
# This script runs the full training pipeline with cross-validation
################################################################################

set -e  # Exit on error

echo "=================================="
echo "RetinalHydraNet Training Pipeline"
echo "=================================="

# Configuration
CSV_PATH="${1:-data/clinical_data.csv}"
IMG_DIR="${2:-data/images}"
EPOCHS="${3:-20}"

# Column names (modify these according to your dataset)
REG_COLUMNS="${REG_COLUMNS:-grs_cad,grs_diabetes}"
BIN_COLUMNS="${BIN_COLUMNS:-has_hypertension,has_glaucoma}"
PATIENT_ID_COL="${PATIENT_ID_COL:-patient_id}"
IMAGE_PATH_COL="${IMAGE_PATH_COL:-image_path}"

# Model configuration
BACKBONE="${BACKBONE:-vit_base_patch16_224}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.0001}"
N_FOLDS="${N_FOLDS:-5}"

# Check if CSV exists
if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV file not found: $CSV_PATH"
    echo ""
    echo "Usage: $0 <csv_path> <img_dir> [epochs]"
    echo ""
    echo "Example:"
    echo "  $0 data/clinical_data.csv data/images 20"
    echo ""
    echo "Required CSV columns:"
    echo "  - $PATIENT_ID_COL: Patient identifier"
    echo "  - $IMAGE_PATH_COL: Path to image file"
    echo "  - Regression targets: $REG_COLUMNS"
    echo "  - Binary targets: $BIN_COLUMNS"
    exit 1
fi

# Check if image directory exists
if [ ! -d "$IMG_DIR" ]; then
    echo "Error: Image directory not found: $IMG_DIR"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  CSV Path: $CSV_PATH"
echo "  Image Dir: $IMG_DIR"
echo "  Epochs: $EPOCHS"
echo "  Regression Columns: $REG_COLUMNS"
echo "  Binary Columns: $BIN_COLUMNS"
echo "  Backbone: $BACKBONE"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LR"
echo "  N Folds: $N_FOLDS"
echo ""

# Create output directories
mkdir -p experiments/checkpoints
mkdir -p experiments/logs
mkdir -p experiments/saliency_maps

# Run training
echo "Starting training..."
echo ""

python src/train.py \
    --csv "$CSV_PATH" \
    --img_dir "$IMG_DIR" \
    --reg_columns "$REG_COLUMNS" \
    --bin_columns "$BIN_COLUMNS" \
    --patient_id_col "$PATIENT_ID_COL" \
    --image_path_col "$IMAGE_PATH_COL" \
    --backbone "$BACKBONE" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --n_folds "$N_FOLDS" \
    --pretrained \
    --checkpoint_dir experiments/checkpoints \
    --log_dir experiments/logs

echo ""
echo "=================================="
echo "Training Complete!"
echo "=================================="
echo ""
echo "Outputs:"
echo "  - Model checkpoints: experiments/checkpoints/"
echo "  - Training logs: experiments/logs/"
echo "  - Results CSV: experiments/results.csv"
echo "  - CV Summary: experiments/cv_summary.json"
echo ""
echo "To view training curves with TensorBoard:"
echo "  tensorboard --logdir experiments/logs"
echo ""
echo "To run inference on new images:"
echo "  python src/inference.py --mode batch \\"
echo "    --csv <test_csv> \\"
echo "    --img_dir <test_img_dir> \\"
echo "    --model experiments/checkpoints/best_model_fold_0.pth \\"
echo "    --output predictions.csv \\"
echo "    --gradcam \\"
echo "    --gradcam_dir experiments/saliency_maps"
echo ""
