# RetinalHydraNet - Multi-Task Oculomics Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning pipeline for simultaneous prediction of systemic biomarkers and ophthalmic clinical metrics from retinal fundus images. Built with patient-level cross-validation, explainability via GradCAM, and robust data integrity.

---

## 🎯 Key Features

- **Multi-Task Learning**: Simultaneous regression (genetic risk scores) and binary classification (clinical outcomes)
- **Patient-Level Stratification**: Rigorous cross-validation with GroupKFold to prevent data leakage
- **Transfer Learning**: Leverages pretrained ViT/ResNet/EfficientNet from timm
- **Explainability**: GradCAM visualizations for model interpretation
- **Production Ready**: Modular codebase with comprehensive error handling and logging

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/retinalhydranet.git
cd retinalhydranet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

1. **Organize your images**:
```
data/images/
├── fundus_001.jpg
├── fundus_002.jpg
└── ...
```

2. **Create CSV file** (`data/clinical_data.csv`):
```csv
patient_id,image_path,grs_cad,grs_diabetes,has_hypertension,has_glaucoma
P001,fundus_001.jpg,0.75,-0.32,1,0
P001,fundus_002.jpg,0.75,-0.32,1,0
P002,fundus_003.jpg,-0.15,1.24,0,1
```

### Training

```bash
# Quick start (uses defaults)
chmod +x run_experiment.sh
./run_experiment.sh data/clinical_data.csv data/images 20

# Custom configuration
python src/train.py \
    --csv data/clinical_data.csv \
    --img_dir data/images \
    --reg_columns grs_cad,grs_diabetes \
    --bin_columns has_hypertension,has_glaucoma \
    --epochs 50 \
    --batch_size 16 \
    --backbone vit_base_patch16_224 \
    --n_folds 5
```

### Monitoring

```bash
# Launch TensorBoard
tensorboard --logdir experiments/logs

# View at http://localhost:6006
```

### Inference

```bash
# Single image with GradCAM
python src/inference.py \
    --mode single \
    --image data/images/test_image.jpg \
    --model experiments/checkpoints/best_model_fold_0.pth \
    --gradcam

# Batch inference
python src/inference.py \
    --mode batch \
    --csv data/test_data.csv \
    --img_dir data/images \
    --model experiments/checkpoints/best_model_fold_0.pth \
    --output predictions.csv
```

---

## 📁 Project Structure

```
RetinalHydraNet/
├── data/
│   ├── images/              # Retinal fundus images
│   └── clinical_data.csv    # Labels and metadata
├── src/
│   ├── config.py           # Configuration and hyperparameters
│   ├── utils.py            # Data splitting and normalization
│   ├── dataset.py          # PyTorch Dataset with augmentations
│   ├── model.py            # Multi-task model architecture
│   ├── train.py            # Training script with CV
│   └── inference.py        # Inference and GradCAM
├── experiments/
│   ├── checkpoints/        # Saved model weights
│   ├── logs/              # TensorBoard logs
│   ├── saliency_maps/     # GradCAM visualizations
│   ├── results.csv        # Prediction results
│   └── cv_summary.json    # Cross-validation metrics
├── requirements.txt        # Python dependencies
├── run_experiment.sh      # Training automation script
├── verify_phase1.py       # Data pipeline verification
├── PROJECT_BRIEF.md       # Detailed technical documentation
└── README.md             # This file
```

---

## 🏗️ Architecture

### The "Hydra" Model

```
Input Image (3×224×224)
         ↓
   Backbone (ViT/ResNet)
         ↓
   Feature Vector
       /    \
      /      \
Reg Head   Bin Head
    |          |
[Linear]   [Linear]
    ↓          ↓
 GRS_CAD   Hypertension
GRS_Diabetes Glaucoma
```

### Multi-Task Loss

```
L_total = λ_reg · MSE(regression) + λ_bin · BCE(binary)
```

---

## 📊 Data Requirements

### CSV Columns

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `patient_id` | str | Unique patient identifier | "P001" |
| `image_path` | str | Path to image (relative/absolute) | "fundus_001.jpg" |
| `grs_cad` | float | Genetic Risk Score for CAD | 0.75 |
| `grs_diabetes` | float | Genetic Risk Score for Diabetes | -0.32 |
| `has_hypertension` | int | Binary: Has hypertension (0/1) | 1 |
| `has_glaucoma` | int | Binary: Has glaucoma (0/1) | 0 |

### Image Specifications

- **Formats**: `.jpg`, `.png`, `.tiff`
- **Resolution**: Any (automatically resized to 224×224)
- **Color Space**: RGB
- **Modality**: Retinal fundus photography

---

## ⚙️ Configuration

Edit `src/config.py` to customize:

```python
# Model
IMG_SIZE = 224
BACKBONE = "vit_base_patch16_224"  # Options: resnet50, efficientnet_b0
DROPOUT_RATE = 0.3

# Training
BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 20
N_FOLDS = 5

# Loss Weights
LAMBDA_REG = 1.0  # Regression loss weight
LAMBDA_BIN = 1.0  # Binary loss weight
```

---

## 🔬 Key Design Principles

### 1. Patient-Level Stratification
Ensures no patient appears in both training and validation sets, preventing optimistic bias.

```python
splitter = DataSplitter(n_folds=5)
folds = splitter.get_folds(df, patient_id_col='patient_id')
```

### 2. Target Normalization
Regression targets are z-score normalized using statistics from **training set only**.

```python
normalizer = TargetNormalizer()
normalizer.fit(train_df, reg_columns)
train_normalized = normalizer.transform(train_df)
val_normalized = normalizer.transform(val_df)
```

### 3. Class Imbalance Handling
Binary targets use `pos_weight` computed from training data.

```python
weights = compute_class_weights(train_df, bin_columns)
criterion = MultiTaskLoss(bin_pos_weights=weights)
```

---

## 📈 Outputs

After training, you'll find:

1. **Model Checkpoints**: `experiments/checkpoints/best_model_fold_{i}.pth`
2. **Predictions**: `experiments/results.csv` with per-patient predictions
3. **CV Summary**: `experiments/cv_summary.json` with mean ± std metrics
4. **TensorBoard Logs**: `experiments/logs/` for training curves
5. **GradCAM**: `experiments/saliency_maps/` with visual explanations

### Example Results

```json
{
  "mean_val_loss": 0.523,
  "std_val_loss": 0.042,
  "mean_val_mae": 0.315,
  "mean_val_auc": 0.847
}
```

---

## 🛠️ Advanced Usage

### Custom Backbones

```bash
# Use ResNet50 (faster training)
python src/train.py --backbone resnet50 ...

# Use EfficientNet (better accuracy)
python src/train.py --backbone efficientnet_b3 ...

# Use RETFound (domain-specific pretrained)
python src/train.py --backbone retfound_cfp ...
```

### Hyperparameter Tuning

```bash
# Increase model capacity
python src/train.py --dropout 0.5 --lr 5e-5 ...

# Adjust loss weighting
python src/train.py --lambda_reg 2.0 --lambda_bin 1.0 ...
```

### Testing Individual Components

```bash
# Test data pipeline
python src/dataset.py

# Test model architecture
python src/model.py

# Test data splitting
python src/utils.py

# Comprehensive Phase 1 verification
python verify_phase1.py
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python src/train.py --batch_size 8 ...

# Use gradient accumulation (edit train.py)
# Or use a smaller backbone
python src/train.py --backbone resnet50 ...
```

### Poor Performance
- **Check data balance**: Review `compute_class_weights()` output
- **Verify splitting**: Check fold logs for patient overlap
- **Increase training time**: Try 50+ epochs
- **Stronger augmentation**: Edit augmentation pipeline in `dataset.py`

### GradCAM Not Working
```bash
# Install separately
pip install grad-cam

# Check compatibility
python -c "import pytorch_grad_cam; print(pytorch_grad_cam.__version__)"
```

---

## 📚 Documentation

- **[PROJECT_BRIEF.md](PROJECT_BRIEF.md)**: Comprehensive technical documentation
- **[requirements.txt](requirements.txt)**: Full dependency list
- **[run_experiment.sh](run_experiment.sh)**: Automated training script

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📖 Citation

If you use this codebase in your research, please cite:

```bibtex
@software{retinalhydranet2024,
  title={RetinalHydraNet: Multi-Task Learning for Oculomics},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/retinalhydranet}
}
```

---

## 🙏 Acknowledgments

- **timm**: PyTorch Image Models library
- **albumentations**: Fast augmentation library
- **pytorch-grad-cam**: GradCAM implementation

---

## 📧 Contact

For questions, issues, or collaboration:

- **GitHub Issues**: [Open an issue](https://github.com/yourusername/retinalhydranet/issues)
- **Email**: your.email@institution.edu

---

**Built with ❤️ for the Oculomics Research Community**

Last Updated: 2026-01-22 | Version: 1.0
