# RetinalHydraNet - Multi-Task Oculomics Pipeline

## Product Requirements Document (PRD)

**Project Name:** RetinalHydraNet (Multi-Task Oculomics Pipeline)
**Version:** 1.0
**Owner:** Principal Investigator / Lead Developer

---

## 1. Executive Summary

The objective is to develop a deep learning pipeline that ingests retinal fundus photography to simultaneously predict systemic biomarkers (Genetic Risk Scores) and local ophthalmic clinical metrics. The system must handle scarce data (~500 images) using transfer learning, address class imbalance, and ensure rigorous patient-level separation to prevent data leakage.

## 2. Core Objectives

- **Multi-Task Prediction**: A single model must output continuous variables (Regression) and binary variables (Classification).
- **Data Integrity**: Strict patient-level stratification for Cross-Validation (5-Fold).
- **Explainability**: Provide visual attribution (Grad-CAM) to validate biological plausibility of genetic risk predictions.
- **Reproducibility**: Modular codebase with configuration-driven training and deterministic seeding.

## 3. User Stories

- **As a Researcher**, I want to feed a CSV of mixed clinical labels and a folder of images so that I can train a model without manually separating folders.
- **As a Clinician**, I want to see a "saliency map" overlay on the retina so I can understand which features (vessels, disc, macula) drive the genetic risk prediction.
- **As a Data Scientist**, I want to view training curves (Train vs Val Loss) to detect overfitting early.

## 4. Functional Requirements

- **Input Handling**: Support .jpg, .png, and .tiff. CSV parsing with flexible column mapping.
- **Preprocessing**: Z-score normalization for regression targets (fit on train, apply to val).
- **Architecture**: timm backbone (ViT or EfficientNet) with a "Hydra" head (split MLP blocks).
- **Loss Function**: Weighted sum of MSE (Regression) and BCEWithLogits (Classification).
- **Output**: saved model weights (.pth), results.csv with per-patient predictions, and logs/ for TensorBoard.

---

## 5. Technical Stack Specification

### 5.1 Core Frameworks

- **Language**: Python 3.9+
- **Deep Learning**: PyTorch 2.0+ (Lightning optional, but pure PyTorch preferred for explicitness).
- **Model Zoo**: timm (PyTorch Image Models) for access to RETFound/ViT/EfficientNet.
- **Augmentation**: albumentations (Essential for "medical-grade" rigid transformations).

### 5.2 Data & Math

- **Data Manipulation**: pandas (CSV handling), numpy.
- **CV Splitting**: scikit-learn (GroupKFold for patient stratification).
- **Image I/O**: opencv-python-headless (Faster than PIL).

### 5.3 MLOps & Observability

- **Logging**: torch.utils.tensorboard (Standard, no external account needed).
- **Explainability**: pytorch-grad-cam (For visualizing attention).
- **Config Management**: argparse or yaml for hyperparameters.

---

## 6. Implementation Guide

### Phase 1: Project Skeleton & Data Pipeline ✅

**Status:** COMPLETED

#### Task 1.1: Initialize the file structure
```
/RetinalHydraNet
├── data/
│   ├── images/
│   └── clinical_data.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── utils.py
│   ├── train.py
│   └── inference.py
├── experiments/
│   ├── checkpoints/
│   ├── logs/
│   └── saliency_maps/
├── requirements.txt
├── run_experiment.sh
└── README.md
```

#### Task 1.2: Implement src/config.py ✅
- Global variables: IMG_SIZE=224, BATCH_SIZE=16, LR=1e-4, N_FOLDS=5
- Path configuration for data and outputs
- Hyperparameter defaults

#### Task 1.3: Implement src/utils.py ✅
- **DataSplitter class**: Uses GroupKFold on patient_id to prevent leakage
- **TargetNormalizer class**: Fits StandardScaler only on training indices
- **Helper functions**: set_seed(), compute_class_weights()

#### Task 1.4: Implement src/dataset.py ✅
- RetinalDataset class with albumentations support
- Train augmentations: Flip, Rotate, Brightness, Contrast, GaussianBlur
- Val augmentations: Resize, Normalize only
- Verification: Test with synthetic data

---

### Phase 2: Model Architecture ("The Hydra") ✅

**Status:** COMPLETED

#### Task 2.1: Implement src/model.py ✅
- Uses timm.create_model (default: vit_base_patch16_224 or resnet50)
- Modified final layer: Removed default classifier
- **Regression head**: Linear → ReLU → Dropout → Linear(out=num_reg)
- **Binary head**: Linear → ReLU → Dropout → Linear(out=num_bin)

#### Task 2.2: Forward pass returns tuple ✅
- Output: (reg_logits, bin_logits)
- Supports both CNN and ViT backbones
- Includes get_last_conv_layer() for GradCAM

#### Task 2.3: MultiTaskLoss implementation ✅
- Combined loss: Loss = (λ_reg × MSELoss) + (λ_bin × BCEWithLogitsLoss)
- Handles class imbalance with pos_weight
- Returns detailed loss breakdown for logging

---

### Phase 3: Training Logic & Loss ✅

**Status:** COMPLETED

#### Task 3.1: Implement src/train.py ✅
- **Trainer class** with train_epoch() and validate_epoch()
- **Cross-validation loop** with patient-level stratification
- **Loss calculation** with configurable weights
- **pos_weight** computed from training set imbalance
- **Metrics**: MAE for regression, AUC for binary classification

#### Task 3.2: TensorBoard logging ✅
- Logs: Loss/Train, Loss/Val, MAE/Val, AUC/Val per epoch
- Separate logs per fold
- Learning rate tracking

#### Task 3.3: Checkpointing ✅
- Saves best_model_fold_{i}.pth when Val Loss improves
- Includes optimizer state, metrics, and epoch
- Early stopping with configurable patience

---

### Phase 4: Inference & Explainability ✅

**Status:** COMPLETED

#### Task 4.1: Implement src/inference.py ✅
- **RetinalPredictor class** for single and batch inference
- Function: predict_single_image(model, image_path)
- Applies saved scaler inverse transform for real genetic risk values
- Supports batch inference from CSV

#### Task 4.2: Integrate GradCAM ✅
- Targets last convolutional layer (or LayerNorm in ViT)
- Saves heatmap overlay to experiments/saliency_maps/
- Supports visualization for both regression and binary outputs
- Generates comprehensive 3-panel plots (Original, Heatmap, Overlay)

---

### Phase 5: Execution & Validation ✅

**Status:** COMPLETED

#### Task 5.1: Create run_experiment.sh ✅
```bash
./run_experiment.sh data/clinical_data.csv data/images 20
```

#### Task 5.2: Expected outputs ✅
- `experiments/results.csv`: Per-patient predictions for all folds
- `experiments/cv_summary.json`: Cross-validation statistics
- `experiments/checkpoints/best_model_fold_{i}.pth`: Saved models
- `experiments/logs/`: TensorBoard logs

---

## 7. Data Format Requirements

### CSV Structure
Your `clinical_data.csv` should contain the following columns:

| Column | Type | Description | Required |
|--------|------|-------------|----------|
| patient_id | str | Unique patient identifier | Yes |
| image_path | str | Relative or absolute path to image | Yes |
| grs_cad | float | Genetic Risk Score for CAD | Yes (regression) |
| grs_diabetes | float | Genetic Risk Score for Diabetes | Yes (regression) |
| has_hypertension | int (0/1) | Binary: Has hypertension | Yes (binary) |
| has_glaucoma | int (0/1) | Binary: Has glaucoma | Yes (binary) |

**Example:**
```csv
patient_id,image_path,grs_cad,grs_diabetes,has_hypertension,has_glaucoma
P001,fundus_001.jpg,0.75,-0.32,1,0
P001,fundus_002.jpg,0.75,-0.32,1,0
P002,fundus_003.jpg,-0.15,1.24,0,1
```

### Image Requirements
- **Formats**: .jpg, .png, .tiff
- **Resolution**: Any (will be resized to 224×224)
- **Color**: RGB (fundus photography)
- **Location**: Place in `data/images/` directory

---

## 8. Usage Instructions

### Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training
```bash
# Basic training
./run_experiment.sh data/clinical_data.csv data/images 20

# Custom configuration
python src/train.py \
    --csv data/clinical_data.csv \
    --img_dir data/images \
    --reg_columns grs_cad,grs_diabetes \
    --bin_columns has_hypertension,has_glaucoma \
    --backbone vit_base_patch16_224 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.0001 \
    --n_folds 5
```

### Monitoring Training
```bash
# Launch TensorBoard
tensorboard --logdir experiments/logs

# Open browser to http://localhost:6006
```

### Inference
```bash
# Single image prediction with GradCAM
python src/inference.py \
    --mode single \
    --image data/images/test_image.jpg \
    --model experiments/checkpoints/best_model_fold_0.pth \
    --gradcam \
    --gradcam_dir experiments/saliency_maps

# Batch inference
python src/inference.py \
    --mode batch \
    --csv data/test_data.csv \
    --img_dir data/images \
    --model experiments/checkpoints/best_model_fold_0.pth \
    --output predictions.csv \
    --gradcam
```

---

## 9. Key Design Decisions

### Patient-Level Stratification
- **Why**: Prevents data leakage when same patient has multiple images
- **Implementation**: GroupKFold with patient_id as grouping variable
- **Verification**: Automated check ensures no patient overlap between train/val

### Target Normalization
- **Why**: Genetic risk scores have different scales; normalization stabilizes training
- **Implementation**: StandardScaler fit only on training fold
- **Critical**: Must inverse transform predictions for interpretation

### Multi-Task Loss
- **Why**: Single model learns shared representations for related tasks
- **Implementation**: Weighted sum of MSE (regression) and BCE (binary)
- **Tuning**: λ_reg and λ_bin can be adjusted based on task importance

### GradCAM for Explainability
- **Why**: Validates that model focuses on clinically relevant features
- **Implementation**: Targets last attention/conv layer
- **Interpretation**: Heatmaps should highlight vessels, optic disc, macula

---

## 10. Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
python src/train.py --batch_size 8 ...

# Use smaller backbone
python src/train.py --backbone resnet50 ...
```

### Poor Performance
- Check class imbalance (pos_weight should adjust automatically)
- Verify patient-level splitting (check fold logs)
- Increase training epochs (default: 20, try 50+)
- Use stronger augmentation

### Missing Dependencies
```bash
# Install GradCAM separately if needed
pip install grad-cam

# Verify PyTorch installation
python -c "import torch; print(torch.__version__)"
```

---

## 11. Citation

If you use this codebase, please cite:

```bibtex
@software{retinalhydranet2024,
  title={RetinalHydraNet: Multi-Task Learning for Oculomics},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/retinalhydranet}
}
```

---

## 12. License

MIT License - See LICENSE file for details

---

## 13. Contact

For questions or issues:
- GitHub Issues: [Project Issues Page]
- Email: your.email@institution.edu

---

**Last Updated**: 2026-01-22
**Version**: 1.0
**Status**: Production Ready
