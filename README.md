# GalaxEye Space AI - Binary Change Detection

**Technical Assignment: Binary pixel-wise change detection on co-registered EO and SAR satellite imagery.**

A deep learning pipeline for detecting changes (damage/destruction) in satellite imagery using a **Concat U-Net** architecture with a pretrained **ResNet34** encoder. Handles multi-modal EO (3-channel RGB) and SAR (1-channel grayscale) satellite imagery.

---

## Model Weights

**Download the trained model weights:**
> 🔗 **[best_model.pth — Google Drive](https://drive.google.com/file/d/1R1Bkz65YdRLPPGnUyMjxmjZqmgJr5obz/view?usp=drive_link)**

Place the downloaded file at `weights/best_model.pth` before running evaluation or inference.

---

## Architecture

**Concat U-Net with ResNet34 Encoder**
- Pre-event (EO, 3ch) and post-event (SAR, 1ch) images concatenated → 4-channel input
- ResNet34 encoder (ImageNet pretrained) for feature extraction
- U-Net decoder for pixel-wise binary change prediction
- Combined BCE + Dice loss with pos_weight=5 for class imbalance
- Total parameters: 24,439,505

> **Note:** A Siamese U-Net variant is also implemented (`SiameseUNet` in `src/model.py`) for use when pre/post images have equal channel counts. The pipeline auto-selects the appropriate architecture based on input channels.

## Results

### Test Set Metrics

| Metric    | Value  |
|-----------|--------|
| **IoU**       | 0.0156 |
| **Precision** | 0.0159 |
| **Recall**    | 0.4158 |
| F1 Score  | 0.0307 |
| Accuracy  | 0.8027 |

### Validation Set Metrics

| Metric    | Value  |
|-----------|--------|
| **IoU**       | 0.1895 |
| **Precision** | 0.2561 |
| **Recall**    | 0.4218 |
| F1 Score  | 0.3187 |
| Accuracy  | 0.9604 |

### Summary

- **Architecture:** Concat U-Net (ResNet34 encoder, ImageNet pretrained)
- **Training:** 2781 samples, 8 epochs, CPU, ~110 min
- **Optimal Threshold:** 0.55
- **Loss:** 0.3×BCE + 0.7×Dice (pos_weight=5.0)

*Performance would improve further with GPU training, more epochs, and advanced techniques (attention mechanisms, TTA, focal loss).*

## Dataset

**Source:** [HuggingFace - doron333/change-detection-dataset](https://huggingface.co/datasets/doron333/change-detection-dataset)

| Split | Samples | Structure |
|-------|---------|-----------|
| Train | 2781    | pre-event (3ch EO) / post-event (1ch SAR) / target |
| Val   | 334     | pre-event (3ch EO) / post-event (1ch SAR) / target |
| Test  | 77      | pre-event (3ch EO) / post-event (1ch SAR) / target |

**Label Remapping:**
- 0 (Background) → 0 (No Change)
- 1 (Intact) → 0 (No Change)
- 2 (Damaged) → 1 (Change)
- 3 (Destroyed) → 1 (Change)

**Class Distribution:** ~97.6% no-change, ~2.4% change (severe imbalance)

## Project Structure

```
Galaxeye_AI/
├── src/
│   ├── __init__.py          # Package init
│   ├── dataset.py           # Dataset loading, augmentation, EO/SAR fusion
│   ├── model.py             # SiameseUNet + ConcatUNet architectures
│   ├── losses.py            # BCEWithLogits + Dice loss
│   ├── metrics.py           # IoU, F1, Precision, Recall, confusion matrix
│   ├── utils.py             # Config, seeding, checkpointing, early stopping
│   └── visualize.py         # Qualitative plots, confusion matrices, curves
├── train.py                 # Training script with early stopping & threshold tuning
├── evaluate.py              # Evaluation on val/test with qualitative outputs
├── inference.py             # Single image pair inference
├── download_dataset.py      # Dataset downloader from HuggingFace
├── config.yaml              # All hyperparameters and paths
├── requirements.txt         # Python dependencies
├── notebooks/
│   └── exploration.ipynb    # Dataset analysis notebook
├── report/
│   ├── generate_report.py   # PDF report generator
│   ├── Technical_Report.pdf # Generated technical report
│   └── Time_Resource_Log.txt# Time & resource log
├── weights/
│   └── best_model.pth       # Best model checkpoint (download link above)
├── outputs/
│   ├── evaluation_results.json
│   ├── training_results.json
│   ├── training_log.txt
│   ├── plots/               # Training curves, confusion matrices
│   └── qualitative/         # Prediction visualizations (val & test)
└── README.md
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
```bash
python download_dataset.py
```

### 3. Download Model Weights
Download `best_model.pth` from the link above and place in `weights/`.

### 4. Train (optional — retrain from scratch)
```bash
python train.py --epochs 30 --batch-size 8
```

### 5. Evaluate
```bash
python evaluate.py --split both
```

### 6. Single Image Inference
```bash
python inference.py --pre path/to/pre.png --post path/to/post.png
```

### 7. Generate Report
```bash
python report/generate_report.py
```

## Key Design Decisions

1. **EO/SAR Fusion:** Channel concatenation (3+1=4ch input) — simple, effective, and handles mixed modalities automatically.
2. **Class Imbalance:** pos_weight=5.0 in BCE loss + Dice loss component (weighted 0.7) for geometric sensitivity to minority class.
3. **Threshold Tuning:** Automatic search over [0.1, 0.9] to find optimal binary threshold for F1.
4. **Data Augmentation:** Horizontal/vertical flip, random rotate, shift-scale-rotate, blur, brightness/contrast — applied consistently to pre+post pairs.
5. **Architecture Selection:** Auto-selects Siamese U-Net (equal channels) vs Concat U-Net (mixed channels) based on input data.

## Potential Improvements

- Train with more epochs on GPU for deeper convergence
- Use attention-based fusion (e.g., cross-attention between EO and SAR features)
- Multi-scale feature aggregation (FPN decoder)
- Test-time augmentation (TTA)
- Focal loss for hard example mining
- Pre-train on larger remote sensing datasets
- Domain adaptation techniques for EO→SAR transfer

## References

- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," 2015
- Daudt et al., "Fully Convolutional Siamese Networks for Change Detection," 2018
- segmentation_models_pytorch: https://github.com/qubvel/segmentation_models.pytorch
