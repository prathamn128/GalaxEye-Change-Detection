#!/usr/bin/env python3
"""
Generate Technical Report PDF for GalaxEye Change Detection assignment.
Uses fpdf2 to create a professional PDF report.
"""

import os
import json
import time
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("Install fpdf2: pip install fpdf2")
    exit(1)


class TechnicalReport(FPDF):
    """Custom PDF report with headers and footers."""

    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.cell(0, 8, "GalaxEye Space AI - Technical Report", 0, 1, "C")
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

    def chapter_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(230, 230, 250)
        self.cell(0, 10, title, 0, 1, "L", fill=True)
        self.ln(3)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, title, 0, 1, "L")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def add_image_if_exists(self, path, w=170):
        if os.path.exists(path):
            try:
                self.image(path, x=20, w=w)
                self.ln(5)
            except Exception as e:
                self.body_text(f"[Image not available: {path}]")


def load_results():
    """Load evaluation results if available."""
    path = "outputs/evaluation_results.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


def load_training_log():
    """Load training log if available."""
    path = "outputs/training_log.txt"
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None


def generate_report():
    """Generate the full technical report PDF."""
    pdf = TechnicalReport()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    results = load_results()
    training_log = load_training_log()

    # ---- Title Page ----
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 15, "Binary Change Detection", 0, 1, "C")
    pdf.cell(0, 15, "on Satellite Imagery", 0, 1, "C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 14)
    pdf.cell(0, 10, "GalaxEye Space AI Research Intern", 0, 1, "C")
    pdf.cell(0, 10, "Technical Assignment", 0, 1, "C")
    pdf.ln(20)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Date: {time.strftime('%B %d, %Y')}", 0, 1, "C")
    pdf.cell(0, 8, "Framework: PyTorch + Segmentation Models PyTorch", 0, 1, "C")

    # ---- 1. Abstract ----
    pdf.add_page()
    pdf.chapter_title("1. Abstract")
    pdf.body_text(
        "This report presents a deep learning solution for binary pixel-wise change detection "
        "on co-registered Electro-Optical (EO) and Synthetic Aperture Radar (SAR) satellite imagery. "
        "We implement a Siamese U-Net architecture with a pretrained ResNet34 encoder that processes "
        "pre-event and post-event image pairs through shared encoder branches, fusing features at "
        "multiple scales via concatenation and 1x1 convolution reduction. The model performs binary "
        "classification of each pixel as 'No Change' (background + intact) or 'Change' (damaged + destroyed), "
        "following a 4-to-2 class remapping scheme. Training employs a combined BCEWithLogitsLoss and "
        "Dice Loss with positive class weighting to address significant class imbalance. The pipeline "
        "includes mixed-precision training, cosine annealing learning rate scheduling, early stopping, "
        "and automatic threshold tuning on the validation set."
    )

    # ---- 2. Literature Survey ----
    pdf.chapter_title("2. Literature Survey")

    pdf.section_title("2.1 Change Detection in Remote Sensing")
    pdf.body_text(
        "Change detection (CD) in remote sensing aims to identify differences in the state of a "
        "geographic area by analyzing images acquired at different times. Traditional approaches relied "
        "on image differencing, PCA, and threshold-based methods. Deep learning has dramatically improved "
        "CD performance, with CNN-based methods achieving state-of-the-art results."
    )

    pdf.section_title("2.2 U-Net and Encoder-Decoder Architectures")
    pdf.body_text(
        "U-Net (Ronneberger et al., 2015) introduced skip connections between encoder and decoder, "
        "enabling precise localization crucial for pixel-wise segmentation. Pretrained encoders "
        "(ResNet, EfficientNet) provide strong feature extraction from ImageNet initialization."
    )

    pdf.section_title("2.3 Siamese Networks for Change Detection")
    pdf.body_text(
        "Daudt et al. (2018) introduced Siamese architectures for CD, where shared-weight encoders "
        "process bi-temporal images independently, and feature differences drive the decoder. This "
        "approach naturally handles temporal changes while maintaining spatial correspondence. Our "
        "implementation uses concatenation-based fusion with learned reduction via 1x1 convolutions."
    )

    pdf.section_title("2.4 EO-SAR Fusion")
    pdf.body_text(
        "Multi-sensor fusion of optical (EO) and radar (SAR) imagery provides complementary information. "
        "EO captures spectral reflectance while SAR provides structural/backscatter information invariant "
        "to cloud cover and illumination. Our approach handles variable channel counts by concatenating "
        "pre+post images along the channel dimension, automatically adapting to 3+3 (EO), 1+1 (SAR), "
        "or mixed sensor inputs."
    )

    # ---- 3. Methodology ----
    pdf.add_page()
    pdf.chapter_title("3. Methodology")

    pdf.section_title("3.1 Dataset")
    pdf.body_text(
        "We use the change-detection-dataset from HuggingFace (doron333/change-detection-dataset), "
        "containing co-registered pre/post satellite image pairs with pixel-wise annotations. "
        "The dataset is split into train, validation, and test sets.\n\n"
        "Label Remapping:\n"
        "  0 (Background) -> 0 (No Change)\n"
        "  1 (Intact) -> 0 (No Change)\n"
        "  2 (Damaged) -> 1 (Change)\n"
        "  3 (Destroyed) -> 1 (Change)"
    )

    pdf.section_title("3.2 Model Architecture")
    pdf.body_text(
        "Siamese U-Net with ResNet34 encoder:\n"
        "1. Shared encoder: Pretrained ResNet34 processes pre and post images independently\n"
        "2. Feature fusion: At each encoder level, features from both branches are concatenated "
        "and reduced via 1x1 Conv + BN + ReLU\n"
        "3. Decoder: Standard U-Net decoder with skip connections\n"
        "4. Output: Single-channel logit map (binary change probability after sigmoid)"
    )

    pdf.section_title("3.3 Loss Function")
    pdf.body_text(
        "Combined loss = 0.5 * BCEWithLogitsLoss + 0.5 * DiceLoss\n\n"
        "BCEWithLogitsLoss uses pos_weight=3.0 to upweight the minority 'change' class. "
        "Dice Loss provides a differentiable approximation of the IoU metric, helping with "
        "class imbalance by focusing on overlap quality."
    )

    pdf.section_title("3.4 Training Details")
    pdf.body_text(
        "- Optimizer: AdamW (lr=0.001, weight_decay=0.0001)\n"
        "- Scheduler: Cosine Annealing (T_max=50, eta_min=1e-6)\n"
        "- Image size: 256x256\n"
        "- Batch size: 8\n"
        "- Augmentations: HorizontalFlip, VerticalFlip, RandomRotate90, ShiftScaleRotate, "
        "GaussianBlur, RandomBrightnessContrast\n"
        "- Mixed precision: Enabled on GPU, disabled on CPU\n"
        "- Early stopping: Patience=10 epochs on validation F1\n"
        "- Gradient clipping: max_norm=1.0\n"
        "- Threshold tuning: Grid search [0.1, 0.9] step 0.05 on validation F1"
    )

    # ---- 4. Results ----
    pdf.add_page()
    pdf.chapter_title("4. Results")

    if results:
        pdf.section_title("4.1 Quantitative Results")

        # Results table
        pdf.set_font("Helvetica", "B", 10)
        col_w = 45
        pdf.cell(col_w, 8, "Metric", 1, 0, "C")
        if "val" in results:
            pdf.cell(col_w, 8, "Validation", 1, 0, "C")
        if "test" in results:
            pdf.cell(col_w, 8, "Test", 1, 0, "C")
        pdf.ln()

        pdf.set_font("Helvetica", "", 10)
        for metric in ["iou", "precision", "recall", "f1", "accuracy"]:
            pdf.cell(col_w, 7, metric.upper(), 1, 0, "C")
            if "val" in results:
                val = results["val"].get(metric, 0)
                pdf.cell(col_w, 7, f"{val:.4f}", 1, 0, "C")
            if "test" in results:
                val = results["test"].get(metric, 0)
                pdf.cell(col_w, 7, f"{val:.4f}", 1, 0, "C")
            pdf.ln()
        pdf.ln(5)
    else:
        pdf.body_text("[Results will be populated after training and evaluation.]")

    pdf.section_title("4.2 Training Curves")
    pdf.add_image_if_exists("outputs/plots/training_curves.png")

    pdf.section_title("4.3 Confusion Matrices")
    pdf.add_image_if_exists("outputs/plots/confusion_matrix_val.png", w=80)
    pdf.add_image_if_exists("outputs/plots/confusion_matrix_test.png", w=80)

    pdf.section_title("4.4 Qualitative Results")
    pdf.body_text("Sample predictions showing pre-event, post-event, ground truth, and model prediction:")
    for split in ["val", "test"]:
        qual_dir = f"outputs/qualitative/{split}"
        if os.path.exists(qual_dir):
            images = sorted(Path(qual_dir).glob("*.png"))[:3]
            for img_path in images:
                pdf.add_image_if_exists(str(img_path), w=170)

    # ---- 5. Future Work ----
    pdf.add_page()
    pdf.chapter_title("5. Future Work")
    pdf.body_text(
        "1. Attention mechanisms: Integrate spatial and channel attention (CBAM, SE blocks) "
        "for better feature discrimination.\n\n"
        "2. Transformer-based models: Explore ChangeFormer or BIT (Binary change detection "
        "with Transformers) for long-range dependency modeling.\n\n"
        "3. Multi-scale fusion: Implement Feature Pyramid Networks (FPN) decoder for better "
        "multi-scale change detection.\n\n"
        "4. Advanced augmentation: CutMix, MixUp, and Mosaic augmentation adapted for "
        "bi-temporal image pairs.\n\n"
        "5. Semi-supervised learning: Leverage unlabeled satellite imagery via consistency "
        "regularization.\n\n"
        "6. Cross-sensor generalization: Train on mixed EO+SAR with sensor-specific adapters "
        "for better domain transfer.\n\n"
        "7. Post-processing: CRF refinement and morphological operations to improve boundary "
        "precision."
    )

    # ---- 6. Conclusion ----
    pdf.chapter_title("6. Conclusion")
    pdf.body_text(
        "We presented a complete pipeline for binary change detection on satellite imagery using "
        "a Siamese U-Net with pretrained ResNet34 encoder. The system handles both EO and SAR "
        "inputs through automatic channel fusion, addresses class imbalance via weighted loss "
        "functions, and includes comprehensive training infrastructure with mixed precision, "
        "early stopping, and threshold optimization. The modular codebase supports easy "
        "experimentation with different encoders, loss functions, and augmentation strategies."
    )

    # ---- 7. Time & Resource Log ----
    pdf.chapter_title("7. Time and Resource Log")

    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(80, 8, "Phase", 1, 0, "C")
    pdf.cell(40, 8, "Time", 1, 0, "C")
    pdf.cell(70, 8, "Resources", 1, 0, "C")
    pdf.ln()

    log_entries = [
        ("Dataset Analysis", "~30 min", "CPU, matplotlib"),
        ("Model Implementation", "~2 hours", "PyTorch, SMP library"),
        ("Training Pipeline", "~1 hour", "PyTorch, AMP"),
        ("Training Execution", "Variable", "CPU/GPU"),
        ("Evaluation & Tuning", "~30 min", "CPU"),
        ("Report & Documentation", "~1 hour", "fpdf2, markdown"),
    ]

    pdf.set_font("Helvetica", "", 10)
    for phase, t, res in log_entries:
        pdf.cell(80, 7, phase, 1, 0, "L")
        pdf.cell(40, 7, t, 1, 0, "C")
        pdf.cell(70, 7, res, 1, 0, "L")
        pdf.ln()

    if training_log:
        pdf.ln(5)
        pdf.section_title("Training Log")
        pdf.set_font("Courier", "", 8)
        for line in training_log.split("\n")[:30]:
            pdf.cell(0, 4, line[:100], 0, 1)

    # ---- References ----
    pdf.add_page()
    pdf.chapter_title("References")
    refs = [
        "1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.",
        "2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.",
        "3. Daudt, R.C., Le Saux, B., & Boulch, A. (2018). Fully Convolutional Siamese Networks for Change Detection. ICIP.",
        "4. Chen, H., & Shi, Z. (2020). A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection. Remote Sensing.",
        "5. Iakubovskii, P. (2019). Segmentation Models PyTorch. GitHub.",
        "6. Milletari, F., Navab, N., & Ahmadi, S.A. (2016). V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation. 3DV.",
    ]
    for ref in refs:
        pdf.body_text(ref)

    # Save
    os.makedirs("report", exist_ok=True)
    output_path = "report/Technical_Report.pdf"
    pdf.output(output_path)
    print(f"[REPORT] Technical report saved to {output_path}")

    # Also save time/resource log as text
    log_text = "GalaxEye Change Detection - Time & Resource Log\n"
    log_text += "=" * 50 + "\n\n"
    for phase, t, res in log_entries:
        log_text += f"{phase:<30} {t:<15} {res}\n"
    if training_log:
        log_text += f"\n{'='*50}\nTraining Log:\n{training_log}\n"

    with open("report/Time_Resource_Log.txt", "w") as f:
        f.write(log_text)
    print("[REPORT] Time/Resource log saved to report/Time_Resource_Log.txt")


if __name__ == "__main__":
    generate_report()
