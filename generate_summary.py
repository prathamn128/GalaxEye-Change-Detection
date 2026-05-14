#!/usr/bin/env python3
"""Generate a comprehensive project summary PDF."""
from fpdf import FPDF

class SummaryPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "GalaxEye Change Detection - Project Summary", 0, 0, "C")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_fill_color(25, 60, 120)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, f"  {title}", 0, 1, "L", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(25, 60, 120)
        self.cell(0, 8, title, 0, 1, "L")
        self.set_text_color(0, 0, 0)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.multi_cell(0, 5.5, "  - " + text)

    def key_value(self, key, value):
        self.set_font("Helvetica", "B", 10)
        self.cell(55, 6, key + ":", 0, 0)
        self.set_font("Helvetica", "", 10)
        self.cell(0, 6, str(value), 0, 1)

pdf = SummaryPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ─── TITLE PAGE ───
pdf.add_page()
pdf.ln(40)
pdf.set_font("Helvetica", "B", 28)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 15, "GalaxEye Space AI", 0, 1, "C")
pdf.set_font("Helvetica", "", 20)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 12, "Binary Change Detection", 0, 1, "C")
pdf.cell(0, 12, "Project Summary", 0, 1, "C")
pdf.ln(10)
pdf.set_draw_color(25, 60, 120)
pdf.set_line_width(0.8)
pdf.line(60, pdf.get_y(), 150, pdf.get_y())
pdf.ln(10)
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, "A complete guide to the project - from problem to solution", 0, 1, "C")
pdf.ln(5)
pdf.cell(0, 8, "Architecture: Concat U-Net (ResNet34 Encoder)", 0, 1, "C")
pdf.cell(0, 8, "Framework: PyTorch + Segmentation Models PyTorch", 0, 1, "C")

# ─── 1. PROBLEM STATEMENT ───
pdf.add_page()
pdf.section_title("1. What is This Project About?")
pdf.body_text(
    "This project solves a BINARY CHANGE DETECTION problem for satellite imagery. "
    "Given two satellite images of the same location taken at different times - one BEFORE "
    "an event (like a disaster) and one AFTER - the goal is to automatically identify "
    "WHICH PIXELS have changed (damaged/destroyed buildings) and which have not."
)
pdf.body_text(
    "Think of it like a 'spot the difference' game, but with satellite photos and using "
    "deep learning to do it automatically at scale."
)

pdf.sub_title("The Challenge")
pdf.body_text(
    "This is not a simple problem because:"
)
pdf.bullet("The 'before' image is an OPTICAL (EO) image with 3 RGB channels (like a normal photo)")
pdf.bullet("The 'after' image is a SAR (radar) image with only 1 grayscale channel (looks very different)")
pdf.bullet("Only ~2.4% of pixels are actually 'changed' - extreme class imbalance")
pdf.bullet("The model must work across different geographic scenes it has never seen before")
pdf.ln(3)

pdf.sub_title("Input and Output")
pdf.body_text(
    "INPUT: A pair of satellite images (pre-event EO + post-event SAR) of the same location.\n"
    "OUTPUT: A binary mask where each pixel is either 0 (No Change) or 1 (Change/Damage)."
)

# ─── 2. DATASET ───
pdf.add_page()
pdf.section_title("2. The Dataset")

pdf.sub_title("Source")
pdf.body_text("HuggingFace: doron333/change-detection-dataset")

pdf.sub_title("Structure")
pdf.set_font("Helvetica", "", 10)
w = 60
pdf.set_font("Helvetica", "B", 10)
pdf.set_fill_color(220, 230, 245)
pdf.cell(w, 7, "Split", 1, 0, "C", fill=True)
pdf.cell(w, 7, "Samples", 1, 0, "C", fill=True)
pdf.cell(w, 7, "Purpose", 1, 1, "C", fill=True)
pdf.set_font("Helvetica", "", 10)
for split, count, purpose in [("Train", "2,781", "Model training"), ("Validation", "334", "Tuning & early stopping"), ("Test", "77", "Final evaluation")]:
    pdf.cell(w, 7, split, 1, 0, "C")
    pdf.cell(w, 7, count, 1, 0, "C")
    pdf.cell(w, 7, purpose, 1, 1, "C")
pdf.ln(3)

pdf.sub_title("Each Sample Contains 3 Images")
pdf.bullet("pre-event/ : Optical (EO) satellite image, 3 channels (RGB), shows buildings and terrain clearly")
pdf.bullet("post-event/ : SAR radar image, 1 channel (grayscale), taken after a disaster event")
pdf.bullet("target/ : Ground truth label mask with values 0-3")
pdf.ln(3)

pdf.sub_title("Label Remapping (Critical Step)")
pdf.body_text(
    "The original labels have 4 classes, but we need binary (change/no-change). "
    "So we REMAP them:"
)
pdf.set_font("Courier", "", 10)
pdf.cell(0, 6, "  0 (Background)  -> 0 (No Change)", 0, 1)
pdf.cell(0, 6, "  1 (Intact)      -> 0 (No Change)", 0, 1)
pdf.cell(0, 6, "  2 (Damaged)     -> 1 (CHANGE)", 0, 1)
pdf.cell(0, 6, "  3 (Destroyed)   -> 1 (CHANGE)", 0, 1)
pdf.set_font("Helvetica", "", 10)
pdf.ln(3)

pdf.sub_title("Class Imbalance Problem")
pdf.body_text(
    "After remapping, ~97.6% of pixels are 'No Change' and only ~2.4% are 'Change'. "
    "This is a SEVERE imbalance. If the model just predicts 'No Change' for everything, "
    "it gets 97.6% accuracy but detects zero changes! We handle this with special loss "
    "functions (explained in Section 5)."
)

# ─── 3. ARCHITECTURE ───
pdf.add_page()
pdf.section_title("3. Model Architecture: Concat U-Net")

pdf.sub_title("Why U-Net?")
pdf.body_text(
    "U-Net is the gold standard for image segmentation tasks. It has an ENCODER "
    "(downsampling path that extracts features) and a DECODER (upsampling path that "
    "produces pixel-level predictions). Skip connections between encoder and decoder "
    "preserve spatial detail, which is critical for precise change detection."
)

pdf.sub_title("Our Architecture: Concat U-Net with ResNet34 Encoder")
pdf.body_text("Here's how our model works step by step:")
pdf.ln(2)

pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 7, "Step 1: Input Concatenation", 0, 1)
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "", 10)
pdf.body_text(
    "The pre-event EO image (3 channels) and post-event SAR image (1 channel) are "
    "CONCATENATED along the channel dimension to form a single 4-channel input tensor. "
    "This is the simplest and most effective way to fuse multi-modal data."
)

pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 7, "Step 2: Feature Extraction (Encoder)", 0, 1)
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "", 10)
pdf.body_text(
    "A ResNet34 encoder (pretrained on ImageNet) processes the 4-channel input. "
    "The first conv layer is modified to accept 4 channels instead of 3. "
    "ResNet34 has 34 layers with residual connections and produces feature maps at "
    "5 different scales (from 128x128 down to 4x4)."
)

pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 7, "Step 3: Pixel-wise Prediction (Decoder)", 0, 1)
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "", 10)
pdf.body_text(
    "The U-Net decoder upsamples the features back to the original resolution. "
    "At each level, it concatenates with the corresponding encoder features (skip "
    "connections). The final output is a single-channel map where each pixel value "
    "represents the probability of change."
)

pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(25, 60, 120)
pdf.cell(0, 7, "Step 4: Thresholding", 0, 1)
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "", 10)
pdf.body_text(
    "The raw output (logits) is passed through a sigmoid function to get probabilities "
    "[0, 1]. Then a threshold (0.55 in our case) is applied: if probability > 0.55, "
    "the pixel is classified as 'Change', otherwise 'No Change'."
)
pdf.ln(2)

pdf.sub_title("Model Stats")
pdf.key_value("Total Parameters", "24,439,505")
pdf.key_value("Encoder", "ResNet34 (ImageNet pretrained)")
pdf.key_value("Decoder", "U-Net")
pdf.key_value("Input Channels", "4 (3 EO + 1 SAR)")
pdf.key_value("Output", "1 channel (binary mask)")
pdf.ln(3)

pdf.sub_title("Why Concat U-Net and not Siamese U-Net?")
pdf.body_text(
    "A Siamese U-Net uses TWO separate encoder branches (one for pre, one for post) "
    "with shared weights. This works great when both images have the SAME number of channels. "
    "But our pre-event has 3 channels (EO) and post-event has 1 channel (SAR) - they're "
    "ASYMMETRIC. So we use Concat U-Net which simply concatenates them into 4 channels. "
    "The code auto-selects the right architecture based on channel counts."
)

# ─── 4. DATA PROCESSING ───
pdf.add_page()
pdf.section_title("4. Data Processing Pipeline")

pdf.sub_title("Image Loading")
pdf.body_text(
    "All images are loaded and resized to 128x128 pixels (from original 1024x1024). "
    "This is done for training speed on CPU. The pre-event image retains 3 RGB channels, "
    "and the post-event SAR image keeps its 1 grayscale channel. "
    "All pixel values are normalized to [0, 1] range."
)

pdf.sub_title("Data Augmentation (Training Only)")
pdf.body_text(
    "To prevent overfitting and increase effective dataset size, we apply random "
    "augmentations during training. Critically, the SAME random transformation is applied "
    "to the pre-event, post-event, AND label images to keep them aligned:"
)
pdf.bullet("Horizontal flip (50% chance)")
pdf.bullet("Vertical flip (50% chance)")
pdf.bullet("Random rotation (up to 15 degrees)")
pdf.bullet("Shift, scale, and rotate")
pdf.bullet("Gaussian blur")
pdf.bullet("Random brightness and contrast adjustments")
pdf.ln(2)
pdf.body_text("Library used: Albumentations (fast, GPU-friendly augmentation library)")

pdf.sub_title("Channel Fusion")
pdf.body_text(
    "After loading and augmenting, the pre-event (3ch) and post-event (1ch) are "
    "concatenated along the channel axis to create a single 4-channel tensor. "
    "This tensor is what the model actually receives as input."
)

# ─── 5. TRAINING STRATEGY ───
pdf.add_page()
pdf.section_title("5. Training Strategy")

pdf.sub_title("Loss Function: Combined BCE + Dice Loss")
pdf.body_text(
    "We use TWO loss functions combined, each handling a different aspect:"
)
pdf.ln(1)
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 6, "1. Binary Cross-Entropy (BCE) Loss - Weight: 0.3", 0, 1)
pdf.set_font("Helvetica", "", 10)
pdf.body_text(
    "Standard pixel-wise classification loss. We use pos_weight=5.0 to make the model "
    "pay 5x more attention to 'Change' pixels vs 'No Change' pixels. This directly "
    "addresses the 97.6% vs 2.4% class imbalance."
)
pdf.set_font("Helvetica", "B", 10)
pdf.cell(0, 6, "2. Dice Loss - Weight: 0.7", 0, 1)
pdf.set_font("Helvetica", "", 10)
pdf.body_text(
    "Dice loss measures the overlap between predicted and actual change regions. "
    "It's especially effective for imbalanced segmentation because it focuses on the "
    "GEOMETRY of the prediction, not individual pixels. We weight it at 0.7 (70% of "
    "total loss) because it's more robust for our imbalanced dataset."
)
pdf.body_text("Final Loss = 0.3 x BCE(pos_weight=5) + 0.7 x Dice Loss")

pdf.sub_title("Optimizer & Scheduler")
pdf.key_value("Optimizer", "Adam (lr=0.0005, weight_decay=0.0001)")
pdf.key_value("Scheduler", "Cosine Annealing (T_max=8, eta_min=1e-6)")
pdf.body_text(
    "Cosine annealing smoothly reduces the learning rate following a cosine curve, "
    "allowing fine-grained learning in later epochs."
)

pdf.sub_title("Training Configuration")
pdf.key_value("Epochs", "8")
pdf.key_value("Batch Size", "8")
pdf.key_value("Image Size", "128 x 128")
pdf.key_value("Device", "CPU")
pdf.key_value("Training Time", "~110 minutes")
pdf.key_value("Gradient Clipping", "1.0 (prevents exploding gradients)")

pdf.sub_title("Threshold Tuning")
pdf.body_text(
    "After each epoch, we search for the optimal threshold (from 0.1 to 0.9) that "
    "maximizes F1 score on the validation set. The best threshold found was 0.55."
)

# ─── 6. EVALUATION METRICS ───
pdf.add_page()
pdf.section_title("6. Evaluation Metrics Explained")

metrics = [
    ("IoU (Intersection over Union)",
     "Measures the overlap between predicted change region and actual change region. "
     "IoU = TP / (TP + FP + FN). A perfect score is 1.0. Also called Jaccard Index."),
    ("Precision",
     "Of all pixels the model PREDICTED as 'Change', what fraction actually ARE change? "
     "Precision = TP / (TP + FP). High precision = few false alarms."),
    ("Recall (Sensitivity)",
     "Of all pixels that actually ARE 'Change', what fraction did the model DETECT? "
     "Recall = TP / (TP + FN). High recall = finds most changes."),
    ("F1 Score",
     "Harmonic mean of Precision and Recall. F1 = 2 x (P x R) / (P + R). "
     "Balances both metrics into a single number."),
    ("Accuracy",
     "Fraction of all pixels correctly classified. Misleading for imbalanced data - "
     "predicting all 'No Change' gives 97.6% accuracy!"),
    ("Confusion Matrix",
     "A 2x2 table showing True Positives (TP), False Positives (FP), "
     "True Negatives (TN), and False Negatives (FN). Gives the complete picture.")
]
for name, desc in metrics:
    pdf.sub_title(name)
    pdf.body_text(desc)

# ─── 7. RESULTS ───
pdf.add_page()
pdf.section_title("7. Results")

pdf.sub_title("Validation Set Results")
w1, w2 = 50, 40
pdf.set_font("Helvetica", "B", 10)
pdf.set_fill_color(220, 230, 245)
pdf.cell(w1, 7, "Metric", 1, 0, "C", fill=True)
pdf.cell(w2, 7, "Value", 1, 0, "C", fill=True)
pdf.cell(w2, 7, "Percentage", 1, 1, "C", fill=True)
pdf.set_font("Helvetica", "", 10)
for m, v in [("IoU", "0.1895"), ("Precision", "0.2561"), ("Recall", "0.4218"), ("F1 Score", "0.3187"), ("Accuracy", "0.9604")]:
    pdf.cell(w1, 7, m, 1, 0, "C")
    pdf.cell(w2, 7, v, 1, 0, "C")
    pdf.cell(w2, 7, f"{float(v)*100:.2f}%", 1, 1, "C")
pdf.ln(3)

pdf.sub_title("Test Set Results")
pdf.set_font("Helvetica", "B", 10)
pdf.set_fill_color(220, 230, 245)
pdf.cell(w1, 7, "Metric", 1, 0, "C", fill=True)
pdf.cell(w2, 7, "Value", 1, 0, "C", fill=True)
pdf.cell(w2, 7, "Percentage", 1, 1, "C", fill=True)
pdf.set_font("Helvetica", "", 10)
for m, v in [("IoU", "0.0155"), ("Precision", "0.0158"), ("Recall", "0.4296"), ("F1 Score", "0.0306"), ("Accuracy", "0.7953")]:
    pdf.cell(w1, 7, m, 1, 0, "C")
    pdf.cell(w2, 7, v, 1, 0, "C")
    pdf.cell(w2, 7, f"{float(v)*100:.2f}%", 1, 1, "C")
pdf.ln(3)

pdf.sub_title("Why Test Performance is Lower")
pdf.body_text(
    "The validation set comes from the SAME scene (scene_01) as training data, so the model "
    "has learned that visual distribution. The test set comes from a DIFFERENT scene (scene_09) "
    "with different terrain, building styles, and SAR characteristics. This is a DOMAIN "
    "GENERALIZATION problem - one of the hardest challenges in machine learning."
)
pdf.body_text("Additional factors:")
pdf.bullet("Only 8 epochs of training on CPU (limited convergence)")
pdf.bullet("128x128 resolution loses fine spatial details from 1024x1024 originals")
pdf.bullet("Extreme class imbalance makes precision inherently fragile")
pdf.bullet("Cross-sensor fusion (optical + radar) adds complexity")
pdf.ln(2)
pdf.body_text(
    "The RECALL of 42.96% on test is actually encouraging - it means the model detects "
    "nearly half of all actual changes. The low precision (1.58%) means it also produces "
    "many false alarms, which could be improved with more training and post-processing."
)

# ─── 8. TECHNIQUES SUMMARY ───
pdf.add_page()
pdf.section_title("8. All Techniques Used")

techniques = [
    ("Transfer Learning", "ResNet34 encoder pretrained on ImageNet (1.2M images). Instead of learning from scratch, we fine-tune a model that already understands edges, textures, and shapes."),
    ("Multi-Modal Fusion", "Combining optical (EO) and radar (SAR) imagery via channel concatenation. EO provides visual detail, SAR provides weather-independent structural information."),
    ("Combined Loss Function", "BCE + Dice loss weighted 0.3/0.7. BCE handles pixel classification, Dice handles region overlap. Together they're more robust than either alone."),
    ("Positive Weight (pos_weight=5)", "Makes the BCE loss penalize missed changes 5x more than false alarms. Directly counteracts the 97.6% no-change imbalance."),
    ("Data Augmentation", "Random flips, rotations, blur, brightness changes. Effectively multiplies dataset size and prevents overfitting."),
    ("Cosine Annealing LR", "Smoothly decays learning rate following a cosine curve. Allows aggressive early learning and fine-tuned late learning."),
    ("Threshold Optimization", "Searches 17 threshold values (0.1-0.9) to find the one maximizing F1. Found 0.55 as optimal."),
    ("Gradient Clipping", "Clips gradients to max norm of 1.0. Prevents training instability from exploding gradients."),
    ("Early Stopping", "Monitors validation F1 and stops training if no improvement for 8 epochs. Prevents overfitting."),
    ("Automatic Architecture Selection", "Code auto-detects channel counts and selects Siamese U-Net (equal channels) or Concat U-Net (asymmetric channels)."),
]
for name, desc in techniques:
    pdf.sub_title(name)
    pdf.body_text(desc)

# ─── 9. CODE STRUCTURE ───
pdf.add_page()
pdf.section_title("9. Code Structure Explained")

files = [
    ("train.py", "Main training script. Loads data, builds model, runs training loop with validation, saves best checkpoint. Includes early stopping and threshold tuning."),
    ("evaluate.py", "Runs trained model on val/test sets. Computes all metrics, generates confusion matrices, saves qualitative visualizations."),
    ("inference.py", "Single image pair prediction. Load any pre/post image pair and get a change mask output."),
    ("download_dataset.py", "Downloads the dataset from HuggingFace automatically."),
    ("config.yaml", "ALL hyperparameters in one place: paths, model config, training params, loss weights, augmentation settings."),
    ("src/model.py", "Defines SiameseUNet and ConcatUNet architectures using segmentation_models_pytorch library."),
    ("src/dataset.py", "PyTorch Dataset class. Handles loading images, label remapping, augmentation, and channel concatenation."),
    ("src/losses.py", "CombinedLoss class implementing weighted BCE + Dice loss."),
    ("src/metrics.py", "MetricsCalculator class computing IoU, Precision, Recall, F1, confusion matrix."),
    ("src/utils.py", "Helper functions: config loading, seeding, checkpoint save/load, EarlyStopping class."),
    ("src/visualize.py", "Plotting functions for training curves, confusion matrices, and qualitative prediction grids."),
    ("report/generate_report.py", "Generates the formal Technical Report PDF from evaluation results."),
]
for fname, desc in files:
    pdf.set_font("Courier", "B", 10)
    pdf.set_text_color(25, 60, 120)
    pdf.cell(0, 6, fname, 0, 1)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, desc)
    pdf.ln(2)

# ─── 10. HOW TO REPRODUCE ───
pdf.add_page()
pdf.section_title("10. How to Reproduce")

steps = [
    ("1. Install Dependencies", "pip install -r requirements.txt"),
    ("2. Download Dataset", "python download_dataset.py"),
    ("3. Train the Model", "python train.py"),
    ("4. Evaluate", "python evaluate.py --split both"),
    ("5. Generate Report", "python report/generate_report.py"),
    ("6. Run Inference", "python inference.py --pre path/to/pre.png --post path/to/post.png"),
]
for title, cmd in steps:
    pdf.sub_title(title)
    pdf.set_font("Courier", "", 10)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(0, 7, f"  {cmd}", 0, 1, fill=True)
    pdf.ln(3)

# ─── 11. POTENTIAL IMPROVEMENTS ───
pdf.section_title("11. Potential Improvements")
improvements = [
    "Train for 50+ epochs on GPU for better convergence",
    "Use 256x256 or 512x512 resolution to preserve spatial detail",
    "Attention-based fusion (cross-attention between EO and SAR features)",
    "Feature Pyramid Network (FPN) decoder for multi-scale detection",
    "Test-Time Augmentation (TTA) - average predictions over flipped versions",
    "Focal Loss for hard example mining (focus on difficult pixels)",
    "Post-processing: morphological operations to remove small false positive blobs",
    "Domain adaptation techniques to bridge train-test scene gap",
    "Pre-train on larger remote sensing datasets (e.g., xBD, LEVIR-CD)",
]
for imp in improvements:
    pdf.bullet(imp)

# ─── 12. KEY TAKEAWAYS ───
pdf.add_page()
pdf.section_title("12. Key Takeaways")

pdf.sub_title("What This Project Demonstrates")
pdf.bullet("End-to-end deep learning pipeline from raw data to evaluation")
pdf.bullet("Multi-modal satellite image fusion (optical + radar)")
pdf.bullet("Handling severe class imbalance in segmentation")
pdf.bullet("Transfer learning with pretrained encoders")
pdf.bullet("Comprehensive evaluation with multiple metrics")
pdf.bullet("Professional code structure with config-driven design")
pdf.bullet("Reproducible research with documented results")
pdf.ln(5)

pdf.sub_title("Quick Reference - Submission Values")
pdf.ln(2)
w1, w2 = 70, 100
pdf.set_font("Helvetica", "B", 11)
pdf.set_fill_color(25, 60, 120)
pdf.set_text_color(255, 255, 255)
pdf.cell(w1, 8, "Field", 1, 0, "C", fill=True)
pdf.cell(w2, 8, "Value", 1, 1, "C", fill=True)
pdf.set_text_color(0, 0, 0)
pdf.set_font("Helvetica", "", 11)
vals = [
    ("Architecture", "Concat U-Net (ResNet34)"),
    ("IoU on Test Split", "0.0155"),
    ("Precision on Test Split", "0.0158"),
    ("Recall on Test Split", "0.4296"),
    ("GitHub Repo", "github.com/prathamn128/GalaxEye-Change-Detection"),
]
for field, val in vals:
    pdf.cell(w1, 8, field, 1, 0, "C")
    pdf.cell(w2, 8, val, 1, 1, "C")

# Save
out_path = "Project_Summary.pdf"
pdf.output(out_path)
print(f"Summary PDF saved to: {out_path}")
