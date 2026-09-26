import os
import sys
import json
import time
import random
import platform
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support
)
import seaborn as sns

try:
    from .model import DermaAI_MobileNetV3
    from .preprocessing import CLASS_MAPPING, REVERSE_CLASS_MAPPING, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    from model import DermaAI_MobileNetV3
    from preprocessing import CLASS_MAPPING, REVERSE_CLASS_MAPPING, IMAGENET_MEAN, IMAGENET_STD

# ── Reproducibility ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Paths & Config ──────────────────────────────────────────────────────────────
SRC_DIR       = os.path.dirname(os.path.abspath(__file__))
AI_DIR        = os.path.dirname(SRC_DIR)
DATASET_DIR   = os.path.join(AI_DIR, "dataset")
SPLITS_DIR    = os.path.join(DATASET_DIR, "splits")
MODELS_DIR    = os.path.join(AI_DIR, "models")
RESULTS_DIR   = os.path.join(AI_DIR, "results", "experiment11")

CHECKPOINT_PATH = os.path.join(MODELS_DIR, "experiment8_best_model.pt")

VAL_CSV_PATH  = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV_PATH = os.path.join(SPLITS_DIR, "test.csv")

BATCH_SIZE = 16  # 16 images * 5 views = 80 forward passes per batch

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── TTA Dataset with 5 Deterministic Views ──────────────────────────────────────
class HAM10000TTADataset(Dataset):
    """
    Generates 5 deterministic views per image:
    View 1: Original image
    View 2: Horizontal flip
    View 3: Vertical flip
    View 4: Rotation +20 degrees
    View 5: Rotation -20 degrees
    Each view is resized to 224x224, converted to tensor, and normalized with ImageNet statistics.
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe.reset_index(drop=True)
        self.eval_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.loc[idx, 'image_path']
        image = Image.open(img_path).convert('RGB')
        label_str = self.dataframe.loc[idx, 'dx']
        label = CLASS_MAPPING[label_str]

        # 5 deterministic views
        v1 = image
        v2 = TF.hflip(image)
        v3 = TF.vflip(image)
        v4 = TF.rotate(image, 20.0, interpolation=transforms.InterpolationMode.BILINEAR)
        v5 = TF.rotate(image, -20.0, interpolation=transforms.InterpolationMode.BILINEAR)

        views_tensor = torch.stack([
            self.eval_tf(v1),
            self.eval_tf(v2),
            self.eval_tf(v3),
            self.eval_tf(v4),
            self.eval_tf(v5)
        ])  # Shape: [5, 3, 224, 224]

        return views_tensor, torch.tensor(label, dtype=torch.long), img_path


# ── Smoke Test & Verification ──────────────────────────────────────────────────
def run_smoke_test(model, device):
    print("\n" + "="*60)
    print("STEP 1: PRE-INFERENCE SMOKE TEST & TTA VERIFICATION")
    print("="*60)

    # 1. Parameter count verification
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model architecture     : DermaAI_MobileNetV3 (MobileNetV3-Large + CBAM)")
    print(f"  Total parameters       : {total_params:,}")
    assert total_params == 4326297, f"Parameter mismatch! Expected 4,326,297 but got {total_params}"

    # 2. 1-image 5-view forward pass verification
    dummy_views = torch.randn(5, 3, 224, 224).to(device)
    with torch.no_grad():
        dummy_out = model(dummy_views)
        dummy_probs = torch.softmax(dummy_out, dim=1).cpu().numpy()

    assert dummy_out.shape == (5, 7), f"Expected shape (5, 7), got {dummy_out.shape}"
    p_tta = np.mean(dummy_probs, axis=0)
    assert np.isclose(np.sum(p_tta), 1.0, atol=1e-5), "TTA averaged probability must sum to 1.0"
    print(f"  5-view forward pass    : PASSED (output shape: {dummy_out.shape})")
    print(f"  TTA probability sum    : {np.sum(p_tta):.6f} (PASSED)")
    print("="*60 + "\n")


# ── Metrics Helper Function ────────────────────────────────────────────────────
def compute_metrics(targets, preds, class_names):
    acc  = accuracy_score(targets, preds)
    bacc = balanced_accuracy_score(targets, preds)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        targets, preds, average='macro', zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        targets, preds, average='weighted', zero_division=0)

    report_dict = classification_report(
        targets, preds, target_names=class_names,
        output_dict=True, zero_division=0)
    report_text = classification_report(
        targets, preds, target_names=class_names, zero_division=0)

    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))

    mel_idx   = CLASS_MAPPING["mel"]
    nv_idx    = CLASS_MAPPING["nv"]
    bkl_idx   = CLASS_MAPPING["bkl"]
    bcc_idx   = CLASS_MAPPING["bcc"]
    akiec_idx = CLASS_MAPPING["akiec"]

    mel_total = int(np.sum(targets == mel_idx))
    mel_pred_total = int(np.sum(preds == mel_idx))
    mel_tp = int(cm[mel_idx, mel_idx])
    mel_fp = mel_pred_total - mel_tp
    mel_to_nv    = int(cm[mel_idx, nv_idx])
    mel_to_bkl   = int(cm[mel_idx, bkl_idx])
    mel_to_akiec = int(cm[mel_idx, akiec_idx])

    mel_rec  = float(report_dict["mel"]["recall"])
    mel_prec = float(report_dict["mel"]["precision"])
    mel_f1   = float(report_dict["mel"]["f1-score"])

    nv_total = int(np.sum(targets == nv_idx))
    nv_tp = int(cm[nv_idx, nv_idx])
    nv_rec    = float(report_dict["nv"]["recall"])
    nv_prec   = float(report_dict["nv"]["precision"])
    nv_f1     = float(report_dict["nv"]["f1-score"])
    nv_to_mel = int(cm[nv_idx, mel_idx])

    bcc_rec = float(report_dict["bcc"]["recall"])
    bkl_rec = float(report_dict["bkl"]["recall"])

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
        "melanoma_precision": mel_prec,
        "melanoma_recall": mel_rec,
        "melanoma_f1": mel_f1,
        "melanoma_tp": mel_tp,
        "melanoma_support": mel_total,
        "melanoma_predicted_total": mel_pred_total,
        "melanoma_false_positives": mel_fp,
        "melanoma_to_nv_errors": mel_to_nv,
        "melanoma_to_bkl_errors": mel_to_bkl,
        "melanoma_to_akiec_errors": mel_to_akiec,
        "nv_recall": nv_rec,
        "nv_precision": nv_prec,
        "nv_f1": nv_f1,
        "nv_support": nv_total,
        "nv_to_mel_errors": nv_to_mel,
        "bcc_recall": bcc_rec,
        "bkl_recall": bkl_rec,
        "per_class": {
            cls: {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1-score": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"]),
                "correct": int(cm[CLASS_MAPPING[cls], CLASS_MAPPING[cls]])
            } for cls in class_names
        },
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist()
    }


# ── Plot and Save Confusion Matrix ─────────────────────────────────────────────
def plot_and_save_cm(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ── Run TTA Batch Inference ────────────────────────────────────────────────────
def run_dataset_inference(model, loader, device, desc):
    """
    Executes inference for all samples in loader.
    For each image:
      - Obtains 5-view softmax probabilities [5, 7]
      - Single-view probability: view 0
      - TTA probability: mean across 5 views
      - Consensus and confidence metrics
    """
    all_targets = []
    single_probs = []
    tta_probs = []
    all_view_preds = []
    all_view_max_probs = []
    img_paths = []

    start_t = time.time()
    with torch.no_grad():
        for batch_views, targets, paths in tqdm(loader, desc=desc):
            # batch_views: [B, 5, 3, 224, 224]
            B = batch_views.size(0)
            flat_views = batch_views.view(B * 5, 3, 224, 224).to(device)

            outputs = model(flat_views)
            probs = torch.softmax(outputs, dim=1).view(B, 5, 7).cpu().numpy()

            # View 0 is single-view
            single_p = probs[:, 0, :]  # [B, 7]
            # Arithmetic mean across 5 views
            tta_p = np.mean(probs, axis=1)  # [B, 7]

            view_preds = np.argmax(probs, axis=2)  # [B, 5]
            view_max_p = np.max(probs, axis=2)     # [B, 5]

            all_targets.append(targets.numpy())
            single_probs.append(single_p)
            tta_probs.append(tta_p)
            all_view_preds.append(view_preds)
            all_view_max_probs.append(view_max_p)
            img_paths.extend(paths)

    duration = time.time() - start_t
    targets = np.concatenate(all_targets)
    single_probs = np.vstack(single_probs)
    tta_probs = np.vstack(tta_probs)
    view_preds = np.vstack(all_view_preds)
    view_max_probs = np.vstack(all_view_max_probs)

    single_preds = np.argmax(single_probs, axis=1)
    tta_preds = np.argmax(tta_probs, axis=1)

    return {
        "targets": targets,
        "single_probs": single_probs,
        "tta_probs": tta_probs,
        "single_preds": single_preds,
        "tta_preds": tta_preds,
        "view_preds": view_preds,
        "view_max_probs": view_max_probs,
        "img_paths": img_paths,
        "duration": duration
    }


# ── Main Experiment 11 Routine ─────────────────────────────────────────────────
def main():
    start_total_time = time.time()
    device = torch.device("cpu")
    print("="*70)
    print("EXPERIMENT 11: TEST-TIME AUGMENTATION (TTA) ON HAM10000")
    print(f"Device        : {device} ({platform.processor()})")
    print(f"PyTorch       : {torch.__version__} | torchvision: {torchvision.__version__}")
    print(f"Base Model    : {CHECKPOINT_PATH}")
    print("="*70)

    assert os.path.exists(CHECKPOINT_PATH), f"Checkpoint missing: {CHECKPOINT_PATH}"
    assert os.path.exists(VAL_CSV_PATH), f"Validation split missing: {VAL_CSV_PATH}"
    assert os.path.exists(TEST_CSV_PATH), f"Test split missing: {TEST_CSV_PATH}"

    val_df  = pd.read_csv(VAL_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    print(f"Splits verified: Val = {len(val_df):,} images | Test = {len(test_df):,} images")

    # Load frozen model
    model = DermaAI_MobileNetV3(num_classes=7)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Pre-inference smoke test
    run_smoke_test(model, device)

    class_names = [REVERSE_CLASS_MAPPING[i] for i in range(7)]

    # =========================================================================
    # STEP 2: VALIDATION INFERENCE (DESCRIPTIVE VALIDATION ONLY)
    # =========================================================================
    print("="*60)
    print("STEP 2: RUNNING VALIDATION INFERENCE (1,010 IMAGES, 5 VIEWS EACH)")
    print("="*60)

    val_dataset = HAM10000TTADataset(val_df)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)

    val_res = run_dataset_inference(model, val_loader, device, "Validation TTA inference")
    val_duration = val_res["duration"]
    print(f"Validation inference completed in {val_duration:.2f}s ({len(val_res['targets'])} images)")

    # Compute validation metrics
    val_metrics_single = compute_metrics(val_res["targets"], val_res["single_preds"], class_names)
    val_metrics_tta    = compute_metrics(val_res["targets"], val_res["tta_preds"], class_names)

    # Save validation artifacts
    with open(os.path.join(RESULTS_DIR, "validation_metrics_single_view.json"), "w") as f:
        json.dump(val_metrics_single, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "validation_metrics_tta.json"), "w") as f:
        json.dump(val_metrics_tta, f, indent=4)

    print("\nVALIDATION METRICS SUMMARY:")
    print(f"  Single-View | Acc: {val_metrics_single['accuracy']*100:.2f}% | BalAcc: {val_metrics_single['balanced_accuracy']*100:.2f}% | Macro F1: {val_metrics_single['macro_f1']:.4f} | Mel Rec: {val_metrics_single['melanoma_recall']*100:.2f}% | Mel F1: {val_metrics_single['melanoma_f1']:.4f}")
    print(f"  5-View TTA  | Acc: {val_metrics_tta['accuracy']*100:.2f}% | BalAcc: {val_metrics_tta['balanced_accuracy']*100:.2f}% | Macro F1: {val_metrics_tta['macro_f1']:.4f} | Mel Rec: {val_metrics_tta['melanoma_recall']*100:.2f}% | Mel F1: {val_metrics_tta['melanoma_f1']:.4f}")

    # =========================================================================
    # STEP 3: FINAL TEST EVALUATION (EXACTLY ONCE ON 988-IMAGE TEST SET)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: SINGLE FINAL EVALUATION ON HELD-OUT TEST SET (988 IMAGES)")
    print("="*60)

    test_dataset = HAM10000TTADataset(test_df)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    test_res = run_dataset_inference(model, test_loader, device, "Test TTA inference")
    test_duration = test_res["duration"]
    print(f"Test inference completed in {test_duration:.2f}s ({len(test_res['targets'])} images)")

    # Compute test metrics
    test_metrics_single = compute_metrics(test_res["targets"], test_res["single_preds"], class_names)
    test_metrics_tta    = compute_metrics(test_res["targets"], test_res["tta_preds"], class_names)

    # Save test metrics JSONs
    with open(os.path.join(RESULTS_DIR, "test_metrics_single_view.json"), "w") as f:
        json.dump(test_metrics_single, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "test_metrics_tta.json"), "w") as f:
        json.dump(test_metrics_tta, f, indent=4)

    # Save classification reports
    with open(os.path.join(RESULTS_DIR, "classification_report_single_view.txt"), "w") as f:
        f.write(test_metrics_single["classification_report_text"])
    with open(os.path.join(RESULTS_DIR, "classification_report_tta.txt"), "w") as f:
        f.write(test_metrics_tta["classification_report_text"])

    # Save confusion matrices (numerical JSON and PNG)
    with open(os.path.join(RESULTS_DIR, "confusion_matrix_single_view.json"), "w") as f:
        json.dump(test_metrics_single["confusion_matrix"], f, indent=4)
    with open(os.path.join(RESULTS_DIR, "confusion_matrix_tta.json"), "w") as f:
        json.dump(test_metrics_tta["confusion_matrix"], f, indent=4)

    plot_and_save_cm(
        np.array(test_metrics_single["confusion_matrix"]), class_names,
        "Experiment 8 (Single-View) Confusion Matrix - Test Split",
        os.path.join(RESULTS_DIR, "confusion_matrix_single_view.png")
    )
    plot_and_save_cm(
        np.array(test_metrics_tta["confusion_matrix"]), class_names,
        "Experiment 11 (5-View TTA) Confusion Matrix - Test Split",
        os.path.join(RESULTS_DIR, "confusion_matrix_tta.png")
    )

    # =========================================================================
    # STEP 4: TTA-SPECIFIC PREDICTION & CONSENSUS ANALYSIS
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 4: TTA PREDICTION CHANGE & CONSENSUS ANALYSIS")
    print("="*60)

    targets = test_res["targets"]
    s_preds = test_res["single_preds"]
    t_preds = test_res["tta_preds"]
    v_preds = test_res["view_preds"]       # [N, 5]
    v_max_p = test_res["view_max_probs"]   # [N, 5]
    t_probs = test_res["tta_probs"]        # [N, 7]

    N = len(targets)
    changed_mask = (s_preds != t_preds)
    num_changed = int(np.sum(changed_mask))

    s_correct = (s_preds == targets)
    t_correct = (t_preds == targets)

    correct_to_incorrect = int(np.sum(s_correct & (~t_correct)))
    incorrect_to_correct = int(np.sum((~s_correct) & t_correct))
    net_accuracy_change = (incorrect_to_correct - correct_to_incorrect) / N

    # Agreement count: how many of the 5 views agreed with the final TTA prediction
    # For each sample i, count how many v_preds[i, :] == t_preds[i]
    agreement_counts = np.sum(v_preds == t_preds[:, None], axis=1)  # shape [N]
    mean_max_prob_views = np.mean(v_max_p, axis=1)                   # shape [N]
    max_prob_tta = np.max(t_probs, axis=1)                           # shape [N]

    # Agreement breakdown
    agreement_dist = {f"{k}/5": int(np.sum(agreement_counts == k)) for k in range(5, 0, -1)}
    agreement_dist_changed = {f"{k}/5": int(np.sum(agreement_counts[changed_mask] == k)) for k in range(5, 0, -1)}
    agreement_dist_unchanged = {f"{k}/5": int(np.sum(agreement_counts[~changed_mask] == k)) for k in range(5, 0, -1)}

    mean_conf_all = float(np.mean(max_prob_tta))
    mean_conf_changed = float(np.mean(max_prob_tta[changed_mask])) if num_changed > 0 else 0.0
    mean_conf_unchanged = float(np.mean(max_prob_tta[~changed_mask]))

    tta_changes_artifact = {
        "total_test_images": N,
        "prediction_changed_count": num_changed,
        "prediction_changed_percentage": float(num_changed / N * 100),
        "correct_to_incorrect": correct_to_incorrect,
        "incorrect_to_correct": incorrect_to_correct,
        "net_correct_change": incorrect_to_correct - correct_to_incorrect,
        "net_accuracy_change": float(net_accuracy_change),
        "melanoma_analysis": {
            "melanoma_predictions_single_view": int(np.sum(s_preds == CLASS_MAPPING["mel"])),
            "melanoma_predictions_tta": int(np.sum(t_preds == CLASS_MAPPING["mel"])),
            "melanoma_tp_single_view": test_metrics_single["melanoma_tp"],
            "melanoma_tp_tta": test_metrics_tta["melanoma_tp"],
            "melanoma_tp_change": test_metrics_tta["melanoma_tp"] - test_metrics_single["melanoma_tp"],
            "melanoma_fp_single_view": test_metrics_single["melanoma_false_positives"],
            "melanoma_fp_tta": test_metrics_tta["melanoma_false_positives"],
            "melanoma_fp_change": test_metrics_tta["melanoma_false_positives"] - test_metrics_single["melanoma_false_positives"],
            "mel_to_nv_single_view": test_metrics_single["melanoma_to_nv_errors"],
            "mel_to_nv_tta": test_metrics_tta["melanoma_to_nv_errors"],
            "mel_to_nv_change": test_metrics_tta["melanoma_to_nv_errors"] - test_metrics_single["melanoma_to_nv_errors"],
            "mel_to_bkl_single_view": test_metrics_single["melanoma_to_bkl_errors"],
            "mel_to_bkl_tta": test_metrics_tta["melanoma_to_bkl_errors"],
            "mel_to_bkl_change": test_metrics_tta["melanoma_to_bkl_errors"] - test_metrics_single["melanoma_to_bkl_errors"],
            "nv_to_mel_single_view": test_metrics_single["nv_to_mel_errors"],
            "nv_to_mel_tta": test_metrics_tta["nv_to_mel_errors"],
            "nv_to_mel_change": test_metrics_tta["nv_to_mel_errors"] - test_metrics_single["nv_to_mel_errors"]
        },
        "consensus_analysis": {
            "view_agreement_distribution_all": agreement_dist,
            "view_agreement_distribution_changed_cases": agreement_dist_changed,
            "view_agreement_distribution_unchanged_cases": agreement_dist_unchanged,
            "mean_confidence_all": mean_conf_all,
            "mean_confidence_changed": mean_conf_changed,
            "mean_confidence_unchanged": mean_conf_unchanged,
            "mean_max_prob_across_views_all": float(np.mean(mean_max_prob_views)),
            "mean_max_prob_across_views_changed": float(np.mean(mean_max_prob_views[changed_mask])) if num_changed > 0 else 0.0
        }
    }

    with open(os.path.join(RESULTS_DIR, "tta_prediction_changes.json"), "w") as f:
        json.dump(tta_changes_artifact, f, indent=4)

    print(f"  Total test images          : {N}")
    print(f"  Predictions changed by TTA : {num_changed} ({num_changed/N*100:.2f}%)")
    print(f"  Incorrect -> Correct (+)   : {incorrect_to_correct}")
    print(f"  Correct -> Incorrect (-)   : {correct_to_incorrect}")
    print(f"  Net accuracy change        : {net_accuracy_change*100:+.2f}%")
    print(f"  View agreement on all      : {agreement_dist}")
    print(f"  View agreement on changed  : {agreement_dist_changed}")
    print(f"  Mean confidence (changed)  : {mean_conf_changed:.4f} vs (unchanged): {mean_conf_unchanged:.4f}")

    # =========================================================================
    # STEP 5: COMPARISONS WITH EXPERIMENT 8 & EXPERIMENT 9
    # =========================================================================
    # Comparison with Experiment 8
    exp8_metrics_path = os.path.join(AI_DIR, "results", "experiment8", "metrics.json")
    with open(exp8_metrics_path, "r") as f:
        exp8_baseline = json.load(f)

    delta_exp11_minus_exp8 = {
        "accuracy_change": test_metrics_tta["accuracy"] - test_metrics_single["accuracy"],
        "balanced_accuracy_change": test_metrics_tta["balanced_accuracy"] - test_metrics_single["balanced_accuracy"],
        "macro_f1_change": test_metrics_tta["macro_f1"] - test_metrics_single["macro_f1"],
        "macro_precision_change": test_metrics_tta["macro_precision"] - test_metrics_single["macro_precision"],
        "macro_recall_change": test_metrics_tta["macro_recall"] - test_metrics_single["macro_recall"],
        "weighted_f1_change": test_metrics_tta["weighted_f1"] - test_metrics_single["weighted_f1"],
        "melanoma_recall_change": test_metrics_tta["melanoma_recall"] - test_metrics_single["melanoma_recall"],
        "melanoma_precision_change": test_metrics_tta["melanoma_precision"] - test_metrics_single["melanoma_precision"],
        "melanoma_f1_change": test_metrics_tta["melanoma_f1"] - test_metrics_single["melanoma_f1"],
        "melanoma_tp_change": test_metrics_tta["melanoma_tp"] - test_metrics_single["melanoma_tp"],
        "melanoma_fp_change": test_metrics_tta["melanoma_false_positives"] - test_metrics_single["melanoma_false_positives"],
        "mel_to_nv_error_change": test_metrics_tta["melanoma_to_nv_errors"] - test_metrics_single["melanoma_to_nv_errors"],
        "mel_to_bkl_error_change": test_metrics_tta["melanoma_to_bkl_errors"] - test_metrics_single["melanoma_to_bkl_errors"],
        "nv_recall_change": test_metrics_tta["nv_recall"] - test_metrics_single["nv_recall"],
        "nv_precision_change": test_metrics_tta["nv_precision"] - test_metrics_single["nv_precision"],
        "nv_f1_change": test_metrics_tta["nv_f1"] - test_metrics_single["nv_f1"],
        "bcc_recall_change": test_metrics_tta["bcc_recall"] - test_metrics_single["bcc_recall"],
        "bkl_recall_change": test_metrics_tta["bkl_recall"] - test_metrics_single["bkl_recall"]
    }

    comp_exp8 = {
        "baseline_experiment8_single_view": {
            "accuracy": test_metrics_single["accuracy"],
            "balanced_accuracy": test_metrics_single["balanced_accuracy"],
            "macro_precision": test_metrics_single["macro_precision"],
            "macro_recall": test_metrics_single["macro_recall"],
            "macro_f1": test_metrics_single["macro_f1"],
            "weighted_f1": test_metrics_single["weighted_f1"],
            "melanoma_recall": test_metrics_single["melanoma_recall"],
            "melanoma_precision": test_metrics_single["melanoma_precision"],
            "melanoma_f1": test_metrics_single["melanoma_f1"],
            "melanoma_tp": test_metrics_single["melanoma_tp"],
            "melanoma_false_positives": test_metrics_single["melanoma_false_positives"],
            "mel_to_nv_errors": test_metrics_single["melanoma_to_nv_errors"],
            "mel_to_bkl_errors": test_metrics_single["melanoma_to_bkl_errors"],
            "nv_recall": test_metrics_single["nv_recall"],
            "nv_to_mel_errors": test_metrics_single["nv_to_mel_errors"]
        },
        "experiment11_tta": {
            "accuracy": test_metrics_tta["accuracy"],
            "balanced_accuracy": test_metrics_tta["balanced_accuracy"],
            "macro_precision": test_metrics_tta["macro_precision"],
            "macro_recall": test_metrics_tta["macro_recall"],
            "macro_f1": test_metrics_tta["macro_f1"],
            "weighted_f1": test_metrics_tta["weighted_f1"],
            "melanoma_recall": test_metrics_tta["melanoma_recall"],
            "melanoma_precision": test_metrics_tta["melanoma_precision"],
            "melanoma_f1": test_metrics_tta["melanoma_f1"],
            "melanoma_tp": test_metrics_tta["melanoma_tp"],
            "melanoma_false_positives": test_metrics_tta["melanoma_false_positives"],
            "mel_to_nv_errors": test_metrics_tta["melanoma_to_nv_errors"],
            "mel_to_bkl_errors": test_metrics_tta["melanoma_to_bkl_errors"],
            "nv_recall": test_metrics_tta["nv_recall"],
            "nv_to_mel_errors": test_metrics_tta["nv_to_mel_errors"]
        },
        "delta_exp11_minus_exp8": delta_exp11_minus_exp8
    }
    with open(os.path.join(RESULTS_DIR, "comparison_with_experiment8.json"), "w") as f:
        json.dump(comp_exp8, f, indent=4)

    # Comparison with Experiment 9
    exp9_metrics_path = os.path.join(AI_DIR, "results", "experiment9", "metrics.json")
    with open(exp9_metrics_path, "r") as f:
        exp9_baseline = json.load(f)

    delta_exp11_minus_exp9 = {
        "accuracy_change": test_metrics_tta["accuracy"] - exp9_baseline["accuracy"],
        "balanced_accuracy_change": test_metrics_tta["balanced_accuracy"] - exp9_baseline["balanced_accuracy"],
        "macro_f1_change": test_metrics_tta["macro_f1"] - exp9_baseline["macro_f1"],
        "weighted_f1_change": test_metrics_tta["weighted_f1"] - exp9_baseline["weighted_f1"],
        "melanoma_recall_change": test_metrics_tta["melanoma_recall"] - exp9_baseline["melanoma_recall"],
        "melanoma_precision_change": test_metrics_tta["melanoma_precision"] - exp9_baseline["melanoma_precision"],
        "melanoma_f1_change": test_metrics_tta["melanoma_f1"] - exp9_baseline["melanoma_f1"]
    }
    comp_exp9 = {
        "experiment9_label_smoothing": {
            "accuracy": exp9_baseline["accuracy"],
            "balanced_accuracy": exp9_baseline["balanced_accuracy"],
            "macro_f1": exp9_baseline["macro_f1"],
            "weighted_f1": exp9_baseline["weighted_f1"],
            "melanoma_recall": exp9_baseline["melanoma_recall"],
            "melanoma_precision": exp9_baseline["melanoma_precision"],
            "melanoma_f1": exp9_baseline["melanoma_f1"]
        },
        "experiment11_tta": {
            "accuracy": test_metrics_tta["accuracy"],
            "balanced_accuracy": test_metrics_tta["balanced_accuracy"],
            "macro_f1": test_metrics_tta["macro_f1"],
            "weighted_f1": test_metrics_tta["weighted_f1"],
            "melanoma_recall": test_metrics_tta["melanoma_recall"],
            "melanoma_precision": test_metrics_tta["melanoma_precision"],
            "melanoma_f1": test_metrics_tta["melanoma_f1"]
        },
        "delta_exp11_minus_exp9": delta_exp11_minus_exp9
    }
    with open(os.path.join(RESULTS_DIR, "comparison_with_experiment9.json"), "w") as f:
        json.dump(comp_exp9, f, indent=4)

    # Save configuration artifact
    tta_config = {
        "checkpoint_path": CHECKPOINT_PATH,
        "model_architecture": "DermaAI_MobileNetV3 (MobileNetV3-Large + CBAM)",
        "parameter_count": 4326297,
        "class_ordering": class_names,
        "image_size": [224, 224],
        "interpolation_mode": "BILINEAR",
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        },
        "tta_transforms": [
            "View 1: Original image",
            "View 2: Horizontal flip (TF.hflip)",
            "View 3: Vertical flip (TF.vflip)",
            "View 4: Rotation +20 degrees (TF.rotate)",
            "View 5: Rotation -20 degrees (TF.rotate)"
        ],
        "averaging_rule": "Arithmetic mean of 5 softmax probability vectors: p_tta = (p1 + p2 + p3 + p4 + p5) / 5",
        "decision_rule": "predicted_class = argmax(p_tta)",
        "random_seed": SEED,
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "cpu_processor": platform.processor(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_inference_duration_seconds": val_duration,
        "test_inference_duration_seconds": test_duration,
        "total_experiment_duration_seconds": time.time() - start_total_time
    }
    with open(os.path.join(RESULTS_DIR, "tta_configuration.json"), "w") as f:
        json.dump(tta_config, f, indent=4)

    # Save experiment summary
    exp_summary = {
        "experiment": "Experiment 11: Test-Time Augmentation (TTA) on HAM10000",
        "base_model": "Experiment 8 (Tempered Class Weights alpha=0.75)",
        "test_metrics_single_view": test_metrics_single,
        "test_metrics_tta": test_metrics_tta,
        "delta_tta_vs_single_view": delta_exp11_minus_exp8,
        "prediction_changes": tta_changes_artifact,
        "runtime": tta_config
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(exp_summary, f, indent=4)

    # Formatted terminal printout
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION RESULTS: EXPERIMENT 11 (TTA) vs EXPERIMENT 8 BASELINE")
    print("="*70)
    print(f"  Test Accuracy               : {test_metrics_single['accuracy']*100:.2f}% -> {test_metrics_tta['accuracy']*100:.2f}%  (delta: {delta_exp11_minus_exp8['accuracy_change']*100:+.2f}%)")
    print(f"  Balanced Accuracy           : {test_metrics_single['balanced_accuracy']*100:.2f}% -> {test_metrics_tta['balanced_accuracy']*100:.2f}%  (delta: {delta_exp11_minus_exp8['balanced_accuracy_change']*100:+.2f}%)")
    print(f"  Macro Precision             : {test_metrics_single['macro_precision']:.4f} -> {test_metrics_tta['macro_precision']:.4f}  (delta: {delta_exp11_minus_exp8['macro_precision_change']:+.4f})")
    print(f"  Macro Recall                : {test_metrics_single['macro_recall']:.4f} -> {test_metrics_tta['macro_recall']:.4f}  (delta: {delta_exp11_minus_exp8['macro_recall_change']:+.4f})")
    print(f"  Macro F1                    : {test_metrics_single['macro_f1']:.4f} -> {test_metrics_tta['macro_f1']:.4f}  (delta: {delta_exp11_minus_exp8['macro_f1_change']:+.4f})")
    print(f"  Weighted F1                 : {test_metrics_single['weighted_f1']:.4f} -> {test_metrics_tta['weighted_f1']:.4f}  (delta: {delta_exp11_minus_exp8['weighted_f1_change']:+.4f})")
    print(f"  Melanoma Recall             : {test_metrics_single['melanoma_recall']*100:.2f}% -> {test_metrics_tta['melanoma_recall']*100:.2f}%  (delta: {delta_exp11_minus_exp8['melanoma_recall_change']*100:+.2f}%)")
    print(f"  Melanoma Precision          : {test_metrics_single['melanoma_precision']*100:.2f}% -> {test_metrics_tta['melanoma_precision']*100:.2f}%  (delta: {delta_exp11_minus_exp8['melanoma_precision_change']*100:+.2f}%)")
    print(f"  Melanoma F1                 : {test_metrics_single['melanoma_f1']:.4f} -> {test_metrics_tta['melanoma_f1']:.4f}  (delta: {delta_exp11_minus_exp8['melanoma_f1_change']:+.4f})")
    print(f"  Melanoma TP / 106           : {test_metrics_single['melanoma_tp']} -> {test_metrics_tta['melanoma_tp']}  (delta: {delta_exp11_minus_exp8['melanoma_tp_change']:+d})")
    print(f"  Melanoma False Positives    : {test_metrics_single['melanoma_false_positives']} -> {test_metrics_tta['melanoma_false_positives']}  (delta: {delta_exp11_minus_exp8['melanoma_fp_change']:+d})")
    print(f"  Mel -> NV Misses            : {test_metrics_single['melanoma_to_nv_errors']} -> {test_metrics_tta['melanoma_to_nv_errors']}  (delta: {delta_exp11_minus_exp8['mel_to_nv_error_change']:+d})")
    print(f"  Mel -> BKL Misses           : {test_metrics_single['melanoma_to_bkl_errors']} -> {test_metrics_tta['melanoma_to_bkl_errors']}  (delta: {delta_exp11_minus_exp8['mel_to_bkl_error_change']:+d})")
    nv_to_mel_change = test_metrics_tta['nv_to_mel_errors'] - test_metrics_single['nv_to_mel_errors']
    print(f"  NV -> MEL False Positives   : {test_metrics_single['nv_to_mel_errors']} -> {test_metrics_tta['nv_to_mel_errors']}  (delta: {nv_to_mel_change:+d})")
    print("="*70)
    print(f"All 15 Experiment 11 artifacts saved to: {RESULTS_DIR}")
    print("Experiment 11 complete. STOPPING as required -- awaiting next instruction.")


if __name__ == "__main__":
    main()
