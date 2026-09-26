import os
import sys
import json
import time
import random
import platform
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
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
    from .preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING, REVERSE_CLASS_MAPPING
except ImportError:
    from model import DermaAI_MobileNetV3
    from preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING, REVERSE_CLASS_MAPPING

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
RESULTS_DIR   = os.path.join(AI_DIR, "results", "experiment10")

CHECKPOINT_PATH = os.path.join(MODELS_DIR, "experiment8_best_model.pt")

VAL_CSV_PATH  = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV_PATH = os.path.join(SPLITS_DIR, "test.csv")

THRESHOLD_GRID = [
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
]

BATCH_SIZE = 32

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Step 1 & 5: Smoke Test & Logic Verification ────────────────────────────────
def run_smoke_test(model, device):
    print("\n" + "="*60)
    print("STEP 1: PRE-INFERENCE SMOKE TEST & OVERRIDE LOGIC VERIFICATION")
    print("="*60)
    
    # 1. Model architecture verification
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model architecture     : DermaAI_MobileNetV3 (MobileNetV3-Large + CBAM)")
    print(f"  Total parameters       : {total_params:,}")
    assert total_params == 4326297, f"Parameter mismatch! Expected 4,326,297 but got {total_params}"
    
    # 2. Dummy forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        dummy_out = model(dummy_input)
        dummy_probs = torch.softmax(dummy_out, dim=1).cpu().numpy()
    
    assert dummy_out.shape == (2, 7), f"Expected shape (2, 7), got {dummy_out.shape}"
    assert np.allclose(np.sum(dummy_probs, axis=1), 1.0, atol=1e-5), "Softmax probabilities must sum to 1.0"
    print(f"  Forward pass test      : PASSED (output shape: {dummy_out.shape})")
    
    # 3. Decision rule verification
    mel_idx = CLASS_MAPPING["mel"]
    # Test case 1: P(mel) >= threshold -> override to mel
    test_prob_1 = np.array([0.05, 0.05, 0.05, 0.05, 0.40, 0.35, 0.05])  # mel=0.40, nv=0.35
    th_1 = 0.35
    pred_1 = mel_idx if test_prob_1[mel_idx] >= th_1 else np.argmax(test_prob_1)
    assert pred_1 == mel_idx, "Override logic failed for P(mel) >= threshold"
    
    # Test case 2: P(mel) < threshold, argmax is nv -> predict nv
    test_prob_2 = np.array([0.05, 0.05, 0.05, 0.05, 0.30, 0.45, 0.05])  # mel=0.30, nv=0.45
    th_2 = 0.35
    pred_2 = mel_idx if test_prob_2[mel_idx] >= th_2 else np.argmax(test_prob_2)
    assert pred_2 == CLASS_MAPPING["nv"], "Fallback to argmax failed"
    
    print(f"  Override logic test    : PASSED")
    print("="*60 + "\n")


# ── Helper to Compute Full Metrics ─────────────────────────────────────────────
def compute_metrics_for_predictions(targets, preds, class_names):
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
    
    mel_idx = CLASS_MAPPING["mel"]
    nv_idx  = CLASS_MAPPING["nv"]
    bkl_idx = CLASS_MAPPING["bkl"]
    bcc_idx = CLASS_MAPPING["bcc"]
    
    mel_total = int(np.sum(targets == mel_idx))
    mel_pred_total = int(np.sum(preds == mel_idx))
    mel_tp = int(cm[mel_idx, mel_idx])
    mel_fp = mel_pred_total - mel_tp
    mel_to_nv  = int(cm[mel_idx, nv_idx])
    mel_to_bkl = int(cm[mel_idx, bkl_idx])
    
    mel_rec  = float(report_dict["mel"]["recall"])
    mel_prec = float(report_dict["mel"]["precision"])
    mel_f1   = float(report_dict["mel"]["f1-score"])
    
    nv_total = int(np.sum(targets == nv_idx))
    nv_tp = int(cm[nv_idx, nv_idx])
    nv_rec  = float(report_dict["nv"]["recall"])
    nv_prec = float(report_dict["nv"]["precision"])
    nv_f1   = float(report_dict["nv"]["f1-score"])
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
        "nv_recall": nv_rec,
        "nv_precision": nv_prec,
        "nv_f1": nv_f1,
        "nv_support": nv_total,
        "nv_to_mel_errors": nv_to_mel,
        "bcc_recall": bcc_rec,
        "bkl_recall": bkl_rec,
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist()
    }


# ── Plot & Save Confusion Matrix ───────────────────────────────────────────────
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


# ── Main Experiment 10 Pipeline ────────────────────────────────────────────────
def main():
    start_time = time.time()
    device = torch.device("cpu")
    print("="*70)
    print("EXPERIMENT 10: POST-HOC MELANOMA DECISION THRESHOLD TUNING")
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
    
    # Load Model
    model = DermaAI_MobileNetV3(num_classes=7)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # Run smoke test
    run_smoke_test(model, device)
    
    class_names = [REVERSE_CLASS_MAPPING[i] for i in range(7)]
    mel_idx = CLASS_MAPPING["mel"]
    
    # =========================================================================
    # STEP 2: VALIDATION INFERENCE (STRICTLY VALIDATION SET ONLY)
    # =========================================================================
    print("="*60)
    print("STEP 2: RUNNING INFERENCE ON VALIDATION SET (1,010 IMAGES)")
    print("="*60)
    
    val_dataset = HAM10000Dataset(val_df, transform=get_transforms(is_train=False))
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=False)
    
    val_probs = []
    val_targets = []
    val_start = time.time()
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            val_probs.append(probs.cpu().numpy())
            val_targets.append(targets.numpy())
            
    val_probs = np.vstack(val_probs)
    val_targets = np.concatenate(val_targets)
    val_duration = time.time() - val_start
    print(f"Validation inference completed in {val_duration:.2f}s ({len(val_targets)} samples)")
    
    # =========================================================================
    # STEP 3: THRESHOLD SWEEP ON VALIDATION SET ONLY
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 3: EVALUATING PREDEFINED THRESHOLD GRID ON VALIDATION DATA")
    print("="*60)
    
    sweep_results = []
    
    for th in THRESHOLD_GRID:
        # Override logic:
        # if P(mel) >= threshold: predicted_class = mel
        # else: predicted_class = argmax(probabilities)
        preds = []
        for p in val_probs:
            if p[mel_idx] >= th:
                preds.append(mel_idx)
            else:
                preds.append(int(np.argmax(p)))
        preds = np.array(preds)
        
        m = compute_metrics_for_predictions(val_targets, preds, class_names)
        
        row = {
            "threshold": float(th),
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "melanoma_precision": m["melanoma_precision"],
            "melanoma_recall": m["melanoma_recall"],
            "melanoma_f1": m["melanoma_f1"],
            "melanoma_tp": m["melanoma_tp"],
            "melanoma_support": m["melanoma_support"],
            "melanoma_predicted_total": m["melanoma_predicted_total"],
            "melanoma_false_positives": m["melanoma_false_positives"],
            "melanoma_to_nv_errors": m["melanoma_to_nv_errors"],
            "melanoma_to_bkl_errors": m["melanoma_to_bkl_errors"],
            "nv_recall": m["nv_recall"],
            "nv_precision": m["nv_precision"],
            "nv_f1": m["nv_f1"],
            "bcc_recall": m["bcc_recall"],
            "bkl_recall": m["bkl_recall"]
        }
        sweep_results.append(row)
    
    sweep_df = pd.DataFrame(sweep_results)
    
    # Save sweep artifacts
    sweep_csv_path = os.path.join(RESULTS_DIR, "threshold_sweep_validation.csv")
    sweep_json_path = os.path.join(RESULTS_DIR, "threshold_sweep_validation.json")
    sweep_df.to_csv(sweep_csv_path, index=False)
    with open(sweep_json_path, "w") as f:
        json.dump(sweep_results, f, indent=4)
        
    print(f"Validation sweep table saved -> {sweep_csv_path}")
    print("\nVALIDATION THRESHOLD SWEEP SUMMARY TABLE:")
    print(sweep_df[[
        "threshold", "accuracy", "balanced_accuracy", "macro_f1",
        "melanoma_recall", "melanoma_precision", "melanoma_f1",
        "melanoma_tp", "melanoma_false_positives", "nv_recall"
    ]].to_string(index=False))
    
    # =========================================================================
    # STEP 4: THRESHOLD SELECTION RULE (STRICTLY VALIDATION DATA ONLY)
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 4: APPLYING THRESHOLD SELECTION RULE")
    print("="*60)
    
    # Primary criterion: maximize melanoma F1
    max_f1 = sweep_df["melanoma_f1"].max()
    print(f"Maximum validation Melanoma F1 : {max_f1:.4f}")
    
    # Effective tie definition: within 0.005 of max F1 (statistical noise margin)
    F1_TOLERANCE = 0.005
    tied_df = sweep_df[sweep_df["melanoma_f1"] >= (max_f1 - F1_TOLERANCE)].copy()
    print(f"Thresholds within {F1_TOLERANCE} of max F1 (effectively tied set):")
    print(tied_df[["threshold", "melanoma_f1", "melanoma_recall", "accuracy"]].to_string(index=False))
    
    # Tie-break rule 1: higher melanoma recall
    max_rec = tied_df["melanoma_recall"].max()
    best_rec_df = tied_df[tied_df["melanoma_recall"] == max_rec].copy()
    print(f"\nHighest Melanoma Recall among effectively tied set: {max_rec:.4f}")
    print(best_rec_df[["threshold", "melanoma_f1", "melanoma_recall", "accuracy"]].to_string(index=False))
    
    # Secondary tie-break: higher overall accuracy
    best_acc = best_rec_df["accuracy"].max()
    final_selected_df = best_rec_df[best_rec_df["accuracy"] == best_acc].iloc[0]
    
    selected_threshold = float(final_selected_df["threshold"])
    print(f"\n>>> SELECTED THRESHOLD: {selected_threshold:.2f} <<<")
    print(f"Selection Rationale:")
    print(f"  1. Primary Criterion : Within {F1_TOLERANCE} of max Melanoma F1 ({max_f1:.4f})")
    print(f"  2. Tie-break Rule 1  : Highest Melanoma Recall in candidate set ({max_rec:.4f})")
    print(f"  3. Tie-break Rule 2  : Highest Accuracy ({best_acc:.4f})")
    
    # Save selected threshold artifact
    selected_th_artifact = {
        "selected_threshold": selected_threshold,
        "selection_dataset": "validation (1,010 images)",
        "primary_criterion": "maximize melanoma F1",
        "max_validation_melanoma_f1": float(max_f1),
        "effective_tie_tolerance": F1_TOLERANCE,
        "tie_break_1": "maximize melanoma recall",
        "tie_break_2": "maximize overall accuracy",
        "validation_metrics_at_selected_threshold": {
            "accuracy": float(final_selected_df["accuracy"]),
            "balanced_accuracy": float(final_selected_df["balanced_accuracy"]),
            "macro_precision": float(final_selected_df["macro_precision"]),
            "macro_recall": float(final_selected_df["macro_recall"]),
            "macro_f1": float(final_selected_df["macro_f1"]),
            "weighted_f1": float(final_selected_df["weighted_f1"]),
            "melanoma_precision": float(final_selected_df["melanoma_precision"]),
            "melanoma_recall": float(final_selected_df["melanoma_recall"]),
            "melanoma_f1": float(final_selected_df["melanoma_f1"]),
            "melanoma_tp": int(final_selected_df["melanoma_tp"]),
            "melanoma_false_positives": int(final_selected_df["melanoma_false_positives"]),
            "melanoma_to_nv_errors": int(final_selected_df["melanoma_to_nv_errors"]),
            "melanoma_to_bkl_errors": int(final_selected_df["melanoma_to_bkl_errors"]),
            "nv_recall": float(final_selected_df["nv_recall"]),
            "bcc_recall": float(final_selected_df["bcc_recall"]),
            "bkl_recall": float(final_selected_df["bkl_recall"])
        }
    }
    
    with open(os.path.join(RESULTS_DIR, "selected_threshold.json"), "w") as f:
        json.dump(selected_th_artifact, f, indent=4)
        
    with open(os.path.join(RESULTS_DIR, "validation_metrics.json"), "w") as f:
        json.dump(selected_th_artifact["validation_metrics_at_selected_threshold"], f, indent=4)
        
    # =========================================================================
    # STEP 5: FINAL EVALUATION ON HELD-OUT TEST SET (988 IMAGES)
    # Exactly ONCE on test set. Test set was NOT inspected during threshold selection.
    # =========================================================================
    print("\n" + "="*60)
    print("STEP 5: SINGLE FINAL EVALUATION ON HELD-OUT TEST SET (988 IMAGES)")
    print(f"Threshold is permanently frozen at: {selected_threshold:.2f}")
    print("="*60)
    
    test_dataset = HAM10000Dataset(test_df, transform=get_transforms(is_train=False))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)
    
    test_probs = []
    test_targets = []
    test_start = time.time()
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Test inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            test_probs.append(probs.cpu().numpy())
            test_targets.append(targets.numpy())
            
    test_probs = np.vstack(test_probs)
    test_targets = np.concatenate(test_targets)
    test_duration = time.time() - test_start
    print(f"Test inference completed in {test_duration:.2f}s ({len(test_targets)} samples)")
    
    # ── Evaluation A: Original Experiment 8 Argmax Rule ──
    argmax_preds = np.argmax(test_probs, axis=1)
    metrics_argmax = compute_metrics_for_predictions(test_targets, argmax_preds, class_names)
    
    # ── Evaluation B: Experiment 10 Threshold-Tuned Decision Rule ──
    tuned_preds = []
    for p in test_probs:
        if p[mel_idx] >= selected_threshold:
            tuned_preds.append(mel_idx)
        else:
            tuned_preds.append(int(np.argmax(p)))
    tuned_preds = np.array(tuned_preds)
    metrics_tuned = compute_metrics_for_predictions(test_targets, tuned_preds, class_names)
    
    # Save classification reports
    with open(os.path.join(RESULTS_DIR, "classification_report_argmax.txt"), "w") as f:
        f.write(metrics_argmax["classification_report_text"])
    with open(os.path.join(RESULTS_DIR, "classification_report_threshold_tuned.txt"), "w") as f:
        f.write(metrics_tuned["classification_report_text"])
        
    # Save test metrics JSONs
    with open(os.path.join(RESULTS_DIR, "test_metrics_argmax.json"), "w") as f:
        json.dump(metrics_argmax, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "test_metrics_threshold_tuned.json"), "w") as f:
        json.dump(metrics_tuned, f, indent=4)
        
    # Plot and save confusion matrices
    plot_and_save_cm(
        np.array(metrics_argmax["confusion_matrix"]), class_names,
        "Experiment 8 (Argmax) Confusion Matrix - Test Split",
        os.path.join(RESULTS_DIR, "confusion_matrix_argmax.png")
    )
    plot_and_save_cm(
        np.array(metrics_tuned["confusion_matrix"]), class_names,
        f"Experiment 10 (Melanoma Threshold = {selected_threshold:.2f}) Confusion Matrix - Test Split",
        os.path.join(RESULTS_DIR, "confusion_matrix_threshold_tuned.png")
    )
    
    # =========================================================================
    # STEP 6: COMPARISONS AND DELTAS
    # =========================================================================
    # Comparison with Experiment 8
    exp8_metrics_path = os.path.join(AI_DIR, "results", "experiment8", "metrics.json")
    with open(exp8_metrics_path, "r") as f:
        exp8_baseline = json.load(f)
        
    delta_exp10_minus_exp8 = {
        "accuracy_change": metrics_tuned["accuracy"] - metrics_argmax["accuracy"],
        "balanced_accuracy_change": metrics_tuned["balanced_accuracy"] - metrics_argmax["balanced_accuracy"],
        "macro_f1_change": metrics_tuned["macro_f1"] - metrics_argmax["macro_f1"],
        "macro_precision_change": metrics_tuned["macro_precision"] - metrics_argmax["macro_precision"],
        "macro_recall_change": metrics_tuned["macro_recall"] - metrics_argmax["macro_recall"],
        "weighted_f1_change": metrics_tuned["weighted_f1"] - metrics_argmax["weighted_f1"],
        "melanoma_recall_change": metrics_tuned["melanoma_recall"] - metrics_argmax["melanoma_recall"],
        "melanoma_precision_change": metrics_tuned["melanoma_precision"] - metrics_argmax["melanoma_precision"],
        "melanoma_f1_change": metrics_tuned["melanoma_f1"] - metrics_argmax["melanoma_f1"],
        "melanoma_tp_change": metrics_tuned["melanoma_tp"] - metrics_argmax["melanoma_tp"],
        "melanoma_fp_change": metrics_tuned["melanoma_false_positives"] - metrics_argmax["melanoma_false_positives"],
        "mel_to_nv_error_change": metrics_tuned["melanoma_to_nv_errors"] - metrics_argmax["melanoma_to_nv_errors"],
        "mel_to_bkl_error_change": metrics_tuned["melanoma_to_bkl_errors"] - metrics_argmax["melanoma_to_bkl_errors"],
        "nv_recall_change": metrics_tuned["nv_recall"] - metrics_argmax["nv_recall"],
        "nv_precision_change": metrics_tuned["nv_precision"] - metrics_argmax["nv_precision"],
        "nv_f1_change": metrics_tuned["nv_f1"] - metrics_argmax["nv_f1"],
        "bcc_recall_change": metrics_tuned["bcc_recall"] - metrics_argmax["bcc_recall"],
        "bkl_recall_change": metrics_tuned["bkl_recall"] - metrics_argmax["bkl_recall"]
    }
    
    comp_exp8 = {
        "baseline_experiment8_argmax": {
            "accuracy": metrics_argmax["accuracy"],
            "balanced_accuracy": metrics_argmax["balanced_accuracy"],
            "macro_f1": metrics_argmax["macro_f1"],
            "weighted_f1": metrics_argmax["weighted_f1"],
            "melanoma_recall": metrics_argmax["melanoma_recall"],
            "melanoma_precision": metrics_argmax["melanoma_precision"],
            "melanoma_f1": metrics_argmax["melanoma_f1"],
            "melanoma_tp": metrics_argmax["melanoma_tp"],
            "melanoma_false_positives": metrics_argmax["melanoma_false_positives"],
            "mel_to_nv_errors": metrics_argmax["melanoma_to_nv_errors"],
            "mel_to_bkl_errors": metrics_argmax["melanoma_to_bkl_errors"],
            "nv_recall": metrics_argmax["nv_recall"]
        },
        "experiment10_threshold_tuned": {
            "threshold": selected_threshold,
            "accuracy": metrics_tuned["accuracy"],
            "balanced_accuracy": metrics_tuned["balanced_accuracy"],
            "macro_f1": metrics_tuned["macro_f1"],
            "weighted_f1": metrics_tuned["weighted_f1"],
            "melanoma_recall": metrics_tuned["melanoma_recall"],
            "melanoma_precision": metrics_tuned["melanoma_precision"],
            "melanoma_f1": metrics_tuned["melanoma_f1"],
            "melanoma_tp": metrics_tuned["melanoma_tp"],
            "melanoma_false_positives": metrics_tuned["melanoma_false_positives"],
            "mel_to_nv_errors": metrics_tuned["melanoma_to_nv_errors"],
            "mel_to_bkl_errors": metrics_tuned["melanoma_to_bkl_errors"],
            "nv_recall": metrics_tuned["nv_recall"]
        },
        "delta_exp10_minus_exp8": delta_exp10_minus_exp8
    }
    with open(os.path.join(RESULTS_DIR, "comparison_with_experiment8.json"), "w") as f:
        json.dump(comp_exp8, f, indent=4)
        
    # Comparison with Experiment 9
    exp9_metrics_path = os.path.join(AI_DIR, "results", "experiment9", "metrics.json")
    with open(exp9_metrics_path, "r") as f:
        exp9_baseline = json.load(f)
        
    delta_exp10_minus_exp9 = {
        "accuracy_change": metrics_tuned["accuracy"] - exp9_baseline["accuracy"],
        "balanced_accuracy_change": metrics_tuned["balanced_accuracy"] - exp9_baseline["balanced_accuracy"],
        "macro_f1_change": metrics_tuned["macro_f1"] - exp9_baseline["macro_f1"],
        "weighted_f1_change": metrics_tuned["weighted_f1"] - exp9_baseline["weighted_f1"],
        "melanoma_recall_change": metrics_tuned["melanoma_recall"] - exp9_baseline["melanoma_recall"],
        "melanoma_precision_change": metrics_tuned["melanoma_precision"] - exp9_baseline["melanoma_precision"],
        "melanoma_f1_change": metrics_tuned["melanoma_f1"] - exp9_baseline["melanoma_f1"]
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
        "experiment10_threshold_tuned": {
            "threshold": selected_threshold,
            "accuracy": metrics_tuned["accuracy"],
            "balanced_accuracy": metrics_tuned["balanced_accuracy"],
            "macro_f1": metrics_tuned["macro_f1"],
            "weighted_f1": metrics_tuned["weighted_f1"],
            "melanoma_recall": metrics_tuned["melanoma_recall"],
            "melanoma_precision": metrics_tuned["melanoma_precision"],
            "melanoma_f1": metrics_tuned["melanoma_f1"]
        },
        "delta_exp10_minus_exp9": delta_exp10_minus_exp9
    }
    with open(os.path.join(RESULTS_DIR, "comparison_with_experiment9.json"), "w") as f:
        json.dump(comp_exp9, f, indent=4)
        
    # Save configuration artifact
    inf_config = {
        "checkpoint_path": CHECKPOINT_PATH,
        "architecture_name": "DermaAI_MobileNetV3 (MobileNetV3-Large + CBAM)",
        "total_parameters": 4326297,
        "num_classes": 7,
        "class_mapping": CLASS_MAPPING,
        "dataset_split_counts": {
            "train": 8017,
            "validation": len(val_df),
            "test": len(test_df)
        },
        "threshold_grid": THRESHOLD_GRID,
        "selected_threshold": selected_threshold,
        "threshold_selection_criterion": "Maximize Validation Melanoma F1 with higher recall tie-break (tolerance=0.005)",
        "decision_rule": "if P(mel) >= threshold: mel else: argmax(probabilities)",
        "random_seed": SEED,
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_inference_duration_seconds": val_duration,
        "test_inference_duration_seconds": test_duration,
        "total_experiment_duration_seconds": time.time() - start_time
    }
    with open(os.path.join(RESULTS_DIR, "inference_configuration.json"), "w") as f:
        json.dump(inf_config, f, indent=4)
        
    # Save experiment summary
    exp_summary = {
        "experiment": "Experiment 10: Post-Hoc Melanoma Decision Threshold Tuning",
        "base_model": "Experiment 8 (Tempered Class Weights alpha=0.75)",
        "selected_threshold": selected_threshold,
        "test_performance_argmax": metrics_argmax,
        "test_performance_threshold_tuned": metrics_tuned,
        "delta_tuned_vs_argmax": delta_exp10_minus_exp8,
        "runtime": inf_config
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(exp_summary, f, indent=4)
        
    # Print formatted comparison summary
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION RESULTS: EXPERIMENT 10 vs EXPERIMENT 8 BASELINE")
    print("="*70)
    print(f"  Selected Melanoma Threshold : {selected_threshold:.2f}")
    print(f"  Test Accuracy               : {metrics_argmax['accuracy']*100:.2f}% -> {metrics_tuned['accuracy']*100:.2f}%  (delta: {delta_exp10_minus_exp8['accuracy_change']*100:+.2f}%)")
    print(f"  Balanced Accuracy           : {metrics_argmax['balanced_accuracy']*100:.2f}% -> {metrics_tuned['balanced_accuracy']*100:.2f}%  (delta: {delta_exp10_minus_exp8['balanced_accuracy_change']*100:+.2f}%)")
    print(f"  Macro F1                    : {metrics_argmax['macro_f1']:.4f} -> {metrics_tuned['macro_f1']:.4f}  (delta: {delta_exp10_minus_exp8['macro_f1_change']:+.4f})")
    print(f"  Weighted F1                 : {metrics_argmax['weighted_f1']:.4f} -> {metrics_tuned['weighted_f1']:.4f}  (delta: {delta_exp10_minus_exp8['weighted_f1_change']:+.4f})")
    print(f"  Melanoma Recall             : {metrics_argmax['melanoma_recall']*100:.2f}% -> {metrics_tuned['melanoma_recall']*100:.2f}%  (delta: {delta_exp10_minus_exp8['melanoma_recall_change']*100:+.2f}%)")
    print(f"  Melanoma Precision          : {metrics_argmax['melanoma_precision']*100:.2f}% -> {metrics_tuned['melanoma_precision']*100:.2f}%  (delta: {delta_exp10_minus_exp8['melanoma_precision_change']*100:+.2f}%)")
    print(f"  Melanoma F1                 : {metrics_argmax['melanoma_f1']:.4f} -> {metrics_tuned['melanoma_f1']:.4f}  (delta: {delta_exp10_minus_exp8['melanoma_f1_change']:+.4f})")
    print(f"  Melanoma TP / 106           : {metrics_argmax['melanoma_tp']} -> {metrics_tuned['melanoma_tp']}  (delta: {delta_exp10_minus_exp8['melanoma_tp_change']:+d})")
    print(f"  Melanoma False Positives    : {metrics_argmax['melanoma_false_positives']} -> {metrics_tuned['melanoma_false_positives']}  (delta: {delta_exp10_minus_exp8['melanoma_fp_change']:+d})")
    print(f"  Mel -> NV Misses            : {metrics_argmax['melanoma_to_nv_errors']} -> {metrics_tuned['melanoma_to_nv_errors']}  (delta: {delta_exp10_minus_exp8['mel_to_nv_error_change']:+d})")
    print(f"  Mel -> BKL Misses           : {metrics_argmax['melanoma_to_bkl_errors']} -> {metrics_tuned['melanoma_to_bkl_errors']}  (delta: {delta_exp10_minus_exp8['mel_to_bkl_error_change']:+d})")
    print(f"  Nevus (NV) Recall           : {metrics_argmax['nv_recall']*100:.2f}% -> {metrics_tuned['nv_recall']*100:.2f}%  (delta: {delta_exp10_minus_exp8['nv_recall_change']*100:+.2f}%)")
    print("="*70)
    print(f"All 14 Experiment 10 artifacts saved to: {RESULTS_DIR}")
    print("Experiment 10 complete. STOPPING as required -- awaiting next instruction.")


if __name__ == "__main__":
    main()
