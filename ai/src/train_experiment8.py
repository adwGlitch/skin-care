import os
import sys
import json
import time
import random
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    precision_recall_fscore_support, f1_score
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
RESULTS_DIR   = os.path.join(AI_DIR, "results", "experiment8")

EXPERIMENT_NAME = "experiment8_tempered_class_weights"
CHECKPOINT_NAME = "experiment8_best_model.pt"

EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 5
ALPHA         = 0.75

# ── Baseline Loaders for Experiments 1 through 7 ───────────────────────────────
BASELINES = {
    f"exp{i}": {
        "name": f"Experiment {i}",
        "path": os.path.join(AI_DIR, "results", f"experiment{i}", "metrics.json"),
    } for i in range(1, 8)
}


def load_baseline_metrics(exp_key):
    info = BASELINES[exp_key]
    if os.path.exists(info["path"]):
        try:
            with open(info["path"], "r") as f:
                d = json.load(f)
                per_class = d.get("per_class", {})
                bcc_rec = per_class.get("bcc", {}).get("recall") if "bcc" in per_class else None
                bkl_rec = per_class.get("bkl", {}).get("recall") if "bkl" in per_class else None
                cm = d.get("confusion_matrix")
                mel_to_nv = d.get("melanoma_to_nv_errors")
                mel_to_bkl = d.get("melanoma_to_bkl_errors")
                nv_rec = d.get("nv_recall")
                if cm is not None and len(cm) >= 7:
                    if mel_to_nv is None:
                        mel_to_nv = int(cm[4][5])
                    if mel_to_bkl is None:
                        mel_to_bkl = int(cm[4][2])
                    if nv_rec is None:
                        nv_total = sum(cm[5])
                        nv_rec = float(cm[5][5] / nv_total) if nv_total > 0 else 0.0
                    if bcc_rec is None:
                        bcc_total = sum(cm[1])
                        bcc_rec = float(cm[1][1] / bcc_total) if bcc_total > 0 else 0.0
                    if bkl_rec is None:
                        bkl_total = sum(cm[2])
                        bkl_rec = float(cm[2][2] / bkl_total) if bkl_total > 0 else 0.0
                return {
                    "accuracy": d.get("accuracy"),
                    "balanced_accuracy": d.get("balanced_accuracy"),
                    "macro_precision": d.get("macro_precision"),
                    "macro_recall": d.get("macro_recall"),
                    "macro_f1": d.get("macro_f1"),
                    "weighted_f1": d.get("weighted_f1"),
                    "melanoma_precision": d.get("melanoma_precision"),
                    "melanoma_recall": d.get("melanoma_recall"),
                    "melanoma_f1": d.get("melanoma_f1"),
                    "mel_to_nv_errors": mel_to_nv,
                    "mel_to_bkl_errors": mel_to_bkl,
                    "nv_recall": nv_rec,
                    "bcc_recall": bcc_rec,
                    "bkl_recall": bkl_rec
                }
        except Exception as e:
            print(f"Warning reading baseline {exp_key}: {e}")
    return {}


# ── Calculate Tempered Class Weights from TRAINING SET ONLY ─────────────────────
def calculate_tempered_class_weights(train_df, device, alpha=0.75):
    """
    Computes tempered and normalized class weights strictly from the training set.
    1. Base weights: w_c = N_train / (K * N_c)
    2. Tempered weights: w_c^(tempered) = w_c^alpha
    3. Normalization factor: E_p[w^(tempered)] = sum_j p[j] * w_j^(tempered) where p[j] = N_train[j] / N_train
    4. Normalized weights: w_c^(norm) = w_c^(tempered) / E_p[w^(tempered)]
    Ensures sum_c p[c] * w_c^(norm) == 1.0.
    """
    print("\n" + "="*60)
    print("CALCULATING TEMPERED CLASS WEIGHTS (STRICTLY TRAINING SET ONLY)")
    print("="*60)
    train_counts = train_df['dx'].value_counts()
    total_train  = len(train_df)
    num_classes  = len(CLASS_MAPPING)

    orig_weights = {}
    p_c          = {}
    tempered_raw = {}

    for cls, idx in sorted(CLASS_MAPPING.items(), key=lambda x: x[1]):
        count = int(train_counts.get(cls, 0))
        p = count / total_train
        p_c[cls] = p
        w = total_train / (num_classes * count) if count > 0 else 1.0
        orig_weights[cls] = w
        tempered_raw[cls] = w ** alpha

    # Expected value under training class distribution
    expected_val = sum(p_c[cls] * tempered_raw[cls] for cls in CLASS_MAPPING)

    # Normalized tempered weights
    norm_weights = {cls: float(tempered_raw[cls] / expected_val) for cls in CLASS_MAPPING}
    weights_list = [norm_weights[REVERSE_CLASS_MAPPING[i]] for i in range(num_classes)]
    weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)

    weighted_mean = sum(p_c[cls] * norm_weights[cls] for cls in CLASS_MAPPING)

    print(f"Total training samples : {total_train}")
    print(f"Number of classes      : {num_classes}")
    print(f"Tempering exponent (alpha) : {alpha}")
    print("Per-class counts, original weights, and tempered normalized weights:")
    for cls, idx in sorted(CLASS_MAPPING.items(), key=lambda x: x[1]):
        print(f"  Class {idx} [{cls:>5}]: count = {train_counts[cls]:>4} ({p_c[cls]*100:>5.2f}%) | "
              f"Orig w = {orig_weights[cls]:.4f} | "
              f"Tempered (raw) = {tempered_raw[cls]:.4f} | "
              f"Normalized w = {norm_weights[cls]:.4f}")

    print(f"\nVerification Check: sum(p[c] * w_norm[c]) = {weighted_mean:.6f}  (Expected ~ 1.000000)")
    print("="*60 + "\n")

    weight_config = {
        "alpha": alpha,
        "total_train_samples": total_train,
        "normalization_formula": "w_norm[c] = (w_orig[c]^alpha) / sum(p[j] * w_orig[j]^alpha)",
        "expected_weighted_mean": weighted_mean,
        "original_weights": orig_weights,
        "tempered_raw_weights": tempered_raw,
        "normalized_tempered_weights": norm_weights,
        "class_probabilities_p_c": p_c
    }

    return weights_tensor, norm_weights, weight_config


# ── Test Set Evaluation (Held-Out 988 Images) ──────────────────────────────────
def run_test_evaluation(model, device, test_df, results_dir, checkpoint_path):
    print("\n" + "="*60)
    print("TEST SET EVALUATION (HELD-OUT TEST SPLIT -- 988 IMAGES)")
    print("="*60)

    test_dataset = HAM10000Dataset(test_df, transform=get_transforms(is_train=False))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # Load best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Test inference"):
            inputs  = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    class_names = [REVERSE_CLASS_MAPPING[i] for i in range(7)]

    acc   = accuracy_score(all_targets, all_preds)
    bacc  = balanced_accuracy_score(all_targets, all_preds)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0)

    report_dict = classification_report(
        all_targets, all_preds, target_names=class_names,
        output_dict=True, zero_division=0)
    report_text = classification_report(
        all_targets, all_preds, target_names=class_names, zero_division=0)

    cm = confusion_matrix(all_targets, all_preds)

    # Class Indices
    mel_idx = CLASS_MAPPING['mel']
    nv_idx  = CLASS_MAPPING['nv']
    bkl_idx = CLASS_MAPPING['bkl']
    bcc_idx = CLASS_MAPPING['bcc']

    mel_total = int(sum(all_targets[i] == mel_idx for i in range(len(all_targets))))
    mel_correct = int(cm[mel_idx][mel_idx])
    mel_to_nv_errors = int(cm[mel_idx][nv_idx])
    mel_to_bkl_errors = int(cm[mel_idx][bkl_idx])

    nv_total = int(sum(all_targets[i] == nv_idx for i in range(len(all_targets))))
    nv_correct = int(cm[nv_idx][nv_idx])
    nv_recall = float(nv_correct / nv_total) if nv_total > 0 else 0.0
    nv_prec = float(report_dict['nv']['precision'])
    nv_f1 = float(report_dict['nv']['f1-score'])

    mel_rec  = float(report_dict["mel"]["recall"])
    mel_prec = float(report_dict["mel"]["precision"])
    mel_f1   = float(report_dict["mel"]["f1-score"])

    bcc_rec = float(report_dict["bcc"]["recall"])
    bkl_rec = float(report_dict["bkl"]["recall"])

    print(f"\n{'--------------------------------------------------'}")
    print(f"  Test Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy      : {bacc:.4f}  ({bacc*100:.2f}%)")
    print(f"  Macro Precision        : {prec_macro:.4f}")
    print(f"  Macro Recall           : {rec_macro:.4f}")
    print(f"  Macro F1               : {f1_macro:.4f}")
    print(f"  Weighted Precision     : {prec_weighted:.4f}")
    print(f"  Weighted Recall        : {rec_weighted:.4f}")
    print(f"  Weighted F1            : {f1_weighted:.4f}")
    print(f"{'--------------------------------------------------'}")
    print(f"  Melanoma (mel) Recall  : {mel_rec:.4f}  ({mel_rec*100:.2f}% -- {mel_correct}/{mel_total} images)")
    print(f"  Melanoma (mel) Prec    : {mel_prec:.4f}")
    print(f"  Melanoma (mel) F1      : {mel_f1:.4f}")
    print(f"  mel -> nv errors       : {mel_to_nv_errors} images")
    print(f"  mel -> bkl errors      : {mel_to_bkl_errors} images")
    print(f"  Majority (nv) Recall   : {nv_recall:.4f}  ({nv_correct}/{nv_total} images)")
    print(f"  Majority (nv) Prec     : {nv_prec:.4f}")
    print(f"  Majority (nv) F1       : {nv_f1:.4f}")
    print(f"  BCC Recall             : {bcc_rec:.4f}  ({cm[bcc_idx][bcc_idx]}/{report_dict['bcc']['support']:.0f} images)")
    print(f"  BKL Recall             : {bkl_rec:.4f}  ({cm[bkl_idx][bkl_idx]}/{report_dict['bkl']['support']:.0f} images)")
    print(f"{'--------------------------------------------------'}")

    print("\nPer-class Metrics:")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    # Save text report
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write("EXPERIMENT 8 -- TEMPERED CLASS-WEIGHTED CROSS-ENTROPY (alpha=0.75)\n")
        f.write("Test Set Evaluation (Held-Out Test Split -- 988 Images)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Backbone          : MobileNetV3-Large + CBAM Attention\n")
        f.write(f"Loss Strategy     : Tempered Class-Weighted CE (alpha=0.75, normalized)\n")
        f.write(f"Test Accuracy     : {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Balanced Accuracy : {bacc:.4f} ({bacc*100:.2f}%)\n")
        f.write(f"Macro Precision   : {prec_macro:.4f}\n")
        f.write(f"Macro Recall      : {rec_macro:.4f}\n")
        f.write(f"Macro F1          : {f1_macro:.4f}\n")
        f.write(f"Weighted Precision: {prec_weighted:.4f}\n")
        f.write(f"Weighted Recall   : {rec_weighted:.4f}\n")
        f.write(f"Weighted F1       : {f1_weighted:.4f}\n\n")
        f.write(f"Melanoma Recall   : {mel_rec:.4f} ({mel_rec*100:.2f}% -- {mel_correct}/{mel_total} images)\n")
        f.write(f"Melanoma Precision: {mel_prec:.4f}\n")
        f.write(f"Melanoma F1       : {mel_f1:.4f}\n")
        f.write(f"mel -> nv errors  : {mel_to_nv_errors} images\n")
        f.write(f"mel -> bkl errors : {mel_to_bkl_errors} images\n\n")
        f.write(f"nv (majority) rec : {nv_recall:.4f} ({nv_correct}/{nv_total} images)\n")
        f.write(f"nv (majority) prec: {nv_prec:.4f}\n")
        f.write(f"nv (majority) f1  : {nv_f1:.4f}\n\n")
        f.write(f"bcc recall        : {bcc_rec:.4f}\n")
        f.write(f"bkl recall        : {bkl_rec:.4f}\n\n")
        f.write(f"Classification Report:\n{report_text}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # Save JSON metrics
    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
        "melanoma_recall": float(mel_rec),
        "melanoma_precision": float(mel_prec),
        "melanoma_f1": float(mel_f1),
        "melanoma_correct_images": mel_correct,
        "melanoma_total_images": mel_total,
        "melanoma_to_nv_errors": mel_to_nv_errors,
        "melanoma_to_bkl_errors": mel_to_bkl_errors,
        "nv_recall": float(nv_recall),
        "nv_precision": float(nv_prec),
        "nv_f1": float(nv_f1),
        "nv_correct_images": nv_correct,
        "nv_total_images": nv_total,
        "bcc_recall": float(bcc_rec),
        "bkl_recall": float(bkl_rec),
        "per_class": report_dict,
        "confusion_matrix": cm.tolist(),
    }
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix -- Test Set (Experiment 8: Tempered Class Weights alpha=0.75)')
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved evaluation artifacts -> {results_dir}")
    return metrics


# ── Comparison Helper Across All Prior Experiments (1 through 7) ───────────────
def compute_comparisons(exp8_metrics, results_dir):
    comparisons = {}
    for exp_key in [f"exp{i}" for i in range(1, 8)]:
        base = load_baseline_metrics(exp_key)
        diff = {
            "accuracy_change": float(exp8_metrics["accuracy"] - base["accuracy"]) if base.get("accuracy") is not None else None,
            "balanced_accuracy_change": float(exp8_metrics["balanced_accuracy"] - base["balanced_accuracy"]) if base.get("balanced_accuracy") is not None else None,
            "macro_f1_change": float(exp8_metrics["macro_f1"] - base["macro_f1"]) if base.get("macro_f1") is not None else None,
            "macro_precision_change": float(exp8_metrics["macro_precision"] - base["macro_precision"]) if base.get("macro_precision") is not None else None,
            "macro_recall_change": float(exp8_metrics["macro_recall"] - base["macro_recall"]) if base.get("macro_recall") is not None else None,
            "weighted_f1_change": float(exp8_metrics["weighted_f1"] - base["weighted_f1"]) if base.get("weighted_f1") is not None else None,
            "melanoma_recall_change": float(exp8_metrics["melanoma_recall"] - base["melanoma_recall"]) if base.get("melanoma_recall") is not None else None,
            "melanoma_precision_change": float(exp8_metrics["melanoma_precision"] - base["melanoma_precision"]) if base.get("melanoma_precision") is not None else None,
            "melanoma_f1_change": float(exp8_metrics["melanoma_f1"] - base["melanoma_f1"]) if base.get("melanoma_f1") is not None else None,
            "mel_to_nv_error_change": int(exp8_metrics["melanoma_to_nv_errors"] - base["mel_to_nv_errors"]) if base.get("mel_to_nv_errors") is not None else None,
            "mel_to_bkl_error_change": int(exp8_metrics["melanoma_to_bkl_errors"] - base["mel_to_bkl_errors"]) if base.get("mel_to_bkl_errors") is not None else None,
            "nv_recall_change": float(exp8_metrics["nv_recall"] - base["nv_recall"]) if base.get("nv_recall") is not None else None,
            "bcc_recall_change": float(exp8_metrics["bcc_recall"] - base["bcc_recall"]) if base.get("bcc_recall") is not None else None,
            "bkl_recall_change": float(exp8_metrics["bkl_recall"] - base["bkl_recall"]) if base.get("bkl_recall") is not None else None,
        }
        
        comp_data = {
            "baseline": base,
            "experiment8_tempered_weights": {
                "accuracy": exp8_metrics["accuracy"],
                "balanced_accuracy": exp8_metrics["balanced_accuracy"],
                "macro_precision": exp8_metrics["macro_precision"],
                "macro_recall": exp8_metrics["macro_recall"],
                "macro_f1": exp8_metrics["macro_f1"],
                "weighted_f1": exp8_metrics["weighted_f1"],
                "melanoma_recall": exp8_metrics["melanoma_recall"],
                "melanoma_precision": exp8_metrics["melanoma_precision"],
                "melanoma_f1": exp8_metrics["melanoma_f1"],
                "mel_to_nv_errors": exp8_metrics["melanoma_to_nv_errors"],
                "mel_to_bkl_errors": exp8_metrics["melanoma_to_bkl_errors"],
                "nv_recall": exp8_metrics["nv_recall"],
                "bcc_recall": exp8_metrics["bcc_recall"],
                "bkl_recall": exp8_metrics["bkl_recall"],
            },
            f"delta_exp8_minus_{exp_key}": diff
        }

        idx_str = exp_key.replace("exp", "")
        file_path = os.path.join(results_dir, f"comparison_with_experiment{idx_str}.json")
        with open(file_path, "w") as f:
            json.dump(comp_data, f, indent=4)
        comparisons[exp_key] = comp_data

    # Print comparison against primary baseline (Experiment 2)
    exp2_diff = comparisons["exp2"]["delta_exp8_minus_exp2"]
    exp2_base = comparisons["exp2"]["baseline"]
    print("\n" + "="*60)
    print("COMPARISON: EXPERIMENT 8 vs EXPERIMENT 2 (PRIMARY BASELINE)")
    print("="*60)
    print(f"  Test Accuracy      : {exp2_base['accuracy']*100:.2f}% -> {exp8_metrics['accuracy']*100:.2f}%  (delta: {exp2_diff['accuracy_change']*100:+.2f}%)")
    print(f"  Balanced Accuracy  : {exp2_base['balanced_accuracy']*100:.2f}% -> {exp8_metrics['balanced_accuracy']*100:.2f}%  (delta: {exp2_diff['balanced_accuracy_change']*100:+.2f}%)")
    print(f"  Macro F1           : {exp2_base['macro_f1']:.4f} -> {exp8_metrics['macro_f1']:.4f}  (delta: {exp2_diff['macro_f1_change']:+.4f})")
    print(f"  Weighted F1        : {exp2_base['weighted_f1']:.4f} -> {exp8_metrics['weighted_f1']:.4f}  (delta: {exp2_diff['weighted_f1_change']:+.4f})")
    print(f"  Melanoma Recall    : {exp2_base['melanoma_recall']*100:.2f}% -> {exp8_metrics['melanoma_recall']*100:.2f}%  (delta: {exp2_diff['melanoma_recall_change']*100:+.2f}%)")
    print(f"  Melanoma Precision : {exp2_base['melanoma_precision']:.4f} -> {exp8_metrics['melanoma_precision']:.4f}  (delta: {exp2_diff['melanoma_precision_change']:+.4f})")
    mel_nv_d = exp2_diff.get('mel_to_nv_error_change')
    mel_nv_str = f"{mel_nv_d:+d}" if mel_nv_d is not None else "N/A"
    mel_bkl_d = exp2_diff.get('mel_to_bkl_error_change')
    mel_bkl_str = f"{mel_bkl_d:+d}" if mel_bkl_d is not None else "N/A"
    print(f"  Mel -> nv errors   : {exp2_base.get('mel_to_nv_errors', 'N/A')} -> {exp8_metrics['melanoma_to_nv_errors']}  (delta: {mel_nv_str})")
    print(f"  Mel -> bkl errors  : {exp2_base.get('mel_to_bkl_errors', 'N/A')} -> {exp8_metrics['melanoma_to_bkl_errors']}  (delta: {mel_bkl_str})")
    print(f"  Majority (nv) Rec  : {exp2_base['nv_recall']*100:.2f}% -> {exp8_metrics['nv_recall']*100:.2f}%  (delta: {exp2_diff['nv_recall_change']*100:+.2f}%)")
    if exp2_base.get('bcc_recall') is not None and exp2_diff.get('bcc_recall_change') is not None:
        print(f"  BCC Recall         : {exp2_base['bcc_recall']*100:.2f}% -> {exp8_metrics['bcc_recall']*100:.2f}%  (delta: {exp2_diff['bcc_recall_change']*100:+.2f}%)")
    if exp2_base.get('bkl_recall') is not None and exp2_diff.get('bkl_recall_change') is not None:
        print(f"  BKL Recall         : {exp2_base['bkl_recall']*100:.2f}% -> {exp8_metrics['bkl_recall']*100:.2f}%  (delta: {exp2_diff['bkl_recall_change']*100:+.2f}%)")
    print("="*60)
    return comparisons


# ── Main Training Routine ───────────────────────────────────────────────────────
def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    training_start = time.time()

    device = torch.device("cpu")
    print(f"Using device: {device} (CPU training environment)")
    print(f"Experiment  : {EXPERIMENT_NAME}")
    print(f"Random seed : {SEED}")

    # 1. Load splits
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    print(f"  Train: {len(train_df)} images | Val: {len(val_df)} images | Test: {len(test_df)} images")

    # 2. Compute Tempered Class Weights STRICTLY from training set only
    weights_tensor, norm_weights_dict, weight_config = calculate_tempered_class_weights(train_df, device, alpha=ALPHA)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Save class weights and tempering configuration
    with open(os.path.join(RESULTS_DIR, "training_class_weights.json"), "w") as f:
        json.dump(norm_weights_dict, f, indent=4)

    with open(os.path.join(RESULTS_DIR, "weight_tempering_configuration.json"), "w") as f:
        json.dump(weight_config, f, indent=4)

    # 3. Datasets and Loaders (keeping identical to Exp 2: natural random sampling, identical augmentation)
    train_dataset = HAM10000Dataset(train_df, transform=get_transforms(is_train=True))
    val_dataset   = HAM10000Dataset(val_df,   transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"Diagnostic -> Device: {device}, num_workers: 0, pin_memory: False, batch_size: {BATCH_SIZE}")

    # Smoke Test
    print("\n[SMOKE TEST] Fetching one batch from train_loader ...")
    try:
        smoke_imgs, smoke_labels = next(iter(train_loader))
        print(f"[SMOKE TEST] Image tensor shape : {smoke_imgs.shape}")
        print(f"[SMOKE TEST] Label tensor shape : {smoke_labels.shape}")
        smoke_imgs   = smoke_imgs.to(device)
        smoke_labels = smoke_labels.to(device)
        print(f"[SMOKE TEST] Batch transferred to {device} successfully.")
        print("[SMOKE TEST] PASSED -- proceeding to model creation.\n")
        del smoke_imgs, smoke_labels
    except Exception as smoke_err:
        import traceback
        print("\n[SMOKE TEST] FAILED -- aborting. Full traceback:")
        traceback.print_exc()
        raise SystemExit(1) from smoke_err

    # 4. Model Architecture: MobileNetV3-Large + CBAM Attention (Exact Exp 2 Architecture)
    model = DermaAI_MobileNetV3(num_classes=7, use_attention=True, pretrained=True)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())

    # Phase 1: Frozen backbone (Epochs 1-15)
    for param in model.features.parameters():
        param.requires_grad = False
    trainable_phase1 = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Phase 2: Unfrozen backbone (Epochs 16-30)
    for param in model.features.parameters():
        param.requires_grad = True
    trainable_phase2 = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Re-freeze for Phase 1
    for param in model.features.parameters():
        param.requires_grad = False

    # Print Pre-Training Verification (Requirement 9)
    print("="*60)
    print("REQUIRED PRE-TRAINING VERIFICATION (REQUIREMENT 9)")
    print("="*60)
    print(f"  Architecture                      : DermaAI_MobileNetV3 (MobileNetV3-Large + CBAM)")
    print(f"  Pretrained Initialization         : MobileNet_V3_Large_Weights.DEFAULT (ImageNet-1K)")
    print(f"  Total Parameters                  : {total_params:,}")
    print(f"  Frozen-Phase Trainable Parameters : {trainable_phase1:,}  (Backbone frozen, Epochs 1-15)")
    print(f"  Fine-Tuning Trainable Parameters  : {trainable_phase2:,}  (Full network, Epochs 16-30)")
    print(f"  Dataset Partition Sizes           : Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"  Random Seed                       : {SEED}")
    print(f"  Batch Size                        : {BATCH_SIZE}")
    print(f"  Original Exp 2 Weights            : {weight_config['original_weights']}")
    print(f"  Tempering Exponent (alpha)            : {ALPHA}")
    print(f"  Final Tempered Normalized Weights : {norm_weights_dict}")
    print(f"  Expected Mean under Train Dist    : {weight_config['expected_weighted_mean']:.6f} (~ 1.000000)")
    print(f"  Optimizer                         : AdamW (weight_decay={WEIGHT_DECAY})")
    print(f"  Learning Rates                    : Phase 1={LEARNING_RATE}, Phase 2={LEARNING_RATE * 0.1}")
    print(f"  Scheduler                         : ReduceLROnPlateau(mode=min, factor=0.1, patience={PATIENCE})")
    print(f"  Augmentation Configuration        : Flip, Rotation(20), ColorJitter(0.1, 0.1, 0.1)")
    print("="*60 + "\n")

    # Save architecture configuration and parameter summary artifacts
    arch_config = {
        "model_name": "DermaAI_MobileNetV3",
        "backbone": "MobileNetV3-Large",
        "attention": "CBAM",
        "pretrained_weights": "MobileNet_V3_Large_Weights.DEFAULT",
        "input_size": [3, 224, 224],
        "num_classes": 7,
        "total_parameters": total_params,
        "frozen_phase_epochs": "1-15",
        "fine_tuning_phase_epochs": "16-30"
    }
    with open(os.path.join(RESULTS_DIR, "architecture_configuration.json"), "w") as f:
        json.dump(arch_config, f, indent=4)

    param_summary = {
        "model_name": "DermaAI_MobileNetV3",
        "total_parameters": total_params,
        "frozen_phase_trainable_parameters": trainable_phase1,
        "frozen_phase_non_trainable_parameters": total_params - trainable_phase1,
        "fine_tuning_trainable_parameters": trainable_phase2,
        "fine_tuning_non_trainable_parameters": 0
    }
    with open(os.path.join(RESULTS_DIR, "parameter_summary.json"), "w") as f:
        json.dump(param_summary, f, indent=4)

    # 5. Optimizer & Scheduler (Identical to Experiment 2)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=PATIENCE)

    # 6. Training Loop (Exact Experiment 2 2-Phase Schedule)
    best_val_loss     = float('inf')
    best_val_epoch    = -1
    best_val_acc      = 0.0
    epochs_no_improve = 0
    checkpoint_path   = os.path.join(MODELS_DIR, CHECKPOINT_NAME)

    history = {
        'epoch': [],
        'phase': [],
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
        'trainable_params': [],
        'epoch_duration_seconds': []
    }

    print(f"Starting Experiment 8 training for up to {EPOCHS} epochs with tempered class weights...")
    print("="*60)

    for epoch in range(EPOCHS):
        epoch_num = epoch + 1
        epoch_start = time.time()
        current_phase = 1 if epoch_num <= 15 else 2

        # Unfreeze backbone at midpoint (Epoch 16+)
        if epoch == EPOCHS // 2:
            print("\n" + "="*60)
            print("==> ENTERING PHASE 2: Unfreezing entire backbone for fine-tuning (Epochs 16-30)...")
            print("="*60)
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=PATIENCE)
            epochs_no_improve = 0  # Reset patience for fine-tuning phase
            print(f"  Trainable params now: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

        # Train
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_num:02d}/{EPOCHS} [Train Phase {current_phase}]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0

        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch_num:02d}/{EPOCHS} [Val]              ")
            for inputs, targets in pbar_val:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
                val_loss    += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total   += targets.size(0)
                correct += (predicted == targets).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc  = correct / total
        current_lr     = optimizer.param_groups[0]['lr']
        epoch_elapsed  = time.time() - epoch_start
        cur_trainable  = sum(p.numel() for p in model.parameters() if p.requires_grad)

        history['epoch'].append(epoch_num)
        history['phase'].append(current_phase)
        history['train_loss'].append(float(epoch_train_loss))
        history['val_loss'].append(float(epoch_val_loss))
        history['val_acc'].append(float(epoch_val_acc))
        history['lr'].append(float(current_lr))
        history['trainable_params'].append(cur_trainable)
        history['epoch_duration_seconds'].append(float(epoch_elapsed))

        print(f"Epoch {epoch_num:02d} (Phase {current_phase}) | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_elapsed:.1f}s")

        scheduler.step(epoch_val_loss)

        # Checkpoint selection strictly by validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss  = epoch_val_loss
            best_val_epoch = epoch_num
            best_val_acc   = epoch_val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  OK Saved best model checkpoint (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered after {epochs_no_improve} epochs without val loss improvement in Phase {current_phase} (Epoch {epoch_num}).")
                break

    training_time_s = time.time() - training_start
    training_time_h = training_time_s / 3600

    print("="*60)
    print(f"Training complete -- best epoch: {best_val_epoch} | "
          f"best val loss: {best_val_loss:.4f} | "
          f"best val acc: {best_val_acc:.4f} | "
          f"total time: {training_time_s:.0f}s ({training_time_h:.2f}h)")

    # 7. Save history & generate training curves
    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    epochs_ran = history['epoch']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Loss curves
    axes[0].plot(epochs_ran, history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(epochs_ran, history['val_loss'],   label='Val Loss', color='red')
    axes[0].axvline(x=15.5, color='purple', linestyle='--', label='Unfreeze Backbone')
    axes[0].set_title('Training & Validation Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True)

    # Accuracy curves
    axes[1].plot(epochs_ran, history['val_acc'], color='green', label='Val Accuracy')
    axes[1].axvline(x=15.5, color='purple', linestyle='--')
    axes[1].set_title('Validation Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True)

    # Learning Rate curves
    axes[2].plot(epochs_ran, history['lr'], color='orange', label='Learning Rate')
    axes[2].axvline(x=15.5, color='purple', linestyle='--')
    axes[2].set_title('Learning Rate'); axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log'); axes[2].legend(); axes[2].grid(True)

    plt.suptitle('Experiment 8 -- Tempered Class-Weighted Training Curves (alpha=0.75)', fontsize=14)
    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"  Training curves -> {curves_path}")

    # 8. Single Test Evaluation
    metrics = run_test_evaluation(model, device, test_df, RESULTS_DIR, checkpoint_path)

    # 9. Comparisons against all prior experiments (1 through 7)
    comparisons = compute_comparisons(metrics, RESULTS_DIR)

    # 10. Experiment Summary & Reproducibility Metadata
    reproducibility = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": str(device),
        "random_seed": SEED,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "scheduler": "ReduceLROnPlateau(mode=min, factor=0.1, patience=5)",
        "model_name": "DermaAI_MobileNetV3",
        "pretrained_weights": "MobileNet_V3_Large_Weights.DEFAULT",
        "total_parameters": total_params,
        "dataset_split_counts": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df)
        },
        "tempering_alpha": ALPHA,
        "original_weights": weight_config["original_weights"],
        "final_tempered_weights": norm_weights_dict,
        "normalization_method": "Expected value under training distribution = 1.0",
        "timestamp_start": training_start,
        "total_duration_seconds": training_time_s,
        "total_duration_hours": training_time_h,
        "best_epoch": best_val_epoch
    }

    summary = {
        "experiment": EXPERIMENT_NAME,
        "config": {
            "epochs_configured": EPOCHS,
            "epochs_ran": len(epochs_ran),
            "batch_size": BATCH_SIZE,
            "loss": "tempered_class_weighted_cross_entropy",
            "alpha": ALPHA,
            "class_weights_source": "training_set_only",
            "class_weights": norm_weights_dict,
            "optimizer": "AdamW",
            "weight_decay": WEIGHT_DECAY,
            "backbone": "MobileNetV3-Large",
            "attention": "CBAM",
            "device": str(device),
            "random_seed": SEED,
            "num_workers": 0,
            "pin_memory": False,
        },
        "dataset": {
            "train_images": len(train_df),
            "val_images":   len(val_df),
            "test_images":  len(test_df),
        },
        "model": {
            "name": "DermaAI_MobileNetV3",
            "total_params": total_params,
            "frozen_phase_trainable_params": trainable_phase1,
            "fine_tuning_trainable_params": trainable_phase2
        },
        "training": {
            "best_epoch": best_val_epoch,
            "best_val_loss": float(best_val_loss),
            "best_val_acc": float(best_val_acc),
            "training_time_seconds": float(training_time_s),
            "training_time_hours": float(training_time_h),
        },
        "checkpoint": checkpoint_path,
        "results_dir": RESULTS_DIR,
        "test_metrics": metrics,
        "primary_comparison_exp2": comparisons["exp2"],
        "reproducibility": reproducibility
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'='*60}")
    print("EXPERIMENT 8 COMPLETE")
    print(f"{'='*60}")
    print(f"  Best epoch      : {best_val_epoch}")
    print(f"  Test Accuracy   : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Balanced Acc    : {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1     : {metrics['weighted_f1']:.4f}")
    print(f"  Melanoma Recall : {metrics['melanoma_recall']:.4f}")
    print(f"  Training time   : {training_time_s:.0f}s ({training_time_h:.2f}h)")
    print(f"  Checkpoint      : {checkpoint_path}")
    print(f"  Results dir     : {RESULTS_DIR}")
    print(f"{'='*60}")
    print("\nExperiment 8 finished. STOPPED -- awaiting next instruction.\n")


if __name__ == "__main__":
    train()
