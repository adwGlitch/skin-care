import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
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
    from .preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING, REVERSE_CLASS_MAPPING
except ImportError:
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
RESULTS_DIR   = os.path.join(AI_DIR, "results", "experiment6")

EXPERIMENT_NAME = "experiment6_architecture_improvement"
CHECKPOINT_NAME = "experiment6_best_model.pt"

EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
PATIENCE      = 5

# ── Prior Experiment Baselines ──────────────────────────────────────────────────
BASELINES = {
    "exp1": {
        "name": "Experiment 1 (Unweighted Baseline)",
        "path": os.path.join(AI_DIR, "results", "experiment1", "metrics.json"),
        "fallback": {
            "accuracy": 0.7317813765182186,
            "balanced_accuracy": 0.7307211250816138,
            "macro_precision": 0.6015076591786525,
            "macro_recall": 0.7307211250816138,
            "macro_f1": 0.6394575183439128,
            "weighted_f1": 0.7486763834648812,
            "melanoma_precision": 0.36923076923076925,
            "melanoma_recall": 0.4528301886792453,
            "melanoma_f1": 0.4067796610169492,
            "mel_to_nv_errors": 25,
            "mel_to_bkl_errors": 23,
            "nv_recall": 0.7791044776119403
        }
    },
    "exp2": {
        "name": "Experiment 2 (Class-Weighted Cross Entropy)",
        "path": os.path.join(AI_DIR, "results", "experiment2", "metrics.json"),
        "fallback": {
            "accuracy": 0.7894736842105263,
            "balanced_accuracy": 0.7510319834759847,
            "macro_precision": 0.6379846463242919,
            "macro_recall": 0.7510319834759847,
            "macro_f1": 0.6818215230851923,
            "weighted_f1": 0.7982858429129571,
            "melanoma_precision": 0.4796747967479675,
            "melanoma_recall": 0.5566037735849056,
            "melanoma_f1": 0.5152838427947598,
            "mel_to_nv_errors": 24,
            "mel_to_bkl_errors": 15,
            "nv_recall": 0.8388059701492537
        }
    },
    "exp3": {
        "name": "Experiment 3 (Focal Loss)",
        "path": os.path.join(AI_DIR, "results", "experiment3", "metrics.json"),
        "fallback": {
            "accuracy": 0.7682186234817814,
            "balanced_accuracy": 0.7567406852431613,
            "macro_precision": 0.6133887165039235,
            "macro_recall": 0.7567406852431613,
            "macro_f1": 0.6704179374026857,
            "weighted_f1": 0.7816075936859664,
            "melanoma_precision": 0.4628099173553719,
            "melanoma_recall": 0.5283018867924528,
            "melanoma_f1": 0.4933920704845815,
            "mel_to_nv_errors": 27,
            "mel_to_bkl_errors": 18,
            "nv_recall": 0.808955223880597
        }
    },
    "exp4": {
        "name": "Experiment 4 (Class-Balanced Sampling)",
        "path": os.path.join(AI_DIR, "results", "experiment4", "metrics.json"),
        "fallback": {
            "accuracy": 0.7024291497975709,
            "balanced_accuracy": 0.6729739503348123,
            "macro_precision": 0.540103730310702,
            "macro_recall": 0.6729739503348123,
            "macro_f1": 0.5599026462719655,
            "weighted_f1": 0.7314545934149027,
            "melanoma_precision": 0.3584905660377358,
            "melanoma_recall": 0.3584905660377358,
            "melanoma_f1": 0.3584905660377358,
            "mel_to_nv_errors": 34,
            "mel_to_bkl_errors": 26,
            "nv_recall": 0.7582089552238806
        }
    },
    "exp5": {
        "name": "Experiment 5 (Controlled Augmentation)",
        "path": os.path.join(AI_DIR, "results", "experiment5", "metrics.json"),
        "fallback": {
            "accuracy": 0.6963562753036437,
            "balanced_accuracy": 0.6499496927713171,
            "macro_precision": 0.46531473022796677,
            "macro_recall": 0.6499496927713171,
            "macro_f1": 0.5223661162672877,
            "weighted_f1": 0.7158011685386405,
            "melanoma_precision": 0.34408602150537637,
            "melanoma_recall": 0.3018867924528302,
            "melanoma_f1": 0.32160804020100503,
            "mel_to_nv_errors": 23,
            "mel_to_bkl_errors": 31,
            "nv_recall": 0.7761194029850746
        }
    }
}


def load_baseline_metrics(exp_key):
    info = BASELINES[exp_key]
    if os.path.exists(info["path"]):
        try:
            with open(info["path"], "r") as f:
                d = json.load(f)
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
                    "mel_to_nv_errors": d.get("melanoma_to_nv_errors", info["fallback"].get("mel_to_nv_errors")),
                    "mel_to_bkl_errors": d.get("melanoma_to_bkl_errors", info["fallback"].get("mel_to_bkl_errors")),
                    "nv_recall": d.get("nv_recall", info["fallback"].get("nv_recall"))
                }
        except Exception as e:
            print(f"Warning reading baseline {exp_key}: {e}")
    return info["fallback"]


# ── Calculate Class Weights from TRAINING SET ONLY ──────────────────────────────
def calculate_training_class_weights(train_df, device):
    """
    Computes balanced class weights strictly from the training dataset.
    Formula: w_c = N_train / (num_classes * N_c)
    Validation and test data are NEVER touched or observed.
    """
    print("\n" + "="*60)
    print("CALCULATING CLASS WEIGHTS STRICTLY FROM TRAINING SET")
    print("="*60)
    train_counts = train_df['dx'].value_counts()
    total_train  = len(train_df)
    num_classes  = len(CLASS_MAPPING)

    weights_dict = {}
    weights_list = []

    print(f"Total training samples: {total_train}")
    print(f"Number of classes: {num_classes}")
    print("Per-class counts and computed weights (TRAINING SET ONLY):")

    for cls, idx in sorted(CLASS_MAPPING.items(), key=lambda x: x[1]):
        count = int(train_counts.get(cls, 0))
        w = total_train / (num_classes * count) if count > 0 else 1.0
        weights_dict[cls] = float(w)
        weights_list.append(w)
        print(f"  Class {idx} [{cls:>5}]: count = {count:>4} ({count/total_train*100:>5.2f}%) -> weight = {w:.4f}")

    weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)
    print("="*60 + "\n")
    return weights_tensor, weights_dict


# ── Model Definition: EfficientNet-B0 ──────────────────────────────────────────
class DermaAI_EfficientNetB0(nn.Module):
    """
    DermaAI architecture with EfficientNet-B0 pretrained backbone.
    - Compound scaling architecture (depth, width, resolution).
    - Features native Squeeze-and-Excitation (SE) channel attention inside all MBConv blocks.
    - Feature extractor: model.features (out_channels=1280).
    - Classification head: AdaptiveAvgPool2d + Dropout(0.2) + Linear(1280, num_classes).
    """
    def __init__(self, num_classes=7, pretrained=True):
        super(DermaAI_EfficientNetB0, self).__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b0(weights=weights)
        
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        in_features = base_model.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ── Test Set Evaluation ─────────────────────────────────────────────────────────
def run_test_evaluation(model, device, test_df, results_dir, checkpoint_path):
    print("\n" + "="*60)
    print("TEST SET EVALUATION (HELD-OUT TEST SPLIT — 988 IMAGES)")
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
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

    report_dict = classification_report(
        all_targets, all_preds, target_names=class_names,
        output_dict=True, zero_division=0)
    report_text = classification_report(
        all_targets, all_preds, target_names=class_names, zero_division=0)

    cm = confusion_matrix(all_targets, all_preds)

    # Detailed melanoma error modes
    # CLASS_MAPPING: 'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
    mel_idx = CLASS_MAPPING['mel']
    nv_idx  = CLASS_MAPPING['nv']
    bkl_idx = CLASS_MAPPING['bkl']

    mel_total = int(sum(all_targets[i] == mel_idx for i in range(len(all_targets))))
    mel_correct = int(cm[mel_idx][mel_idx])
    mel_to_nv_errors = int(cm[mel_idx][nv_idx])
    mel_to_bkl_errors = int(cm[mel_idx][bkl_idx])
    nv_total = int(sum(all_targets[i] == nv_idx for i in range(len(all_targets))))
    nv_correct = int(cm[nv_idx][nv_idx])
    nv_recall = float(nv_correct / nv_total) if nv_total > 0 else 0.0

    print(f"\n{'--------------------------------------------------'}")
    print(f"  Test Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy      : {bacc:.4f}  ({bacc*100:.2f}%)")
    print(f"  Macro Precision        : {prec_macro:.4f}")
    print(f"  Macro Recall           : {rec_macro:.4f}")
    print(f"  Macro F1               : {f1_macro:.4f}")
    print(f"  Weighted F1            : {f1_weighted:.4f}")
    print(f"{'--------------------------------------------------'}")

    mel_rec  = report_dict["mel"]["recall"]
    mel_prec = report_dict["mel"]["precision"]
    mel_f1   = report_dict["mel"]["f1-score"]
    print(f"\n  *** Melanoma (mel) Recall    : {mel_rec:.4f}  ({mel_rec*100:.2f}% — {mel_correct}/{mel_total} images)")
    print(f"  *** Melanoma (mel) Precision : {mel_prec:.4f}")
    print(f"  *** Melanoma (mel) F1        : {mel_f1:.4f}")
    print(f"  *** mel -> nv misclassifications  : {mel_to_nv_errors}")
    print(f"  *** mel -> bkl misclassifications : {mel_to_bkl_errors}")
    print(f"  *** Majority class (nv) Recall    : {nv_recall:.4f}  ({nv_correct}/{nv_total} images)")

    print(f"\n{'--------------------------------------------------'}")
    print("Per-class Metrics:")
    print(f"{'--------------------------------------------------'}")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    # Save text report
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write("EXPERIMENT 6 — ARCHITECTURE IMPROVEMENT (EfficientNet-B0) Test Set Evaluation\n")
        f.write(f"{'='*60}\n")
        f.write(f"Backbone          : EfficientNet-B0 (Pretrained: EfficientNet_B0_Weights.DEFAULT)\n")
        f.write(f"Test Accuracy     : {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Balanced Accuracy : {bacc:.4f} ({bacc*100:.2f}%)\n")
        f.write(f"Macro Precision   : {prec_macro:.4f}\n")
        f.write(f"Macro Recall      : {rec_macro:.4f}\n")
        f.write(f"Macro F1          : {f1_macro:.4f}\n")
        f.write(f"Weighted F1       : {f1_weighted:.4f}\n\n")
        f.write(f"Melanoma Recall   : {mel_rec:.4f} ({mel_rec*100:.2f}% — {mel_correct}/{mel_total} images)\n")
        f.write(f"Melanoma Precision: {mel_prec:.4f}\n")
        f.write(f"Melanoma F1       : {mel_f1:.4f}\n")
        f.write(f"mel -> nv errors  : {mel_to_nv_errors} images\n")
        f.write(f"mel -> bkl errors : {mel_to_bkl_errors} images\n")
        f.write(f"nv (majority) rec : {nv_recall:.4f} ({nv_correct}/{nv_total} images)\n\n")
        f.write(f"Classification Report:\n{report_text}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    # Save JSON metrics
    metrics = {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "melanoma_recall": float(mel_rec),
        "melanoma_precision": float(mel_prec),
        "melanoma_f1": float(mel_f1),
        "melanoma_correct_images": mel_correct,
        "melanoma_total_images": mel_total,
        "melanoma_to_nv_errors": mel_to_nv_errors,
        "melanoma_to_bkl_errors": mel_to_bkl_errors,
        "nv_recall": float(nv_recall),
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
    plt.title('Confusion Matrix — Test Set (Experiment 6: EfficientNet-B0)')
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved evaluation artifacts -> {results_dir}")
    return metrics


# ── Comparison Helper ───────────────────────────────────────────────────────────
def compute_comparisons(exp6_metrics, results_dir):
    comparisons = {}
    for exp_key in ["exp1", "exp2", "exp3", "exp4", "exp5"]:
        base = load_baseline_metrics(exp_key)
        diff = {
            "accuracy_change": float(exp6_metrics["accuracy"] - base["accuracy"]) if base.get("accuracy") is not None else None,
            "balanced_accuracy_change": float(exp6_metrics["balanced_accuracy"] - base["balanced_accuracy"]) if base.get("balanced_accuracy") is not None else None,
            "macro_f1_change": float(exp6_metrics["macro_f1"] - base["macro_f1"]) if base.get("macro_f1") is not None else None,
            "macro_precision_change": float(exp6_metrics["macro_precision"] - base["macro_precision"]) if base.get("macro_precision") is not None else None,
            "macro_recall_change": float(exp6_metrics["macro_recall"] - base["macro_recall"]) if base.get("macro_recall") is not None else None,
            "weighted_f1_change": float(exp6_metrics["weighted_f1"] - base["weighted_f1"]) if base.get("weighted_f1") is not None else None,
            "melanoma_recall_change": float(exp6_metrics["melanoma_recall"] - base["melanoma_recall"]) if base.get("melanoma_recall") is not None else None,
            "melanoma_precision_change": float(exp6_metrics["melanoma_precision"] - base["melanoma_precision"]) if base.get("melanoma_precision") is not None else None,
            "melanoma_f1_change": float(exp6_metrics["melanoma_f1"] - base["melanoma_f1"]) if base.get("melanoma_f1") is not None else None,
            "mel_to_nv_error_change": int(exp6_metrics["melanoma_to_nv_errors"] - base["mel_to_nv_errors"]) if base.get("mel_to_nv_errors") is not None else None,
            "mel_to_bkl_error_change": int(exp6_metrics["melanoma_to_bkl_errors"] - base["mel_to_bkl_errors"]) if base.get("mel_to_bkl_errors") is not None else None,
            "nv_recall_change": float(exp6_metrics["nv_recall"] - base["nv_recall"]) if base.get("nv_recall") is not None else None,
        }
        
        comp_data = {
            "baseline": base,
            "experiment6_efficientnet_b0": {
                "accuracy": exp6_metrics["accuracy"],
                "balanced_accuracy": exp6_metrics["balanced_accuracy"],
                "macro_precision": exp6_metrics["macro_precision"],
                "macro_recall": exp6_metrics["macro_recall"],
                "macro_f1": exp6_metrics["macro_f1"],
                "weighted_f1": exp6_metrics["weighted_f1"],
                "melanoma_recall": exp6_metrics["melanoma_recall"],
                "melanoma_precision": exp6_metrics["melanoma_precision"],
                "melanoma_f1": exp6_metrics["melanoma_f1"],
                "mel_to_nv_errors": exp6_metrics["melanoma_to_nv_errors"],
                "mel_to_bkl_errors": exp6_metrics["melanoma_to_bkl_errors"],
                "nv_recall": exp6_metrics["nv_recall"],
            },
            f"delta_exp6_minus_{exp_key}": diff
        }

        # Filename matching required names: comparison_with_experiment1.json etc.
        idx_str = exp_key.replace("exp", "")
        file_path = os.path.join(results_dir, f"comparison_with_experiment{idx_str}.json")
        with open(file_path, "w") as f:
            json.dump(comp_data, f, indent=4)
        comparisons[exp_key] = comp_data

    # Print comparative delta against primary baseline (Experiment 2)
    exp2_diff = comparisons["exp2"]["delta_exp6_minus_exp2"]
    exp2_base = comparisons["exp2"]["baseline"]
    print("\n" + "="*60)
    print("COMPARISON: EXPERIMENT 6 vs EXPERIMENT 2 (PRIMARY BASELINE)")
    print("="*60)
    print(f"  Test Accuracy      : {exp2_base['accuracy']*100:.2f}% -> {exp6_metrics['accuracy']*100:.2f}%  (delta: {exp2_diff['accuracy_change']*100:+.2f}%)")
    print(f"  Balanced Accuracy  : {exp2_base['balanced_accuracy']*100:.2f}% -> {exp6_metrics['balanced_accuracy']*100:.2f}%  (delta: {exp2_diff['balanced_accuracy_change']*100:+.2f}%)")
    print(f"  Macro F1           : {exp2_base['macro_f1']:.4f} -> {exp6_metrics['macro_f1']:.4f}  (delta: {exp2_diff['macro_f1_change']:+.4f})")
    print(f"  Weighted F1        : {exp2_base['weighted_f1']:.4f} -> {exp6_metrics['weighted_f1']:.4f}  (delta: {exp2_diff['weighted_f1_change']:+.4f})")
    print(f"  Melanoma Recall    : {exp2_base['melanoma_recall']*100:.2f}% -> {exp6_metrics['melanoma_recall']*100:.2f}%  (delta: {exp2_diff['melanoma_recall_change']*100:+.2f}%)")
    print(f"  Melanoma F1        : {exp2_base['melanoma_f1']:.4f} -> {exp6_metrics['melanoma_f1']:.4f}  (delta: {exp2_diff['melanoma_f1_change']:+.4f})")
    print(f"  Mel -> nv errors   : {exp2_base['mel_to_nv_errors']} -> {exp6_metrics['melanoma_to_nv_errors']}  (delta: {exp2_diff['mel_to_nv_error_change']:+d})")
    print(f"  Mel -> bkl errors  : {exp2_base['mel_to_bkl_errors']} -> {exp6_metrics['melanoma_to_bkl_errors']}  (delta: {exp2_diff['mel_to_bkl_error_change']:+d})")
    print(f"  Majority (nv) Rec  : {exp2_base['nv_recall']*100:.2f}% -> {exp6_metrics['nv_recall']*100:.2f}%  (delta: {exp2_diff['nv_recall_change']*100:+.2f}%)")
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

    # 2. Compute class weights STRICTLY from training set only
    weights_tensor, class_weights_dict = calculate_training_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Save training class weights
    with open(os.path.join(RESULTS_DIR, "training_class_weights.json"), "w") as f:
        json.dump(class_weights_dict, f, indent=4)

    # 3. Datasets and Loaders (keeping identical to Exp 2: unweighted random shuffle, standard transforms)
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
        print("[SMOKE TEST] PASSED — proceeding to model creation.\n")
        del smoke_imgs, smoke_labels
    except Exception as smoke_err:
        import traceback
        print("\n[SMOKE TEST] FAILED — aborting. Full traceback:")
        traceback.print_exc()
        raise SystemExit(1) from smoke_err

    # 4. Model Architecture: EfficientNet-B0
    model = DermaAI_EfficientNetB0(num_classes=7, pretrained=True)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())

    # Freeze backbone initially (Epochs 1-15)
    for param in model.features.parameters():
        param.requires_grad = False
    trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Unfrozen parameter count for fine-tuning phase
    for param in model.features.parameters():
        param.requires_grad = True
    trainable_full = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Re-freeze for Epochs 1-15
    for param in model.features.parameters():
        param.requires_grad = False

    print("="*60)
    print("ARCHITECTURE & PARAMETER SPECIFICATION")
    print("="*60)
    print(f"  Model Name                      : EfficientNet-B0")
    print(f"  Pretrained Weights              : EfficientNet_B0_Weights.DEFAULT (ImageNet-1K V1)")
    print(f"  Total Parameters                : {total_params:,}")
    print(f"  Trainable Params (Frozen Phase) : {trainable_frozen:,}")
    print(f"  Trainable Params (Fine-Tuning)  : {trainable_full:,}")
    print(f"  Attention Module                : Native Squeeze-and-Excitation (SE) blocks in MBConv")
    print(f"  Classifier Modification         : Dropout(p=0.2) + Linear(1280, 7)")
    print(f"  Input Resolution                : 224 x 224")
    print(f"  Weights Cached Locally          : Yes (torch hub checkpoints)")
    print("="*60 + "\n")

    # Save architecture configuration and parameter summary artifacts
    arch_config = {
        "model_name": "EfficientNet-B0",
        "backbone_class": "torchvision.models.efficientnet_b0",
        "pretrained_weights": "EfficientNet_B0_Weights.DEFAULT",
        "weights_cached_locally": True,
        "input_size": [3, 224, 224],
        "attention_module": "Native Squeeze-and-Excitation (SE) channel attention in all MBConv blocks",
        "classifier_head_modification": "Replaced original 1000-class linear projection with nn.Sequential(nn.Dropout(p=0.2), nn.Linear(1280, 7))",
        "num_classes": 7,
        "frozen_phase_epochs": "1-15",
        "fine_tuning_phase_epochs": "16-30"
    }
    with open(os.path.join(RESULTS_DIR, "architecture_configuration.json"), "w") as f:
        json.dump(arch_config, f, indent=4)

    param_summary = {
        "model_name": "EfficientNet-B0",
        "total_parameters": total_params,
        "trainable_parameters_frozen_phase": trainable_frozen,
        "non_trainable_parameters_frozen_phase": total_params - trainable_frozen,
        "trainable_parameters_fine_tuning": trainable_full,
        "non_trainable_parameters_fine_tuning": 0
    }
    with open(os.path.join(RESULTS_DIR, "parameter_summary.json"), "w") as f:
        json.dump(param_summary, f, indent=4)

    # 5. Optimizer & Scheduler (Identical to Experiment 2)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3)

    # 6. Training Loop
    best_val_loss     = float('inf')
    best_val_epoch    = -1
    epochs_no_improve = 0
    checkpoint_path   = os.path.join(MODELS_DIR, CHECKPOINT_NAME)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    print(f"Starting Experiment 6 training for up to {EPOCHS} epochs with class-weighted CrossEntropy...")
    print("="*60)

    for epoch in range(EPOCHS):
        # Unfreeze backbone at midpoint (Epoch 16+)
        if epoch == EPOCHS // 2:
            print("\nUnfreezing backbone for fine-tuning phase (epochs 16-30)...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(
                model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=3)
            print(f"  Trainable params now: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

        # Train
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]")
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
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Val]  ")
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

        history['train_loss'].append(float(epoch_train_loss))
        history['val_loss'].append(float(epoch_val_loss))
        history['val_acc'].append(float(epoch_val_acc))
        history['lr'].append(float(current_lr))

        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.4f} | "
              f"LR: {current_lr:.2e}")

        scheduler.step(epoch_val_loss)

        # Checkpoint selection strictly by validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss  = epoch_val_loss
            best_val_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  OK Saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping after {epoch+1} epochs.")
                break

    training_time_s = time.time() - training_start
    training_time_h = training_time_s / 3600

    print("="*60)
    print(f"Training complete — best epoch: {best_val_epoch} | "
          f"best val loss: {best_val_loss:.4f} | "
          f"time: {training_time_s:.0f}s ({training_time_h:.2f}h)")

    # 7. Save history & curves
    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    epochs_ran = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs_ran, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs_ran, history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs_ran, history['val_acc'], color='green', label='Val Accuracy')
    axes[1].set_title('Val Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs_ran, history['lr'], color='orange', label='LR')
    axes[2].set_title('Learning Rate'); axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log'); axes[2].legend(); axes[2].grid(True)

    plt.suptitle('Experiment 6 — Architecture Improvement (EfficientNet-B0) Training Curves', fontsize=14)
    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"  Training curves -> {curves_path}")

    # 8. Single Test Evaluation
    metrics = run_test_evaluation(model, device, test_df, RESULTS_DIR, checkpoint_path)

    # 9. Comparisons against all prior experiments
    comparisons = compute_comparisons(metrics, RESULTS_DIR)

    # 10. Experiment Summary
    summary = {
        "experiment": EXPERIMENT_NAME,
        "config": {
            "epochs_configured": EPOCHS,
            "epochs_ran": len(epochs_ran),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss": "class_weighted_cross_entropy",
            "class_weights_source": "training_set_only",
            "class_weights": class_weights_dict,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
            "backbone": "EfficientNet-B0",
            "pretrained_weights": "EfficientNet_B0_Weights.DEFAULT",
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
            "name": "EfficientNet-B0",
            "total_params": total_params,
            "trainable_params_frozen_phase": trainable_frozen,
            "trainable_params_full": trainable_full,
        },
        "training": {
            "best_epoch": best_val_epoch,
            "best_val_loss": float(best_val_loss),
            "training_time_seconds": float(training_time_s),
            "training_time_hours": float(training_time_h),
        },
        "checkpoint": checkpoint_path,
        "results_dir": RESULTS_DIR,
        "test_metrics": metrics,
        "primary_comparison_exp2": comparisons["exp2"]
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'='*60}")
    print("EXPERIMENT 6 COMPLETE")
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
    print("\nExperiment 6 finished. STOPPED — awaiting next instruction.\n")


if __name__ == "__main__":
    train()
