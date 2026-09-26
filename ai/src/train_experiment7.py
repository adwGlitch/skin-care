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
RESULTS_DIR   = os.path.join(AI_DIR, "results", "experiment7")

EXPERIMENT_NAME = "experiment7_gradual_fine_tuning"
CHECKPOINT_NAME = "experiment7_best_model.pt"

EPOCHS        = 30
BATCH_SIZE    = 32
PATIENCE      = 5
WEIGHT_DECAY  = 1e-4

# Fine-tuning schedule stage definitions
STAGE_CONFIGS = {
    1: {
        "name": "Stage 1 — Classifier Warm-Up",
        "epochs": (1, 5),
        "backbone_lr": None,
        "head_lr": 1e-4,
        "unfrozen_layers": "Classifier & CBAM head only; entire backbone features frozen",
        "description": "Allow newly initialized classification layers to adapt before modifying pretrained feature representations."
    },
    2: {
        "name": "Stage 2 — Partial Backbone Fine-Tuning",
        "epochs": (6, 15),
        "backbone_lr": 1e-5,
        "head_lr": 5e-5,
        "unfrozen_layers": "Final backbone stage (MobileNetV3 features[13:17]) + Classifier & CBAM head",
        "description": "Allow high-level pretrained features to adapt gradually while protecting lower-level representations."
    },
    3: {
        "name": "Stage 3 — Full Fine-Tuning",
        "epochs": (16, 30),
        "backbone_lr": 5e-6,
        "head_lr": 1e-5,
        "unfrozen_layers": "Entire network (all backbone features + Classifier & CBAM head)",
        "description": "Conservative full-network adaptation to prevent disruption of useful pretrained features."
    }
}

# ── Baseline Loaders for Experiments 1 through 6 ───────────────────────────────
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
        "name": "Experiment 2 (Class-Weighted Primary Baseline)",
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
    },
    "exp6": {
        "name": "Experiment 6 (EfficientNet-B0)",
        "path": os.path.join(AI_DIR, "results", "experiment6", "metrics.json"),
        "fallback": {
            "accuracy": 0.7176113360323887,
            "balanced_accuracy": 0.726623633411615,
            "macro_precision": 0.5476780940173092,
            "macro_recall": 0.726623633411615,
            "macro_f1": 0.6076128543409124,
            "weighted_f1": 0.7389136580000619,
            "melanoma_precision": 0.4013605442176871,
            "melanoma_recall": 0.5566037735849056,
            "melanoma_f1": 0.466403162055336,
            "mel_to_nv_errors": 14,
            "mel_to_bkl_errors": 20,
            "nv_recall": 0.7522388059701492
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


# ── Helper for Head vs Backbone Parameters ─────────────────────────────────────
def get_head_params(model):
    """Returns all parameters of the CBAM attention and classification head."""
    params = []
    if model.use_attention:
        params.extend(list(model.attention.parameters()))
    params.extend(list(model.classifier.parameters()))
    return params


# ── Stage-Wise Setup Function ──────────────────────────────────────────────────
def setup_stage(stage_num, model):
    """
    Configures requires_grad flags and builds optimizer & scheduler for the specified stage.
    Stage 1 (Epochs 1-5): Complete backbone frozen. Head LR = 1e-4.
    Stage 2 (Epochs 6-15): Final stage (features[13:17]) unfrozen. Backbone LR = 1e-5, Head LR = 5e-5.
    Stage 3 (Epochs 16-30): Entire network unfrozen. Backbone LR = 5e-6, Head LR = 1e-5.
    """
    head_params = get_head_params(model)
    # Ensure head parameters always require grad
    for p in head_params:
        p.requires_grad = True

    if stage_num == 1:
        # Freeze entire backbone
        for p in model.features.parameters():
            p.requires_grad = False

        param_groups = [
            {'params': head_params, 'lr': 1e-4, 'weight_decay': WEIGHT_DECAY, 'name': 'head'}
        ]

    elif stage_num == 2:
        # Freeze features[0:13], unfreeze features[13:17]
        for i in range(13):
            for p in model.features[i].parameters():
                p.requires_grad = False
        stage2_backbone_params = []
        for i in range(13, 17):
            for p in model.features[i].parameters():
                p.requires_grad = True
                stage2_backbone_params.append(p)

        param_groups = [
            {'params': stage2_backbone_params, 'lr': 1e-5, 'weight_decay': WEIGHT_DECAY, 'name': 'final_backbone_stage'},
            {'params': head_params, 'lr': 5e-5, 'weight_decay': WEIGHT_DECAY, 'name': 'head'}
        ]

    elif stage_num == 3:
        # Unfreeze all backbone features
        for p in model.features.parameters():
            p.requires_grad = True

        param_groups = [
            {'params': list(model.features.parameters()), 'lr': 5e-6, 'weight_decay': WEIGHT_DECAY, 'name': 'full_backbone'},
            {'params': head_params, 'lr': 1e-5, 'weight_decay': WEIGHT_DECAY, 'name': 'head'}
        ]
    else:
        raise ValueError(f"Invalid stage_num: {stage_num}")

    optimizer = optim.AdamW(param_groups)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=PATIENCE)

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    return optimizer, scheduler, trainable_count, frozen_count


# ── Test Set Evaluation (Held-Out 988 Images) ──────────────────────────────────
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
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0)

    report_dict = classification_report(
        all_targets, all_preds, target_names=class_names,
        output_dict=True, zero_division=0)
    report_text = classification_report(
        all_targets, all_preds, target_names=class_names, zero_division=0)

    cm = confusion_matrix(all_targets, all_preds)

    # Detailed melanoma and nevus analysis
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
    nv_prec = float(report_dict['nv']['precision'])
    nv_f1 = float(report_dict['nv']['f1-score'])

    mel_rec  = float(report_dict["mel"]["recall"])
    mel_prec = float(report_dict["mel"]["precision"])
    mel_f1   = float(report_dict["mel"]["f1-score"])

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
    print(f"  Melanoma (mel) Recall  : {mel_rec:.4f}  ({mel_rec*100:.2f}% — {mel_correct}/{mel_total} images)")
    print(f"  Melanoma (mel) Prec    : {mel_prec:.4f}")
    print(f"  Melanoma (mel) F1      : {mel_f1:.4f}")
    print(f"  mel -> nv errors       : {mel_to_nv_errors} images")
    print(f"  mel -> bkl errors      : {mel_to_bkl_errors} images")
    print(f"  Majority (nv) Recall   : {nv_recall:.4f}  ({nv_correct}/{nv_total} images)")
    print(f"  Majority (nv) Prec     : {nv_prec:.4f}")
    print(f"  Majority (nv) F1       : {nv_f1:.4f}")
    print(f"{'--------------------------------------------------'}")

    print("\nPer-class Metrics:")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    # Save text report
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write("EXPERIMENT 7 — GRADUAL FINE-TUNING STRATEGY ON HAM10000\n")
        f.write("Test Set Evaluation (Held-Out Test Split — 988 Images)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Backbone          : MobileNetV3-Large + CBAM Attention\n")
        f.write(f"Strategy          : 3-Stage Gradual Fine-Tuning\n")
        f.write(f"Test Accuracy     : {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Balanced Accuracy : {bacc:.4f} ({bacc*100:.2f}%)\n")
        f.write(f"Macro Precision   : {prec_macro:.4f}\n")
        f.write(f"Macro Recall      : {rec_macro:.4f}\n")
        f.write(f"Macro F1          : {f1_macro:.4f}\n")
        f.write(f"Weighted Precision: {prec_weighted:.4f}\n")
        f.write(f"Weighted Recall   : {rec_weighted:.4f}\n")
        f.write(f"Weighted F1       : {f1_weighted:.4f}\n\n")
        f.write(f"Melanoma Recall   : {mel_rec:.4f} ({mel_rec*100:.2f}% — {mel_correct}/{mel_total} images)\n")
        f.write(f"Melanoma Precision: {mel_prec:.4f}\n")
        f.write(f"Melanoma F1       : {mel_f1:.4f}\n")
        f.write(f"mel -> nv errors  : {mel_to_nv_errors} images\n")
        f.write(f"mel -> bkl errors : {mel_to_bkl_errors} images\n\n")
        f.write(f"nv (majority) rec : {nv_recall:.4f} ({nv_correct}/{nv_total} images)\n")
        f.write(f"nv (majority) prec: {nv_prec:.4f}\n")
        f.write(f"nv (majority) f1  : {nv_f1:.4f}\n\n")
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
    plt.title('Confusion Matrix — Test Set (Experiment 7: Gradual Fine-Tuning)')
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved evaluation artifacts -> {results_dir}")
    return metrics


# ── Comparison Helper Across All Prior Experiments ─────────────────────────────
def compute_comparisons(exp7_metrics, results_dir):
    comparisons = {}
    for exp_key in ["exp1", "exp2", "exp3", "exp4", "exp5", "exp6"]:
        base = load_baseline_metrics(exp_key)
        diff = {
            "accuracy_change": float(exp7_metrics["accuracy"] - base["accuracy"]) if base.get("accuracy") is not None else None,
            "balanced_accuracy_change": float(exp7_metrics["balanced_accuracy"] - base["balanced_accuracy"]) if base.get("balanced_accuracy") is not None else None,
            "macro_f1_change": float(exp7_metrics["macro_f1"] - base["macro_f1"]) if base.get("macro_f1") is not None else None,
            "macro_precision_change": float(exp7_metrics["macro_precision"] - base["macro_precision"]) if base.get("macro_precision") is not None else None,
            "macro_recall_change": float(exp7_metrics["macro_recall"] - base["macro_recall"]) if base.get("macro_recall") is not None else None,
            "weighted_f1_change": float(exp7_metrics["weighted_f1"] - base["weighted_f1"]) if base.get("weighted_f1") is not None else None,
            "melanoma_recall_change": float(exp7_metrics["melanoma_recall"] - base["melanoma_recall"]) if base.get("melanoma_recall") is not None else None,
            "melanoma_precision_change": float(exp7_metrics["melanoma_precision"] - base["melanoma_precision"]) if base.get("melanoma_precision") is not None else None,
            "melanoma_f1_change": float(exp7_metrics["melanoma_f1"] - base["melanoma_f1"]) if base.get("melanoma_f1") is not None else None,
            "mel_to_nv_error_change": int(exp7_metrics["melanoma_to_nv_errors"] - base["mel_to_nv_errors"]) if base.get("mel_to_nv_errors") is not None else None,
            "mel_to_bkl_error_change": int(exp7_metrics["melanoma_to_bkl_errors"] - base["mel_to_bkl_errors"]) if base.get("mel_to_bkl_errors") is not None else None,
            "nv_recall_change": float(exp7_metrics["nv_recall"] - base["nv_recall"]) if base.get("nv_recall") is not None else None,
        }
        
        comp_data = {
            "baseline": base,
            "experiment7_gradual_fine_tuning": {
                "accuracy": exp7_metrics["accuracy"],
                "balanced_accuracy": exp7_metrics["balanced_accuracy"],
                "macro_precision": exp7_metrics["macro_precision"],
                "macro_recall": exp7_metrics["macro_recall"],
                "macro_f1": exp7_metrics["macro_f1"],
                "weighted_f1": exp7_metrics["weighted_f1"],
                "melanoma_recall": exp7_metrics["melanoma_recall"],
                "melanoma_precision": exp7_metrics["melanoma_precision"],
                "melanoma_f1": exp7_metrics["melanoma_f1"],
                "mel_to_nv_errors": exp7_metrics["melanoma_to_nv_errors"],
                "mel_to_bkl_errors": exp7_metrics["melanoma_to_bkl_errors"],
                "nv_recall": exp7_metrics["nv_recall"],
            },
            f"delta_exp7_minus_{exp_key}": diff
        }

        idx_str = exp_key.replace("exp", "")
        file_path = os.path.join(results_dir, f"comparison_with_experiment{idx_str}.json")
        with open(file_path, "w") as f:
            json.dump(comp_data, f, indent=4)
        comparisons[exp_key] = comp_data

    # Print comparison against primary baseline (Experiment 2)
    exp2_diff = comparisons["exp2"]["delta_exp7_minus_exp2"]
    exp2_base = comparisons["exp2"]["baseline"]
    print("\n" + "="*60)
    print("COMPARISON: EXPERIMENT 7 vs EXPERIMENT 2 (PRIMARY BASELINE)")
    print("="*60)
    print(f"  Test Accuracy      : {exp2_base['accuracy']*100:.2f}% -> {exp7_metrics['accuracy']*100:.2f}%  (delta: {exp2_diff['accuracy_change']*100:+.2f}%)")
    print(f"  Balanced Accuracy  : {exp2_base['balanced_accuracy']*100:.2f}% -> {exp7_metrics['balanced_accuracy']*100:.2f}%  (delta: {exp2_diff['balanced_accuracy_change']*100:+.2f}%)")
    print(f"  Macro F1           : {exp2_base['macro_f1']:.4f} -> {exp7_metrics['macro_f1']:.4f}  (delta: {exp2_diff['macro_f1_change']:+.4f})")
    print(f"  Weighted F1        : {exp2_base['weighted_f1']:.4f} -> {exp7_metrics['weighted_f1']:.4f}  (delta: {exp2_diff['weighted_f1_change']:+.4f})")
    print(f"  Melanoma Recall    : {exp2_base['melanoma_recall']*100:.2f}% -> {exp7_metrics['melanoma_recall']*100:.2f}%  (delta: {exp2_diff['melanoma_recall_change']*100:+.2f}%)")
    print(f"  Melanoma F1        : {exp2_base['melanoma_f1']:.4f} -> {exp7_metrics['melanoma_f1']:.4f}  (delta: {exp2_diff['melanoma_f1_change']:+.4f})")
    print(f"  Mel -> nv errors   : {exp2_base['mel_to_nv_errors']} -> {exp7_metrics['melanoma_to_nv_errors']}  (delta: {exp2_diff['mel_to_nv_error_change']:+d})")
    print(f"  Mel -> bkl errors  : {exp2_base['mel_to_bkl_errors']} -> {exp7_metrics['melanoma_to_bkl_errors']}  (delta: {exp2_diff['mel_to_bkl_error_change']:+d})")
    print(f"  Majority (nv) Rec  : {exp2_base['nv_recall']*100:.2f}% -> {exp7_metrics['nv_recall']*100:.2f}%  (delta: {exp2_diff['nv_recall_change']*100:+.2f}%)")
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
        print("[SMOKE TEST] PASSED — proceeding to model creation.\n")
        del smoke_imgs, smoke_labels
    except Exception as smoke_err:
        import traceback
        print("\n[SMOKE TEST] FAILED — aborting. Full traceback:")
        traceback.print_exc()
        raise SystemExit(1) from smoke_err

    # 4. Model Architecture: MobileNetV3-Large + CBAM Attention
    model = DermaAI_MobileNetV3(num_classes=7, use_attention=True, pretrained=True)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())

    # Pre-calculate parameter counts for each stage
    _, _, s1_trainable, s1_frozen = setup_stage(1, model)
    _, _, s2_trainable, s2_frozen = setup_stage(2, model)
    _, _, s3_trainable, s3_frozen = setup_stage(3, model)

    # Re-initialize to Stage 1
    optimizer, scheduler, current_trainable, current_frozen = setup_stage(1, model)

    # Print Full Pre-Training Verification (Requirement 8)
    print("="*60)
    print("MODEL VERIFICATION BEFORE TRAINING (REQUIREMENT 8)")
    print("="*60)
    print(f"1.  Model Architecture Name     : DermaAI_MobileNetV3")
    print(f"2.  Backbone Name               : MobileNetV3-Large")
    print(f"3.  CBAM Attention Module       : Present (Channel + Spatial Attention)")
    print(f"4.  Total Parameter Count       : {total_params:,}")
    print(f"5.  Stage 1 Trainable Params    : {s1_trainable:,}  (Frozen: {s1_frozen:,})")
    print(f"6.  Stage 2 Trainable Params    : {s2_trainable:,}  (Frozen: {s2_frozen:,})")
    print(f"7.  Stage 3 Trainable Params    : {s3_trainable:,}  (Frozen: {s3_frozen:,})")
    print(f"8.  Optimizer                   : AdamW (weight_decay={WEIGHT_DECAY})")
    print(f"9.  Stage 1 Learning Rates      : head={STAGE_CONFIGS[1]['head_lr']}")
    print(f"    Stage 2 Learning Rates      : final_backbone={STAGE_CONFIGS[2]['backbone_lr']}, head={STAGE_CONFIGS[2]['head_lr']}")
    print(f"    Stage 3 Learning Rates      : full_backbone={STAGE_CONFIGS[3]['backbone_lr']}, head={STAGE_CONFIGS[3]['head_lr']}")
    print(f"10. Weight Decay                : {WEIGHT_DECAY}")
    print(f"11. Number of Classes           : 7 (akiec, bcc, bkl, df, mel, nv, vasc)")
    print(f"12. Input Resolution            : 224 x 224 x 3")
    print(f"13. Dataset Split Sizes         : Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"14. Class Weights Source        : TRAINING SET ONLY")
    print(f"15. Random Seed                 : {SEED}")
    print(f"16. Batch Size                  : {BATCH_SIZE}")
    print(f"Verification Check: Stage 1 features[0].requires_grad = {next(model.features.parameters()).requires_grad}")
    print(f"Verification Check: Stage 1 head.requires_grad     = {next(model.classifier.parameters()).requires_grad}")
    print("="*60 + "\n")

    # Save architecture configuration artifact
    arch_config = {
        "model_name": "DermaAI_MobileNetV3",
        "backbone": "MobileNetV3-Large",
        "pretrained_weights": "MobileNet_V3_Large_Weights.DEFAULT",
        "attention_module": "CBAM (Channel & Spatial Attention)",
        "input_size": [3, 224, 224],
        "num_classes": 7,
        "total_parameters": total_params,
        "stages": STAGE_CONFIGS
    }
    with open(os.path.join(RESULTS_DIR, "architecture_configuration.json"), "w") as f:
        json.dump(arch_config, f, indent=4)

    # Save parameter summary artifact
    param_summary = {
        "total_parameters": total_params,
        "stage1_trainable_parameters": s1_trainable,
        "stage1_frozen_parameters": s1_frozen,
        "stage2_trainable_parameters": s2_trainable,
        "stage2_frozen_parameters": s2_frozen,
        "stage3_trainable_parameters": s3_trainable,
        "stage3_frozen_parameters": s3_frozen
    }
    with open(os.path.join(RESULTS_DIR, "parameter_summary.json"), "w") as f:
        json.dump(param_summary, f, indent=4)

    # Save fine-tuning schedule artifact
    with open(os.path.join(RESULTS_DIR, "fine_tuning_schedule.json"), "w") as f:
        json.dump(STAGE_CONFIGS, f, indent=4)

    # 5. Training Loop
    best_val_loss     = float('inf')
    best_val_epoch    = -1
    best_val_acc      = 0.0
    epochs_no_improve = 0
    checkpoint_path   = os.path.join(MODELS_DIR, CHECKPOINT_NAME)

    history = {
        'epoch': [],
        'stage': [],
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'lr_groups': [],
        'trainable_params': [],
        'epoch_duration_seconds': []
    }

    current_stage = 1
    print(f"Starting Experiment 7 training for up to {EPOCHS} epochs with 3-stage gradual fine-tuning...")
    print(f"==> ENTERING STAGE 1: Epochs 1-5 (Classifier Warm-Up, head_lr=1e-4)")
    print("="*60)

    for epoch in range(EPOCHS):
        epoch_num = epoch + 1
        epoch_start_time = time.time()

        # Handle Stage Transitions
        if epoch_num == 6:
            current_stage = 2
            print("\n" + "="*60)
            print("==> TRANSITION TO STAGE 2: Epochs 6-15 (Partial Backbone Fine-Tuning)")
            print("    Unfreezing final backbone stage (features[13:17]) with backbone_lr=1e-5, head_lr=5e-5")
            print("="*60)
            optimizer, scheduler, current_trainable, current_frozen = setup_stage(2, model)
            epochs_no_improve = 0  # Reset patience for newly transitioned stage
            print(f"    Trainable params now: {current_trainable:,} | Frozen: {current_frozen:,}\n")

        elif epoch_num == 16:
            current_stage = 3
            print("\n" + "="*60)
            print("==> TRANSITION TO STAGE 3: Epochs 16-30 (Full Fine-Tuning)")
            print("    Unfreezing entire network with full_backbone_lr=5e-6, head_lr=1e-5")
            print("="*60)
            optimizer, scheduler, current_trainable, current_frozen = setup_stage(3, model)
            epochs_no_improve = 0  # Reset patience for newly transitioned stage
            print(f"    Trainable params now: {current_trainable:,} | Frozen: {current_frozen:,}\n")

        # Train
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_num:02d}/{EPOCHS} [Train Stage {current_stage}]")
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
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch_num:02d}/{EPOCHS} [Val]            ")
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
        epoch_elapsed  = time.time() - epoch_start_time

        # Extract learning rates for all parameter groups
        current_lrs = {g.get('name', f"group_{idx}"): g['lr'] for idx, g in enumerate(optimizer.param_groups)}

        history['epoch'].append(epoch_num)
        history['stage'].append(current_stage)
        history['train_loss'].append(float(epoch_train_loss))
        history['val_loss'].append(float(epoch_val_loss))
        history['val_acc'].append(float(epoch_val_acc))
        history['lr_groups'].append(current_lrs)
        history['trainable_params'].append(current_trainable)
        history['epoch_duration_seconds'].append(float(epoch_elapsed))

        lr_str = " | ".join([f"{k}: {v:.1e}" for k, v in current_lrs.items()])
        print(f"Epoch {epoch_num:02d} (Stage {current_stage}) | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.4f} | "
              f"{lr_str} | Time: {epoch_elapsed:.1f}s")

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
                print(f"\nEarly stopping triggered after {epochs_no_improve} epochs without val loss improvement in Stage {current_stage} (Epoch {epoch_num}).")
                break

    training_time_s = time.time() - training_start
    training_time_h = training_time_s / 3600

    print("="*60)
    print(f"Training complete — best epoch: {best_val_epoch} | "
          f"best val loss: {best_val_loss:.4f} | "
          f"best val acc: {best_val_acc:.4f} | "
          f"total time: {training_time_s:.0f}s ({training_time_h:.2f}h)")

    # 6. Save history & generate training curves
    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    epochs_ran = history['epoch']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # Loss curves
    axes[0].plot(epochs_ran, history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(epochs_ran, history['val_loss'],   label='Val Loss', color='red')
    axes[0].axvline(x=5.5, color='gray', linestyle='--', label='Stage 1 -> 2')
    axes[0].axvline(x=15.5, color='purple', linestyle='--', label='Stage 2 -> 3')
    axes[0].set_title('Training & Validation Loss'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True)

    # Accuracy curves
    axes[1].plot(epochs_ran, history['val_acc'], color='green', label='Val Accuracy')
    axes[1].axvline(x=5.5, color='gray', linestyle='--')
    axes[1].axvline(x=15.5, color='purple', linestyle='--')
    axes[1].set_title('Validation Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True)

    # Learning Rate curves (Head vs Backbone)
    head_lrs = [g.get('head', list(g.values())[-1]) for g in history['lr_groups']]
    backbone_lrs = [g.get('final_backbone_stage', g.get('full_backbone', 0.0)) for g in history['lr_groups']]
    axes[2].plot(epochs_ran, head_lrs, color='orange', label='Head LR')
    axes[2].plot(epochs_ran, backbone_lrs, color='brown', label='Backbone LR')
    axes[2].axvline(x=5.5, color='gray', linestyle='--')
    axes[2].axvline(x=15.5, color='purple', linestyle='--')
    axes[2].set_title('Differential Learning Rates'); axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log'); axes[2].legend(); axes[2].grid(True)

    plt.suptitle('Experiment 7 — 3-Stage Gradual Fine-Tuning Training Curves', fontsize=14)
    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"  Training curves -> {curves_path}")

    # 7. Single Test Evaluation
    metrics = run_test_evaluation(model, device, test_df, RESULTS_DIR, checkpoint_path)

    # 8. Comparisons against all prior experiments (1 through 6)
    comparisons = compute_comparisons(metrics, RESULTS_DIR)

    # 9. Experiment Summary & Reproducibility Metadata
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
        "class_weights": class_weights_dict,
        "timestamp_start": training_start,
        "total_duration_seconds": training_time_s,
        "total_duration_hours": training_time_h
    }

    summary = {
        "experiment": EXPERIMENT_NAME,
        "config": {
            "epochs_configured": EPOCHS,
            "epochs_ran": len(epochs_ran),
            "batch_size": BATCH_SIZE,
            "loss": "class_weighted_cross_entropy",
            "class_weights_source": "training_set_only",
            "class_weights": class_weights_dict,
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
            "stage1_trainable_params": s1_trainable,
            "stage2_trainable_params": s2_trainable,
            "stage3_trainable_params": s3_trainable
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
    print("EXPERIMENT 7 COMPLETE")
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
    print("\nExperiment 7 finished. STOPPED — awaiting next instruction.\n")


if __name__ == "__main__":
    train()
