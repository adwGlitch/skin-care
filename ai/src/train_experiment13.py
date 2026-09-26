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
from torchvision import transforms
from PIL import Image
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
    from .preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING, REVERSE_CLASS_MAPPING, IMAGENET_MEAN, IMAGENET_STD
except ImportError:
    from model import DermaAI_MobileNetV3
    from preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING, REVERSE_CLASS_MAPPING, IMAGENET_MEAN, IMAGENET_STD

# ── Reproducibility ─────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── Configuration Constants ─────────────────────────────────────────────────────
SRC_DIR       = os.path.dirname(os.path.abspath(__file__))
AI_DIR        = os.path.dirname(SRC_DIR)
DATASET_DIR   = os.path.join(AI_DIR, "dataset")
SPLITS_DIR    = os.path.join(DATASET_DIR, "splits")
MODELS_DIR    = os.path.join(AI_DIR, "models")
RESULTS_DIR   = os.path.join(AI_DIR, "results", "experiment13")

CHECKPOINT_PATH = os.path.join(MODELS_DIR, "experiment13_best_model.pt")

EPOCHS             = 30
FROZEN_EPOCHS      = 15
BATCH_SIZE         = 32
IMAGE_SIZE         = 224  # Standard Experiment 8 input resolution
INITIAL_LR         = 1e-4
FINE_TUNE_LR       = 1e-5
WEIGHT_DECAY       = 1e-4
SCHEDULER_PATIENCE = 5
ALPHA              = 0.75
MELANOMA_MULTIPLIER = 1.20  # THE ONLY EXPERIMENTAL VARIABLE

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Baseline Loaders for Historical Comparison ─────────────────────────────────
def load_baseline_metrics(exp_num):
    path = os.path.join(AI_DIR, "results", f"experiment{exp_num}", "metrics.json")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                d = json.load(f)
            per_class = d.get("per_class", {})
            bcc_rec = per_class.get("bcc", {}).get("recall") if "bcc" in per_class else None
            bkl_rec = per_class.get("bkl", {}).get("recall") if "bkl" in per_class else None
            cm = d.get("confusion_matrix")
            mel_to_nv = d.get("melanoma_to_nv_errors")
            mel_to_bkl = d.get("melanoma_to_bkl_errors")
            nv_rec = d.get("nv_recall")
            nv_to_mel = d.get("nv_to_mel_errors")
            if cm is not None and len(cm) >= 7:
                if mel_to_nv is None:
                    mel_to_nv = int(cm[4][5])
                if mel_to_bkl is None:
                    mel_to_bkl = int(cm[4][2])
                if nv_to_mel is None:
                    nv_to_mel = int(cm[5][4])
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
                "nv_to_mel_errors": nv_to_mel,
                "bcc_recall": bcc_rec,
                "bkl_recall": bkl_rec
            }
        except Exception as e:
            print(f"Warning reading baseline experiment{exp_num}: {e}")
    return {}


# ── Section 22: Calculate & Verify Melanoma-Focused Tempered Class Weights ──────
def calculate_experiment13_class_weights(train_df, device, alpha=0.75, mel_mult=1.20):
    print("\n" + "="*70)
    print("SECTION 22: CLASS WEIGHT CONSTRUCTION & VERIFICATION (TRAINING SET ONLY)")
    print("="*70)

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

    # A. Reproduce Experiment 8 Weights
    expected_val_8 = sum(p_c[cls] * tempered_raw[cls] for cls in CLASS_MAPPING)
    norm_weights_8 = {cls: float(tempered_raw[cls] / expected_val_8) for cls in CLASS_MAPPING}

    print("A. Experiment 8 Baseline Weights (alpha = 0.75):")
    for cls in sorted(CLASS_MAPPING.keys(), key=lambda x: CLASS_MAPPING[x]):
        print(f"   [{cls:>5}]: count = {train_counts[cls]:>4} ({p_c[cls]*100:>5.2f}%) | "
              f"Exp8 Normalized Weight = {norm_weights_8[cls]:.4f}")
    sum_8 = sum(p_c[cls] * norm_weights_8[cls] for cls in CLASS_MAPPING)
    print(f"   Exp 8 Weighted Sum: {sum_8:.6f} (target: 1.000000)")
    assert abs(sum_8 - 1.0) < 1e-4, f"Exp 8 weights failed normalization: {sum_8}"

    # B. Apply Melanoma Multiplier = 1.20
    weights_pre_renorm = {cls: norm_weights_8[cls] for cls in CLASS_MAPPING}
    weights_pre_renorm['mel'] = norm_weights_8['mel'] * mel_mult

    print(f"\nB. Pre-Renormalization Weights (melanoma multiplied by {mel_mult:.2f}):")
    for cls in sorted(CLASS_MAPPING.keys(), key=lambda x: CLASS_MAPPING[x]):
        print(f"   [{cls:>5}]: Pre-Renorm Weight = {weights_pre_renorm[cls]:.4f}")

    # C. Renormalize using existing convention sum_c p(c) * w(c) = 1.000000
    expected_val_13 = sum(p_c[cls] * weights_pre_renorm[cls] for cls in CLASS_MAPPING)
    norm_weights_13 = {cls: float(weights_pre_renorm[cls] / expected_val_13) for cls in CLASS_MAPPING}

    weights_list = [norm_weights_13[REVERSE_CLASS_MAPPING[i]] for i in range(num_classes)]
    weights_tensor = torch.tensor(weights_list, dtype=torch.float32).to(device)

    # D. Verification of Sum = 1.000000
    sum_13 = sum(p_c[cls] * norm_weights_13[cls] for cls in CLASS_MAPPING)
    print(f"\nC. Final Normalized Experiment 13 Weights:")
    for cls in sorted(CLASS_MAPPING.keys(), key=lambda x: CLASS_MAPPING[x]):
        ratio = norm_weights_13[cls] / norm_weights_8[cls]
        print(f"   [{cls:>5}]: Final Normalized Weight = {norm_weights_13[cls]:.4f} | Ratio vs Exp8 = {ratio:.4f}")

    print(f"\nD. Experiment 13 Weighted Sum: {sum_13:.6f} (target: 1.000000)")
    assert abs(sum_13 - 1.0) < 1e-4, f"Exp 13 weights failed normalization: {sum_13}"

    # E. Relative Proportion Invariance Check for Non-Melanoma Classes
    non_mel_classes = [c for c in CLASS_MAPPING.keys() if c != 'mel']
    scaling_factors = [norm_weights_13[c] / norm_weights_8[c] for c in non_mel_classes]
    assert all(np.isclose(s, scaling_factors[0], atol=1e-5) for s in scaling_factors), \
        "Non-melanoma relative proportions must remain exactly invariant!"
    print(f"E. Non-melanoma uniform scaling factor: {scaling_factors[0]:.6f}")
    mel_relative_ratio = (norm_weights_13['mel'] / norm_weights_13['nv']) / (norm_weights_8['mel'] / norm_weights_8['nv'])
    print(f"F. Melanoma relative ratio vs non-melanoma: {mel_relative_ratio:.4f} (target: 1.2000)")
    assert np.isclose(mel_relative_ratio, mel_mult, atol=1e-4), "Melanoma relative ratio must equal 1.20!"
    print("="*70 + "\n")

    weight_artifact = {
        "alpha": alpha,
        "melanoma_multiplier": mel_mult,
        "original_experiment8_weights": norm_weights_8,
        "pre_renormalization_weights": weights_pre_renorm,
        "final_normalized_experiment13_weights": norm_weights_13,
        "expected_weighted_sum": sum_13,
        "class_priors_p_c": p_c,
        "raw_counts": {cls: int(train_counts[cls]) for cls in CLASS_MAPPING}
    }

    return weights_tensor, norm_weights_13, weight_artifact


# ── Requirement 21: Pre-Training Architecture Verification ─────────────────────
def verify_pre_training(model, train_df, val_df, test_df, norm_weights_8, norm_weights_13):
    print("="*70)
    print("REQUIREMENT 21: PRE-TRAINING VERIFICATION")
    print("="*70)

    total_params = sum(p.numel() for p in model.parameters())

    # Phase 1: backbone frozen
    for p in model.features.parameters():
        p.requires_grad = False
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Phase 2: full network unfrozen
    for p in model.parameters():
        p.requires_grad = True
    full_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"1.  Architecture                       : DermaAI_MobileNetV3")
    print(f"2.  Backbone                           : MobileNetV3-Large (weights=IMAGENET1K_V1)")
    print(f"3.  CBAM attention                     : in_planes=960, reduction=16, kernel=7")
    print(f"4.  Parameter count                    : {total_params:,} (Frozen: {frozen_trainable:,}, Full: {full_trainable:,})")
    print(f"5.  Input resolution                   : {IMAGE_SIZE}x{IMAGE_SIZE} (Identical to Experiment 8)")
    print(f"6.  Dataset split                      : Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")
    print(f"7.  Seed                               : {SEED}")
    print(f"8.  Batch size                         : {BATCH_SIZE}")
    print(f"9.  Experiment 8 class weights         : {norm_weights_8}")
    print(f"10. Melanoma multiplier                : {MELANOMA_MULTIPLIER:.2f} (ONLY EXPERIMENTAL VARIABLE)")
    print(f"11. Final normalized weights           : {norm_weights_13}")
    print(f"12. Loss type                          : CrossEntropyLoss(weight=weights_tensor)")
    print(f"13. Optimizer                          : AdamW (weight_decay={WEIGHT_DECAY})")
    print(f"14. Learning rates                     : Phase 1 = {INITIAL_LR}, Phase 2 = {FINE_TUNE_LR}")
    print(f"15. Scheduler                          : ReduceLROnPlateau(mode='min', factor=0.1, patience={SCHEDULER_PATIENCE})")
    print(f"16. Training augmentation              : HFlip, VFlip, Rotate(20), ColorJitter(0.1,0.1,0.1), Normalize")
    print(f"17. Class mapping                      : {CLASS_MAPPING}")
    print("="*70 + "\n")

    return {
        "architecture_name": "DermaAI_MobileNetV3 (MobileNetV3-Large + CBAM)",
        "total_parameters": total_params,
        "phase1_trainable_parameters": frozen_trainable,
        "phase2_trainable_parameters": full_trainable,
        "input_resolution": [IMAGE_SIZE, IMAGE_SIZE],
        "dataset_split": {
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df)
        },
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW",
        "weight_decay": WEIGHT_DECAY,
        "initial_lr": INITIAL_LR,
        "fine_tune_lr": FINE_TUNE_LR,
        "scheduler": f"ReduceLROnPlateau(mode='min', factor=0.1, patience={SCHEDULER_PATIENCE})",
        "melanoma_multiplier": MELANOMA_MULTIPLIER,
        "loss": "Tempered Class-Weighted Cross-Entropy (alpha=0.75, mel_multiplier=1.20)"
    }


# ── Requirement 23: Pre-Training Smoke Test ────────────────────────────────────
def run_smoke_test(model, criterion, optimizer, device):
    print("="*60)
    print("REQUIREMENT 23: ONE-BATCH SMOKE TEST")
    print("="*60)

    model.train()
    dummy_input = torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_target = torch.randint(0, 7, (BATCH_SIZE,)).to(device)

    # 1. Input shape check
    assert dummy_input.shape == (BATCH_SIZE, 3, 224, 224), f"Input shape mismatch: {dummy_input.shape}"
    print(f"  Input shape check          : PASSED ({dummy_input.shape})")

    # 2. Forward pass check
    optimizer.zero_grad()
    dummy_out = model(dummy_input)
    assert dummy_out.shape == (BATCH_SIZE, 7), f"Output shape mismatch: {dummy_out.shape}"
    assert not torch.isnan(dummy_out).any(), "NaN in forward pass output"
    assert not torch.isinf(dummy_out).any(), "Inf in forward pass output"
    print(f"  Forward pass check         : PASSED ({dummy_out.shape})")

    # 3. Loss calculation check
    loss = criterion(dummy_out, dummy_target)
    assert not torch.isnan(loss), "NaN in loss computation"
    assert not torch.isinf(loss), "Inf in loss computation"
    assert loss.item() > 0, "Loss must be positive"
    print(f"  Loss calculation check     : PASSED (loss = {loss.item():.4f})")

    # 4. Backward pass check
    loss.backward()
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0, "Zero gradient norm after backward pass"
    assert not np.isnan(grad_norm), "NaN in gradients"
    assert not np.isinf(grad_norm), "Inf in gradients"
    print(f"  Backward pass check        : PASSED (grad norm = {grad_norm:.4f})")

    # 5. Optimizer step check
    optimizer.step()
    optimizer.zero_grad()
    print(f"  Optimizer step check       : PASSED")
    print("="*60)
    print("SMOKE TEST COMPLETE -- ALL CHECKS PASSED. STARTING FULL TRAINING.\n")


# ── Training & Detailed Validation Monitoring Loop ─────────────────────────────
def train_model(model, train_loader, val_loader, criterion, device):
    history = {
        "epoch": [],
        "phase": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "val_macro_precision": [],
        "val_macro_recall": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "val_melanoma_precision": [],
        "val_melanoma_recall": [],
        "val_melanoma_f1": [],
        "learning_rate": [],
        "trainable_params": [],
        "epoch_duration_seconds": []
    }

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    best_epoch    = -1

    phase1_start = time.time()

    # Phase 1: Freeze backbone
    print("\n" + "="*60)
    print("PHASE 1: TRAINING HEAD & CBAM (EPOCHS 1-15, BACKBONE FROZEN)")
    print(f"Head Learning Rate: {INITIAL_LR}")
    print("="*60)

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if hasattr(model, 'attention'):
        for p in model.attention.parameters():
            p.requires_grad = True

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=INITIAL_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=SCHEDULER_PATIENCE
    )

    total_training_start = time.time()
    phase1_dur = 0.0
    phase2_dur = 0.0

    class_names = [REVERSE_CLASS_MAPPING[i] for i in range(7)]
    mel_idx = CLASS_MAPPING['mel']

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        # Phase transition at Epoch 16
        if epoch == FROZEN_EPOCHS + 1:
            phase1_dur = time.time() - total_training_start
            print("\n" + "="*60)
            print("PHASE 2: FULL FINE-TUNING (EPOCHS 16-30, ALL LAYERS UNFROZEN)")
            print(f"Fine-Tuning Learning Rate: {FINE_TUNE_LR}")
            print("="*60)

            for p in model.parameters():
                p.requires_grad = True

            optimizer = optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.1, patience=SCHEDULER_PATIENCE
            )

        current_phase = 1 if epoch <= FROZEN_EPOCHS else 2
        trainable_p   = sum(p.numel() for p in model.parameters() if p.requires_grad)
        current_lr    = optimizer.param_groups[0]['lr']

        # Training Step
        model.train()
        running_loss = 0.0
        train_batches = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch:2d}/{EPOCHS} [Phase {current_phase}]", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_batches += 1

        train_loss = running_loss / train_batches

        # Validation Step with Comprehensive Metrics
        model.eval()
        val_loss_total = 0.0
        val_batches    = 0
        all_val_preds  = []
        all_val_targets= []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss_total += loss.item()
                val_batches += 1

                _, preds = torch.max(outputs, 1)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())

        val_loss = val_loss_total / val_batches
        all_val_preds = np.array(all_val_preds)
        all_val_targets = np.array(all_val_targets)

        val_acc  = accuracy_score(all_val_targets, all_val_preds)
        val_bacc = balanced_accuracy_score(all_val_targets, all_val_preds)
        val_prec_m, val_rec_m, val_f1_m, _ = precision_recall_fscore_support(
            all_val_targets, all_val_preds, average='macro', zero_division=0)
        _, _, val_f1_w, _ = precision_recall_fscore_support(
            all_val_targets, all_val_preds, average='weighted', zero_division=0)

        # Melanoma validation metrics
        val_report = classification_report(
            all_val_targets, all_val_preds, target_names=class_names,
            output_dict=True, zero_division=0)
        val_mel_prec = float(val_report['mel']['precision'])
        val_mel_rec  = float(val_report['mel']['recall'])
        val_mel_f1   = float(val_report['mel']['f1-score'])

        scheduler.step(val_loss)
        epoch_dur = time.time() - epoch_start

        history["epoch"].append(epoch)
        history["phase"].append(current_phase)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["val_balanced_accuracy"].append(val_bacc)
        history["val_macro_precision"].append(float(val_prec_m))
        history["val_macro_recall"].append(float(val_rec_m))
        history["val_macro_f1"].append(float(val_f1_m))
        history["val_weighted_f1"].append(float(val_f1_w))
        history["val_melanoma_precision"].append(val_mel_prec)
        history["val_melanoma_recall"].append(val_mel_rec)
        history["val_melanoma_f1"].append(val_mel_f1)
        history["learning_rate"].append(current_lr)
        history["trainable_params"].append(trainable_p)
        history["epoch_duration_seconds"].append(epoch_dur)

        print(f"Epoch {epoch:2d} (Phase {current_phase}) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"BalAcc: {val_bacc:.4f} | "
              f"Mel Rec: {val_mel_rec:.4f} | "
              f"Mel F1: {val_mel_f1:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_dur:.1f}s")

        # Checkpoint Selection: lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_epoch    = epoch
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  OK Saved best model checkpoint (val_loss={val_loss:.4f}, val_acc={val_acc:.4f})")

    total_training_time = time.time() - total_training_start
    phase2_dur = total_training_time - phase1_dur
    print("="*60)
    print(f"Training complete -- best epoch: {best_epoch} | "
          f"best val loss: {best_val_loss:.4f} | "
          f"best val acc: {best_val_acc:.4f} | "
          f"total time: {total_training_time:.0f}s ({total_training_time/3600:.2f}h)")

    return history, best_epoch, best_val_loss, best_val_acc, total_training_time, phase1_dur, phase2_dur


# ── Plot Training Curves ───────────────────────────────────────────────────────
def plot_training_curves(history, save_path):
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss Curves
    axes[0].plot(epochs, history["train_loss"], 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, history["val_loss"], 'r-s', label='Val Loss', markersize=4)
    axes[0].axvline(x=15.5, color='gray', linestyle='--', label='Phase 2 Unfreeze')
    axes[0].set_title('Experiment 13 Loss Curves', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Tempered CE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy & Melanoma F1 Curves
    axes[1].plot(epochs, [a * 100 for a in history["val_accuracy"]], 'g-^', label='Val Accuracy (%)', markersize=4)
    axes[1].plot(epochs, [m * 100 for m in history["val_melanoma_f1"]], 'm-d', label='Val Mel F1 (%)', markersize=4)
    axes[1].axvline(x=15.5, color='gray', linestyle='--', label='Phase 2 Unfreeze')
    axes[1].set_title('Experiment 13 Validation Accuracy & Melanoma F1', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Score (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ── Test Set Evaluation (Held-Out 988 Images, Evaluated Once) ───────────────────
def evaluate_test_set(model, device, test_df):
    print("\n" + "="*60)
    print("TEST SET EVALUATION (HELD-OUT TEST SPLIT -- 988 IMAGES)")
    print("="*60)

    test_dataset = HAM10000Dataset(test_df, transform=get_transforms(is_train=False))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # Load frozen best checkpoint
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    all_preds   = []
    all_targets = []
    test_start = time.time()

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Test inference"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    test_duration = time.time() - test_start
    all_targets = np.array(all_targets)
    all_preds   = np.array(all_preds)
    class_names = [REVERSE_CLASS_MAPPING[i] for i in range(7)]

    acc  = accuracy_score(all_targets, all_preds)
    bacc = balanced_accuracy_score(all_targets, all_preds)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='macro', zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='weighted', zero_division=0)

    report_dict = classification_report(
        all_targets, all_preds, target_names=class_names,
        output_dict=True, zero_division=0)
    report_text = classification_report(
        all_targets, all_preds, target_names=class_names, zero_division=0)

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(7)))

    mel_idx   = CLASS_MAPPING['mel']
    nv_idx    = CLASS_MAPPING['nv']
    bkl_idx   = CLASS_MAPPING['bkl']
    bcc_idx   = CLASS_MAPPING['bcc']
    akiec_idx = CLASS_MAPPING['akiec']

    mel_total = int(sum(all_targets == mel_idx))
    mel_pred_total = int(sum(all_preds == mel_idx))
    mel_tp = int(cm[mel_idx, mel_idx])
    mel_fp = mel_pred_total - mel_tp

    mel_to_nv    = int(cm[mel_idx, nv_idx])
    mel_to_bkl   = int(cm[mel_idx, bkl_idx])
    mel_to_akiec = int(cm[mel_idx, akiec_idx])
    mel_to_bcc   = int(cm[mel_idx, bcc_idx])

    nv_total = int(sum(all_targets == nv_idx))
    nv_tp = int(cm[nv_idx, nv_idx])
    nv_rec  = float(report_dict['nv']['recall'])
    nv_prec = float(report_dict['nv']['precision'])
    nv_f1   = float(report_dict['nv']['f1-score'])
    nv_to_mel = int(cm[nv_idx, mel_idx])

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
    print(f"  Melanoma (mel) Recall  : {mel_rec:.4f}  ({mel_rec*100:.2f}% -- {mel_tp}/{mel_total} images)")
    print(f"  Melanoma (mel) Prec    : {mel_prec:.4f}")
    print(f"  Melanoma (mel) F1      : {mel_f1:.4f}")
    print(f"  Total Mel Predictions  : {mel_pred_total} (False Positives: {mel_fp})")
    print(f"  mel -> nv errors       : {mel_to_nv} images")
    print(f"  mel -> bkl errors      : {mel_to_bkl} images")
    print(f"  mel -> akiec errors    : {mel_to_akiec} images")
    print(f"  mel -> bcc errors      : {mel_to_bcc} images")
    print(f"  Majority (nv) Recall   : {nv_rec:.4f}  ({nv_tp}/{nv_total} images)")
    print(f"  Majority (nv) Prec     : {nv_prec:.4f}")
    print(f"  Majority (nv) F1       : {nv_f1:.4f}")
    print(f"  nv -> mel errors       : {nv_to_mel} images")
    print(f"  BCC Recall             : {bcc_rec:.4f}")
    print(f"  BKL Recall             : {bkl_rec:.4f}")
    print(f"{'--------------------------------------------------'}")
    print("\nPer-class Metrics:\n" + report_text)
    print("Confusion Matrix:\n", cm)

    test_metrics = {
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
        "melanoma_correct_images": mel_tp,
        "melanoma_total_images": mel_total,
        "melanoma_predicted_total": mel_pred_total,
        "melanoma_false_positives": mel_fp,
        "melanoma_to_nv_errors": mel_to_nv,
        "melanoma_to_bkl_errors": mel_to_bkl,
        "melanoma_to_akiec_errors": mel_to_akiec,
        "melanoma_to_bcc_errors": mel_to_bcc,
        "nv_recall": nv_rec,
        "nv_precision": nv_prec,
        "nv_f1": nv_f1,
        "nv_correct_images": nv_tp,
        "nv_total_images": nv_total,
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
        "confusion_matrix": cm.tolist(),
        "test_inference_duration_seconds": test_duration
    }

    return test_metrics, report_text, cm


# ── Main Experiment Execution ──────────────────────────────────────────────────
def main():
    total_start = time.time()
    device = torch.device("cpu")
    print("="*70)
    print("EXPERIMENT 13: MELANOMA-FOCUSED TEMPERED CLASS WEIGHTING")
    print(f"Device        : {device} ({platform.processor()})")
    print(f"PyTorch       : {torch.__version__} | torchvision: {torchvision.__version__}")
    print("="*70)

    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))

    print(f"Dataset splits verified: Train={len(train_df):,} | Val={len(val_df):,} | Test={len(test_df):,}")

    # Calculate & verify weights
    weights_tensor, norm_weights_13, weight_artifact = calculate_experiment13_class_weights(
        train_df, device, alpha=ALPHA, mel_mult=MELANOMA_MULTIPLIER
    )

    # Initialize model from ImageNet pretrained weights (independent initialization)
    model = DermaAI_MobileNetV3(num_classes=7, use_attention=True, pretrained=True)
    model.to(device)

    # Architecture verification before training
    arch_config = verify_pre_training(
        model, train_df, val_df, test_df,
        weight_artifact["original_experiment8_weights"], norm_weights_13
    )

    # Loss: Tempered Class-Weighted Cross Entropy with 20% melanoma increase
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Optimizer & smoke test
    optimizer = optim.AdamW(model.parameters(), lr=INITIAL_LR, weight_decay=WEIGHT_DECAY)
    run_smoke_test(model, criterion, optimizer, device)

    # DataLoaders (natural unweighted random shuffle, 224x224 standard)
    train_dataset = HAM10000Dataset(train_df, transform=get_transforms(is_train=True))
    val_dataset   = HAM10000Dataset(val_df, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # Execute 30-epoch 2-phase training
    history, best_epoch, best_val_loss, best_val_acc, train_dur, p1_dur, p2_dur = train_model(
        model, train_loader, val_loader, criterion, device
    )

    # Plot training curves
    plot_training_curves(history, os.path.join(RESULTS_DIR, "training_curves.png"))

    # Held-out test evaluation (single pass)
    test_metrics, report_text, cm = evaluate_test_set(model, device, test_df)

    # Plot confusion matrix
    class_names = [REVERSE_CLASS_MAPPING[i] for i in range(7)]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title('Experiment 13 Confusion Matrix - Test Split', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Save all 15 required artifacts
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report_text)
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "confusion_matrix.json"), "w") as f:
        json.dump(cm.tolist(), f, indent=4)
    with open(os.path.join(RESULTS_DIR, "architecture_configuration.json"), "w") as f:
        json.dump(arch_config, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "training_class_weights.json"), "w") as f:
        json.dump(weight_artifact, f, indent=4)

    param_summary = {
        "total_parameters": arch_config["total_parameters"],
        "phase1_trainable_parameters": arch_config["phase1_trainable_parameters"],
        "phase2_trainable_parameters": arch_config["phase2_trainable_parameters"]
    }
    with open(os.path.join(RESULTS_DIR, "parameter_summary.json"), "w") as f:
        json.dump(param_summary, f, indent=4)

    loss_config = {
        "loss_name": "Melanoma-Focused Tempered Class-Weighted Cross-Entropy",
        "alpha": ALPHA,
        "melanoma_multiplier": MELANOMA_MULTIPLIER,
        "base_loss": "torch.nn.CrossEntropyLoss",
        "weights_tensor": [float(w) for w in weights_tensor.cpu().numpy()],
        "expected_weighted_sum": weight_artifact["expected_weighted_sum"]
    }
    with open(os.path.join(RESULTS_DIR, "loss_configuration.json"), "w") as f:
        json.dump(loss_config, f, indent=4)

    prep_config = {
        "input_resolution": [224, 224],
        "image_size": 224,
        "normalization_mean": IMAGENET_MEAN,
        "normalization_std": IMAGENET_STD,
        "interpolation": "BILINEAR"
    }
    with open(os.path.join(RESULTS_DIR, "preprocessing_configuration.json"), "w") as f:
        json.dump(prep_config, f, indent=4)

    aug_config = {
        "policy": "Experiment 8 standard augmentation",
        "transforms": [
            "Resize((224, 224))",
            "RandomHorizontalFlip(p=0.5)",
            "RandomVerticalFlip(p=0.5)",
            "RandomRotation(degrees=20)",
            "ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)",
            "ToTensor()",
            "Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])"
        ]
    }
    with open(os.path.join(RESULTS_DIR, "augmentation_configuration.json"), "w") as f:
        json.dump(aug_config, f, indent=4)

    mel_mod_artifact = {
        "rationale": "Test whether mild 20% increase in relative melanoma training weight improves melanoma recall while preserving overall accuracy",
        "multiplier": MELANOMA_MULTIPLIER,
        "exp8_mel_weight": weight_artifact["original_experiment8_weights"]["mel"],
        "pre_renorm_mel_weight": weight_artifact["pre_renormalization_weights"]["mel"],
        "final_norm_mel_weight": weight_artifact["final_normalized_experiment13_weights"]["mel"],
        "relative_ratio_vs_non_mel": 1.2000
    }
    with open(os.path.join(RESULTS_DIR, "melanoma_weight_modification.json"), "w") as f:
        json.dump(mel_mod_artifact, f, indent=4)

    comp_cost = {
        "mean_epoch_duration_seconds": float(np.mean(history["epoch_duration_seconds"])),
        "phase1_duration_seconds": p1_dur,
        "phase2_duration_seconds": p2_dur,
        "total_training_duration_seconds": train_dur,
        "test_inference_duration_seconds": test_metrics["test_inference_duration_seconds"],
        "total_experiment_duration_seconds": time.time() - total_start,
        "comparison_with_exp8_exp9_exp12": {
            "exp8_training_time_seconds": 9877.0,
            "exp9_training_time_seconds": 9877.0,
            "exp12_training_time_seconds": 76950.0,
            "exp13_training_time_seconds": train_dur
        }
    }
    with open(os.path.join(RESULTS_DIR, "computational_cost.json"), "w") as f:
        json.dump(comp_cost, f, indent=4)

    # Comparisons with Experiments 1 through 12
    for i in range(1, 13):
        base_d = load_baseline_metrics(i)
        delta_d = {}
        if base_d:
            delta_d = {
                "accuracy_change": test_metrics["accuracy"] - (base_d.get("accuracy") or 0.0),
                "balanced_accuracy_change": test_metrics["balanced_accuracy"] - (base_d.get("balanced_accuracy") or 0.0),
                "macro_f1_change": test_metrics["macro_f1"] - (base_d.get("macro_f1") or 0.0),
                "weighted_f1_change": test_metrics["weighted_f1"] - (base_d.get("weighted_f1") or 0.0),
                "melanoma_recall_change": test_metrics["melanoma_recall"] - (base_d.get("melanoma_recall") or 0.0),
                "melanoma_precision_change": test_metrics["melanoma_precision"] - (base_d.get("melanoma_precision") or 0.0),
                "melanoma_f1_change": test_metrics["melanoma_f1"] - (base_d.get("melanoma_f1") or 0.0)
            }
        comp_file = {
            f"experiment{i}_baseline": base_d,
            "experiment13_melanoma_weighted": {
                "accuracy": test_metrics["accuracy"],
                "balanced_accuracy": test_metrics["balanced_accuracy"],
                "macro_f1": test_metrics["macro_f1"],
                "weighted_f1": test_metrics["weighted_f1"],
                "melanoma_recall": test_metrics["melanoma_recall"],
                "melanoma_precision": test_metrics["melanoma_precision"],
                "melanoma_f1": test_metrics["melanoma_f1"]
            },
            f"delta_exp13_minus_exp{i}": delta_d
        }
        with open(os.path.join(RESULTS_DIR, f"comparison_with_experiment{i}.json"), "w") as f:
            json.dump(comp_file, f, indent=4)

    total_time = time.time() - total_start
    exp_summary = {
        "experiment": "Experiment 13: Melanoma-Focused Tempered Class Weighting (Multiplier = 1.20)",
        "experimental_variable": "relative melanoma class weight (1.20x)",
        "base_control": "Experiment 8 (Tempered Class Weights alpha=0.75)",
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_melanoma_recall": test_metrics["melanoma_recall"],
        "test_melanoma_precision": test_metrics["melanoma_precision"],
        "test_melanoma_f1": test_metrics["melanoma_f1"],
        "training_duration_seconds": train_dur,
        "total_experiment_duration_seconds": total_time,
        "checkpoint_path": CHECKPOINT_PATH
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(exp_summary, f, indent=4)

    # Primary Comparison Printout (Experiment 13 vs Experiment 8)
    exp8_base = load_baseline_metrics(8)
    print("\n" + "="*70)
    print("PRIMARY COMPARISON: EXPERIMENT 13 vs EXPERIMENT 8 BASELINE")
    print("="*70)
    if exp8_base:
        d_acc = test_metrics["accuracy"] - exp8_base["accuracy"]
        d_bacc = test_metrics["balanced_accuracy"] - exp8_base["balanced_accuracy"]
        d_mf1 = test_metrics["macro_f1"] - exp8_base["macro_f1"]
        d_wf1 = test_metrics["weighted_f1"] - exp8_base["weighted_f1"]
        d_mrec = test_metrics["melanoma_recall"] - exp8_base["melanoma_recall"]
        d_mprec = test_metrics["melanoma_precision"] - exp8_base["melanoma_precision"]
        d_mf1_score = test_metrics["melanoma_f1"] - exp8_base["melanoma_f1"]
        d_mel_nv = test_metrics["melanoma_to_nv_errors"] - exp8_base["mel_to_nv_errors"]
        d_mel_bkl = test_metrics["melanoma_to_bkl_errors"] - exp8_base["mel_to_bkl_errors"]
        d_nv_rec = test_metrics["nv_recall"] - exp8_base["nv_recall"]
        d_nv_to_mel = test_metrics["nv_to_mel_errors"] - exp8_base["nv_to_mel_errors"]

        print(f"  Test Accuracy          : {exp8_base['accuracy']*100:.2f}% -> {test_metrics['accuracy']*100:.2f}%  (delta: {d_acc*100:+.2f}%)")
        print(f"  Balanced Accuracy      : {exp8_base['balanced_accuracy']*100:.2f}% -> {test_metrics['balanced_accuracy']*100:.2f}%  (delta: {d_bacc*100:+.2f}%)")
        print(f"  Macro F1               : {exp8_base['macro_f1']:.4f} -> {test_metrics['macro_f1']:.4f}  (delta: {d_mf1:+.4f})")
        print(f"  Weighted F1            : {exp8_base['weighted_f1']:.4f} -> {test_metrics['weighted_f1']:.4f}  (delta: {d_wf1:+.4f})")
        print(f"  Melanoma Recall        : {exp8_base['melanoma_recall']*100:.2f}% -> {test_metrics['melanoma_recall']*100:.2f}%  (delta: {d_mrec*100:+.2f}%)")
        print(f"  Melanoma Precision     : {exp8_base['melanoma_precision']*100:.2f}% -> {test_metrics['melanoma_precision']*100:.2f}%  (delta: {d_mprec*100:+.2f}%)")
        print(f"  Melanoma F1            : {exp8_base['melanoma_f1']:.4f} -> {test_metrics['melanoma_f1']:.4f}  (delta: {d_mf1_score:+.4f})")
        print(f"  Mel -> NV Misses       : {exp8_base['mel_to_nv_errors']} -> {test_metrics['melanoma_to_nv_errors']}  (delta: {d_mel_nv:+d})")
        print(f"  Mel -> BKL Misses      : {exp8_base['mel_to_bkl_errors']} -> {test_metrics['melanoma_to_bkl_errors']}  (delta: {d_mel_bkl:+d})")
        print(f"  Nevus (NV) Recall      : {exp8_base['nv_recall']*100:.2f}% -> {test_metrics['nv_recall']*100:.2f}%  (delta: {d_nv_rec*100:+.2f}%)")
        print(f"  NV -> MEL Errors       : {exp8_base['nv_to_mel_errors']} -> {test_metrics['nv_to_mel_errors']}  (delta: {d_nv_to_mel:+d})")
    print("="*70)
    print(f"All 27 Experiment 13 artifacts saved to: {RESULTS_DIR}")
    print(f"Model saved to: {CHECKPOINT_PATH}")
    print("Experiment 13 complete. STOPPING as required -- awaiting next instruction.")


if __name__ == "__main__":
    main()
