import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "..", "dataset")
SPLITS_DIR    = os.path.join(DATASET_DIR, "splits")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "experiment3")
EXP2_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "experiment2")

EXPERIMENT_NAME = "experiment3_focal_loss"
CHECKPOINT_NAME = "experiment3_best_model.pt"

EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
PATIENCE      = 5
GAMMA         = 2.0
USE_ATTENTION = True

# Experiment 2 Baseline Metrics for Direct Comparison
EXP2_BASELINE = {
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
}


# ── Focal Loss Implementation ──────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Multi-Class Alpha-Balanced Focal Loss:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the ground-truth class.
    
    alpha: Tensor of class weights calculated STRICTLY from the training set only.
    gamma: Focusing parameter (gamma=2.0) to down-weight easy examples and focus on hard examples.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # Cross entropy loss per sample (unreduced): -log(p_t)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # Probability of true class: p_t = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        # Modulating factor: (1 - p_t)^gamma
        modulating_factor = (1.0 - pt) ** self.gamma
        focal_loss = modulating_factor * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            # Normalized by sum of alpha weights in the batch for stable gradient scaling
            loss = (alpha_t * focal_loss).sum() / (alpha_t.sum() + 1e-8)
        else:
            loss = focal_loss.mean()

        return loss


# ── Calculate Alpha Weights from TRAINING SET ONLY ──────────────────────────────
def calculate_training_alpha_weights(train_df, device):
    """
    Computes alpha weights strictly from the training dataset.
    Formula: alpha_c = N_train / (num_classes * N_c)
    Validation and test data are NEVER used for weight calculation.
    """
    print("\n" + "="*60)
    print("CALCULATING ALPHA WEIGHTS STRICTLY FROM TRAINING SET ONLY")
    print("="*60)
    train_counts = train_df['dx'].value_counts()
    total_train  = len(train_df)
    num_classes  = len(CLASS_MAPPING)

    alpha_dict = {}
    alpha_list = []

    print(f"Total training samples: {total_train}")
    print(f"Number of classes: {num_classes}")
    print("Per-class counts and computed alpha weights (TRAINING SET ONLY):")

    for cls, idx in sorted(CLASS_MAPPING.items(), key=lambda x: x[1]):
        count = int(train_counts.get(cls, 0))
        w = total_train / (num_classes * count) if count > 0 else 1.0
        alpha_dict[cls] = float(w)
        alpha_list.append(w)
        print(f"  Class {idx} [{cls:>5}]: count = {count:>4} ({count/total_train*100:>5.2f}%) -> alpha = {w:.4f}")

    alpha_tensor = torch.tensor(alpha_list, dtype=torch.float32).to(device)
    print("="*60 + "\n")
    return alpha_tensor, alpha_dict


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

    # Analyze specific melanoma error modes
    mel_idx = CLASS_MAPPING["mel"]
    nv_idx  = CLASS_MAPPING["nv"]
    bkl_idx = CLASS_MAPPING["bkl"]

    mel_total = int(cm[mel_idx].sum())
    mel_correct = int(cm[mel_idx][mel_idx])
    mel_to_nv = int(cm[mel_idx][nv_idx])
    mel_to_bkl = int(cm[mel_idx][bkl_idx])

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
    print(f"\n  *** Melanoma (mel) Recall    : {mel_rec:.4f}  ({mel_rec*100:.2f}%) [{mel_correct}/{mel_total} images]")
    print(f"  *** Melanoma (mel) Precision : {mel_prec:.4f}")
    print(f"  *** Melanoma (mel) F1        : {mel_f1:.4f}")
    print(f"  *** Melanoma -> nv errors    : {mel_to_nv} images")
    print(f"  *** Melanoma -> bkl errors   : {mel_to_bkl} images")

    print(f"\n{'--------------------------------------------------'}")
    print("Per-class Metrics:")
    print(f"{'--------------------------------------------------'}")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    # Save text report
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write("EXPERIMENT 3 — FOCAL LOSS (gamma=2.0) Test Set Evaluation\n")
        f.write(f"{'='*50}\n")
        f.write(f"Test Accuracy     : {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Balanced Accuracy : {bacc:.4f} ({bacc*100:.2f}%)\n")
        f.write(f"Macro Precision   : {prec_macro:.4f}\n")
        f.write(f"Macro Recall      : {rec_macro:.4f}\n")
        f.write(f"Macro F1          : {f1_macro:.4f}\n")
        f.write(f"Weighted F1       : {f1_weighted:.4f}\n\n")
        f.write(f"Melanoma Recall   : {mel_rec:.4f} ({mel_rec*100:.2f}%) [{mel_correct}/{mel_total} images]\n")
        f.write(f"Melanoma Precision: {mel_prec:.4f}\n")
        f.write(f"Melanoma F1       : {mel_f1:.4f}\n")
        f.write(f"Melanoma -> nv errors : {mel_to_nv} images\n")
        f.write(f"Melanoma -> bkl errors: {mel_to_bkl} images\n\n")
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
        "melanoma_to_nv_errors": mel_to_nv,
        "melanoma_to_bkl_errors": mel_to_bkl,
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
    plt.title('Confusion Matrix — Test Set (Experiment 3: Focal Loss gamma=2.0)')
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved evaluation artifacts -> {results_dir}")
    return metrics


# ── Comparison Helper ───────────────────────────────────────────────────────────
def compute_comparison_with_experiment2(exp3_metrics, results_dir):
    diff = {
        "accuracy_change": float(exp3_metrics["accuracy"] - EXP2_BASELINE["accuracy"]),
        "balanced_accuracy_change": float(exp3_metrics["balanced_accuracy"] - EXP2_BASELINE["balanced_accuracy"]),
        "macro_f1_change": float(exp3_metrics["macro_f1"] - EXP2_BASELINE["macro_f1"]),
        "macro_precision_change": float(exp3_metrics["macro_precision"] - EXP2_BASELINE["macro_precision"]),
        "macro_recall_change": float(exp3_metrics["macro_recall"] - EXP2_BASELINE["macro_recall"]),
        "weighted_f1_change": float(exp3_metrics["weighted_f1"] - EXP2_BASELINE["weighted_f1"]),
        "melanoma_recall_change": float(exp3_metrics["melanoma_recall"] - EXP2_BASELINE["melanoma_recall"]),
        "melanoma_precision_change": float(exp3_metrics["melanoma_precision"] - EXP2_BASELINE["melanoma_precision"]),
        "melanoma_f1_change": float(exp3_metrics["melanoma_f1"] - EXP2_BASELINE["melanoma_f1"]),
        "mel_to_nv_error_change": int(exp3_metrics["melanoma_to_nv_errors"] - EXP2_BASELINE["mel_to_nv_errors"]),
        "mel_to_bkl_error_change": int(exp3_metrics["melanoma_to_bkl_errors"] - EXP2_BASELINE["mel_to_bkl_errors"]),
    }

    comparison = {
        "experiment2_baseline": EXP2_BASELINE,
        "experiment3_focal_loss": {
            "accuracy": exp3_metrics["accuracy"],
            "balanced_accuracy": exp3_metrics["balanced_accuracy"],
            "macro_precision": exp3_metrics["macro_precision"],
            "macro_recall": exp3_metrics["macro_recall"],
            "macro_f1": exp3_metrics["macro_f1"],
            "weighted_f1": exp3_metrics["weighted_f1"],
            "melanoma_recall": exp3_metrics["melanoma_recall"],
            "melanoma_precision": exp3_metrics["melanoma_precision"],
            "melanoma_f1": exp3_metrics["melanoma_f1"],
            "mel_to_nv_errors": exp3_metrics["melanoma_to_nv_errors"],
            "mel_to_bkl_errors": exp3_metrics["melanoma_to_bkl_errors"],
        },
        "delta_exp3_minus_exp2": diff
    }

    comp_path = os.path.join(results_dir, "comparison_with_experiment2.json")
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=4)

    print("\n" + "="*60)
    print("COMPARISON: EXPERIMENT 3 (FOCAL LOSS) vs EXPERIMENT 2 (CLASS-WEIGHTED CE)")
    print("="*60)
    print(f"  Test Accuracy      : {EXP2_BASELINE['accuracy']*100:.2f}% -> {exp3_metrics['accuracy']*100:.2f}%  (delta: {diff['accuracy_change']*100:+.2f}%)")
    print(f"  Balanced Accuracy  : {EXP2_BASELINE['balanced_accuracy']*100:.2f}% -> {exp3_metrics['balanced_accuracy']*100:.2f}%  (delta: {diff['balanced_accuracy_change']*100:+.2f}%)")
    print(f"  Macro F1           : {EXP2_BASELINE['macro_f1']:.4f} -> {exp3_metrics['macro_f1']:.4f}  (delta: {diff['macro_f1_change']:+.4f})")
    print(f"  Melanoma Recall    : {EXP2_BASELINE['melanoma_recall']*100:.2f}% -> {exp3_metrics['melanoma_recall']*100:.2f}%  (delta: {diff['melanoma_recall_change']*100:+.2f}%)")
    print(f"  Melanoma Precision : {EXP2_BASELINE['melanoma_precision']*100:.2f}% -> {exp3_metrics['melanoma_precision']*100:.2f}%  (delta: {diff['melanoma_precision_change']*100:+.2f}%)")
    print(f"  Melanoma F1        : {EXP2_BASELINE['melanoma_f1']:.4f} -> {exp3_metrics['melanoma_f1']:.4f}  (delta: {diff['melanoma_f1_change']:+.4f})")
    print(f"  Mel -> nv errors   : {EXP2_BASELINE['mel_to_nv_errors']} -> {exp3_metrics['melanoma_to_nv_errors']}  (delta: {diff['mel_to_nv_error_change']:+d} images)")
    print(f"  Mel -> bkl errors  : {EXP2_BASELINE['mel_to_bkl_errors']} -> {exp3_metrics['melanoma_to_bkl_errors']}  (delta: {diff['mel_to_bkl_error_change']:+d} images)")
    print("="*60)
    return comparison


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

    # 2. Compute alpha weights STRICTLY from training set only
    alpha_tensor, alpha_dict = calculate_training_alpha_weights(train_df, device)

    # 3. Configure Focal Loss
    criterion = FocalLoss(alpha=alpha_tensor, gamma=GAMMA)

    # Save exact training loss configuration
    loss_config = {
        "loss_name": "alpha_balanced_focal_loss",
        "gamma": GAMMA,
        "alpha_weighting": "training_set_only",
        "alpha_formula": "alpha_c = N_train / (num_classes * N_c)",
        "formula": "FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)",
        "alpha_values": alpha_dict,
        "classes": list(CLASS_MAPPING.keys()),
        "batch_reduction": "weighted_sum_normalized_by_batch_alpha"
    }
    with open(os.path.join(RESULTS_DIR, "training_loss_configuration.json"), "w") as f:
        json.dump(loss_config, f, indent=4)

    # 4. Datasets and Loaders (identical to Exp 2)
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
        print("[SMOKE TEST] PASSED — proceeding to full training.\n")
        del smoke_imgs, smoke_labels
    except Exception as smoke_err:
        import traceback
        print("\n[SMOKE TEST] FAILED — aborting. Full traceback:")
        traceback.print_exc()
        raise SystemExit(1) from smoke_err

    # 5. Model (identical architecture: MobileNetV3 + CBAM Attention)
    model = DermaAI_MobileNetV3(num_classes=7, use_attention=USE_ATTENTION, pretrained=True)
    model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params — Total: {total_params:,} | Trainable: {trainable_params:,}")

    # Freeze backbone initially (Epochs 1-15)
    for param in model.features.parameters():
        param.requires_grad = False
    trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  (Backbone frozen phase, epochs 1-15) Trainable params: {trainable_frozen:,}")

    # 6. Optimizer & Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3)

    # 7. Training loop
    best_val_loss     = float('inf')
    best_val_epoch    = -1
    epochs_no_improve = 0
    checkpoint_path   = os.path.join(MODELS_DIR, CHECKPOINT_NAME)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    print(f"\nStarting Experiment 3 training for up to {EPOCHS} epochs using Focal Loss (gamma={GAMMA})...")
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

        # Checkpoint selection strictly by validation performance
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

    # 8. Save history
    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    epochs_ran = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs_ran, history['train_loss'], label='Train Loss')
    axes[0].plot(epochs_ran, history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss (Focal Loss gamma=2.0)'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs_ran, history['val_acc'], color='green', label='Val Accuracy')
    axes[1].set_title('Val Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs_ran, history['lr'], color='orange', label='LR')
    axes[2].set_title('Learning Rate'); axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log'); axes[2].legend(); axes[2].grid(True)

    plt.suptitle('Experiment 3 — Focal Loss (gamma=2.0) Training Curves', fontsize=14)
    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"  Training curves -> {curves_path}")

    # 9. Single Held-Out Test Evaluation
    metrics = run_test_evaluation(model, device, test_df, RESULTS_DIR, checkpoint_path)

    # 10. Comparison against Experiment 2 Baseline
    comparison = compute_comparison_with_experiment2(metrics, RESULTS_DIR)

    # 11. Experiment Summary
    summary = {
        "experiment": EXPERIMENT_NAME,
        "config": {
            "epochs_configured": EPOCHS,
            "epochs_ran": len(epochs_ran),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss": "focal_loss",
            "gamma": GAMMA,
            "alpha_source": "training_set_only",
            "alpha_weights": alpha_dict,
            "use_attention": USE_ATTENTION,
            "optimizer": "AdamW",
            "backbone": "MobileNetV3-Large",
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
            "total_params": total_params,
            "trainable_params_frozen_phase": trainable_frozen,
            "trainable_params_full": trainable_params,
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
        "comparison_with_experiment2": comparison
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'='*60}")
    print("EXPERIMENT 3 COMPLETE")
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
    print("\nExperiment 3 finished. STOPPED — awaiting next instruction.\n")


if __name__ == "__main__":
    train()
