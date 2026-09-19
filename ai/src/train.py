import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe on Windows CPU
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

# -- Config ----------------------------------------------------------------------
DATASET_DIR   = os.path.join(os.path.dirname(__file__), "..", "dataset")
SPLITS_DIR    = os.path.join(DATASET_DIR, "splits")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
MODELS_DIR    = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "experiment1")

EXPERIMENT_NAME = "experiment1_ham10000"
CHECKPOINT_NAME = "experiment1_best_model.pt"

EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
PATIENCE      = 5
LOSS_TYPE     = "weighted_cross_entropy"
USE_ATTENTION = True


# -- Loss ------------------------------------------------------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss    = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt         = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def get_loss_function(loss_type, device):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()

    weights_path = os.path.join(PROCESSED_DIR, "class_weights.json")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing {weights_path}. Run prepare_data.py first.")

    with open(weights_path) as f:
        weight_dict = json.load(f)

    weights = [weight_dict[k] for k in CLASS_MAPPING.keys()]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    if loss_type == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=weights_tensor)
    elif loss_type == "focal":
        return FocalLoss(alpha=weights_tensor, gamma=2)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# -- Test evaluation -------------------------------------------------------------
def run_test_evaluation(model, device, test_df, results_dir, checkpoint_path):
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)

    test_dataset = HAM10000Dataset(test_df, transform=get_transforms(is_train=False))
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    # Load best checkpoint weights
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

    print(f"\n{'-'*50}")
    print(f"  Test Accuracy          : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Balanced Accuracy      : {bacc:.4f}  ({bacc*100:.2f}%)")
    print(f"  Macro Precision        : {prec_macro:.4f}")
    print(f"  Macro Recall           : {rec_macro:.4f}")
    print(f"  Macro F1               : {f1_macro:.4f}")
    print(f"  Weighted F1            : {f1_weighted:.4f}")
    print(f"{'-'*50}")

    mel_rec = mel_prec = mel_f1 = None
    if "mel" in report_dict:
        mel_rec  = report_dict["mel"]["recall"]
        mel_prec = report_dict["mel"]["precision"]
        mel_f1   = report_dict["mel"]["f1-score"]
        print(f"\n  *** Melanoma (mel) Recall    : {mel_rec:.4f}")
        print(f"  *** Melanoma (mel) Precision : {mel_prec:.4f}")
        print(f"  *** Melanoma (mel) F1        : {mel_f1:.4f}")

    print(f"\n{'-'*50}")
    print("Per-class Metrics:")
    print(f"{'-'*50}")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    # Save text report
    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write("EXPERIMENT 1 — HAM10000 Test Set Evaluation\n")
        f.write(f"{'='*50}\n")
        f.write(f"Test Accuracy     : {acc:.4f}\n")
        f.write(f"Balanced Accuracy : {bacc:.4f}\n")
        f.write(f"Macro Precision   : {prec_macro:.4f}\n")
        f.write(f"Macro Recall      : {rec_macro:.4f}\n")
        f.write(f"Macro F1          : {f1_macro:.4f}\n")
        f.write(f"Weighted F1       : {f1_weighted:.4f}\n\n")
        if mel_rec is not None:
            f.write(f"Melanoma Recall   : {mel_rec:.4f}\n")
            f.write(f"Melanoma Precision: {mel_prec:.4f}\n")
            f.write(f"Melanoma F1       : {mel_f1:.4f}\n\n")
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
        "melanoma_recall": float(mel_rec) if mel_rec is not None else None,
        "melanoma_precision": float(mel_prec) if mel_prec is not None else None,
        "melanoma_f1": float(mel_f1) if mel_f1 is not None else None,
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
    plt.title('Confusion Matrix — Test Set (Experiment 1)')
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved evaluation artifacts -> {results_dir}")
    return metrics


# -- Main ------------------------------------------------------------------------
def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    training_start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Experiment  : {EXPERIMENT_NAME}")

    # 1. Load splits
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val_df   = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    test_df  = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    print(f"  Train: {len(train_df)} images | Val: {len(val_df)} images | Test: {len(test_df)} images")

    train_dataset = HAM10000Dataset(train_df, transform=get_transforms(is_train=True))
    val_dataset   = HAM10000Dataset(val_df,   transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"Diagnostic -> Device: {device}, num_workers: 0, pin_memory: False, batch_size: {BATCH_SIZE}")

    # -- SMOKE TEST --------------------------------------------------------------
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

    # 2. Model
    model = DermaAI_MobileNetV3(num_classes=7, use_attention=USE_ATTENTION, pretrained=True)
    model.to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params — Total: {total_params:,} | Trainable: {trainable_params:,}")

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False
    trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  (Backbone frozen) Trainable: {trainable_frozen:,}")

    # 3. Loss / Optimizer / Scheduler
    criterion = get_loss_function(LOSS_TYPE, device)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3)

    # 4. Training loop
    best_val_loss     = float('inf')
    best_val_epoch    = -1
    epochs_no_improve = 0
    checkpoint_path   = os.path.join(MODELS_DIR, CHECKPOINT_NAME)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    print(f"\nStarting training for up to {EPOCHS} epochs using {LOSS_TYPE} loss...")
    print("="*60)

    for epoch in range(EPOCHS):
        # Unfreeze backbone at midpoint
        if epoch == EPOCHS // 2:
            print("\nUnfreezing backbone for fine-tuning...")
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

    # 5. Save history
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

    plt.suptitle('Experiment 1 — HAM10000 Training Curves', fontsize=14)
    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"  Training curves -> {curves_path}")

    # 6. Test evaluation
    metrics = run_test_evaluation(model, device, test_df, RESULTS_DIR, checkpoint_path)

    # 7. Experiment summary
    summary = {
        "experiment": EXPERIMENT_NAME,
        "config": {
            "epochs_configured": EPOCHS,
            "epochs_ran": len(epochs_ran),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss": LOSS_TYPE,
            "use_attention": USE_ATTENTION,
            "optimizer": "AdamW",
            "backbone": "MobileNetV3-Large",
            "device": str(device),
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
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'='*60}")
    print("EXPERIMENT 1 COMPLETE")
    print(f"{'='*60}")
    print(f"  Best epoch      : {best_val_epoch}")
    print(f"  Test Accuracy   : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Balanced Acc    : {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro F1        : {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1     : {metrics['weighted_f1']:.4f}")
    if metrics.get('melanoma_recall') is not None:
        print(f"  Melanoma Recall : {metrics['melanoma_recall']:.4f}")
    print(f"  Training time   : {training_time_s:.0f}s ({training_time_h:.2f}h)")
    print(f"  Checkpoint      : {checkpoint_path}")
    print(f"  Results dir     : {RESULTS_DIR}")
    print(f"{'='*60}")
    print("\nExperiment 1 finished. STOPPED — awaiting next instruction.\n")


if __name__ == "__main__":
    train()

