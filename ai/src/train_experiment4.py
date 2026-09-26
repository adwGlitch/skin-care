import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
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
RESULTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "results", "experiment4")

EXPERIMENT_NAME = "experiment4_class_balanced_sampling"
CHECKPOINT_NAME = "experiment4_best_model.pt"

EPOCHS        = 30
BATCH_SIZE    = 32
LEARNING_RATE = 1e-4
PATIENCE      = 5
USE_ATTENTION = True

# Baselines for comparison
EXP1_BASELINE = {
    "accuracy": 0.7317813765182186,
    "balanced_accuracy": 0.7307211250816138,
    "macro_precision": 0.6015076591786525,
    "macro_recall": 0.7307211250816138,
    "macro_f1": 0.6394575183439128,
    "weighted_f1": 0.7486763834648812,
    "melanoma_precision": 0.36923076923076925,
    "melanoma_recall": 0.4528301886792453,
    "melanoma_f1": 0.4067796610169492,
}

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


# ── Class-Balanced Sampler Construction (TRAINING SET ONLY) ─────────────────────
def setup_class_balanced_sampler(train_df):
    """
    Constructs a WeightedRandomSampler where sample selection probabilities are
    inversely proportional to class frequencies calculated STRICTLY from the training set.
    Validation and test labels are NEVER touched.
    """
    print("\n" + "="*60)
    print("CONFIGURING CLASS-BALANCED SAMPLER (TRAINING SET ONLY)")
    print("="*60)
    
    train_labels = train_df['dx'].map(CLASS_MAPPING).values
    class_counts = pd.Series(train_labels).value_counts().sort_index()
    total_train = len(train_labels)
    num_classes = len(CLASS_MAPPING)

    # Class weight = 1.0 / count_c
    class_weights = 1.0 / class_counts.values
    # Per-sample weights
    sample_weights = class_weights[train_labels]
    
    # Sampler: draws exactly total_train samples per epoch with replacement
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=total_train,
        replacement=True
    )

    # Calculate theoretical sampling probabilities
    # P(class c) = count_c * (1.0 / count_c) / sum(count_k * (1.0 / count_k)) = 1 / C = 14.2857%
    sampling_info = {}
    distribution_info = {}

    print(f"Total training samples: {total_train}")
    print(f"Number of classes: {num_classes}")
    print("Class breakdown and sampling configuration:")
    for cls, idx in sorted(CLASS_MAPPING.items(), key=lambda x: x[1]):
        count = int(class_counts[idx])
        orig_pct = (count / total_train) * 100.0
        sampling_prob = (1.0 / num_classes) * 100.0  # Exactly 14.2857%
        sampling_info[cls] = {
            "class_id": idx,
            "original_count": count,
            "original_percentage": float(orig_pct),
            "sample_weight": float(class_weights[idx]),
            "expected_sampling_probability_percent": float(sampling_prob),
            "expected_samples_per_epoch": float(total_train / num_classes)
        }
        distribution_info[cls] = {
            "count": count,
            "percentage": float(orig_pct)
        }
        print(f"  Class {idx} [{cls:>5}]: original count = {count:>4} ({orig_pct:>5.2f}%) -> sampling prob = {sampling_prob:>5.2f}% (~{total_train/num_classes:.1f} samples/epoch)")

    print("="*60 + "\n")
    return sampler, sampling_info, distribution_info


# ── Held-Out Test Evaluation ────────────────────────────────────────────────────
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

    mel_idx = CLASS_MAPPING["mel"]
    nv_idx  = CLASS_MAPPING["nv"]
    bkl_idx = CLASS_MAPPING["bkl"]

    mel_total   = int(cm[mel_idx].sum())
    mel_correct = int(cm[mel_idx][mel_idx])
    mel_to_nv   = int(cm[mel_idx][nv_idx])
    mel_to_bkl  = int(cm[mel_idx][bkl_idx])

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
        f.write("EXPERIMENT 4 — CLASS-BALANCED SAMPLING Test Set Evaluation\n")
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
    plt.title('Confusion Matrix — Test Set (Experiment 4: Class-Balanced Sampling)')
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"\nSaved evaluation artifacts -> {results_dir}")
    return metrics


# ── Dual Comparison Helpers ─────────────────────────────────────────────────────
def compute_comparisons(exp4_metrics, results_dir):
    # Comparison with Exp 1 Baseline
    diff1 = {
        "accuracy_change": float(exp4_metrics["accuracy"] - EXP1_BASELINE["accuracy"]),
        "balanced_accuracy_change": float(exp4_metrics["balanced_accuracy"] - EXP1_BASELINE["balanced_accuracy"]),
        "macro_f1_change": float(exp4_metrics["macro_f1"] - EXP1_BASELINE["macro_f1"]),
        "macro_precision_change": float(exp4_metrics["macro_precision"] - EXP1_BASELINE["macro_precision"]),
        "macro_recall_change": float(exp4_metrics["macro_recall"] - EXP1_BASELINE["macro_recall"]),
        "weighted_f1_change": float(exp4_metrics["weighted_f1"] - EXP1_BASELINE["weighted_f1"]),
        "melanoma_recall_change": float(exp4_metrics["melanoma_recall"] - EXP1_BASELINE["melanoma_recall"]),
        "melanoma_precision_change": float(exp4_metrics["melanoma_precision"] - EXP1_BASELINE["melanoma_precision"]),
        "melanoma_f1_change": float(exp4_metrics["melanoma_f1"] - EXP1_BASELINE["melanoma_f1"]),
    }
    comp1 = {
        "experiment1_baseline": EXP1_BASELINE,
        "experiment4_class_balanced_sampling": {
            "accuracy": exp4_metrics["accuracy"],
            "balanced_accuracy": exp4_metrics["balanced_accuracy"],
            "macro_precision": exp4_metrics["macro_precision"],
            "macro_recall": exp4_metrics["macro_recall"],
            "macro_f1": exp4_metrics["macro_f1"],
            "weighted_f1": exp4_metrics["weighted_f1"],
            "melanoma_recall": exp4_metrics["melanoma_recall"],
            "melanoma_precision": exp4_metrics["melanoma_precision"],
            "melanoma_f1": exp4_metrics["melanoma_f1"],
        },
        "delta_exp4_minus_exp1": diff1
    }
    with open(os.path.join(results_dir, "comparison_with_experiment1.json"), "w") as f:
        json.dump(comp1, f, indent=4)

    # Comparison with Exp 2 (Class-Weighted CE)
    diff2 = {
        "accuracy_change": float(exp4_metrics["accuracy"] - EXP2_BASELINE["accuracy"]),
        "balanced_accuracy_change": float(exp4_metrics["balanced_accuracy"] - EXP2_BASELINE["balanced_accuracy"]),
        "macro_f1_change": float(exp4_metrics["macro_f1"] - EXP2_BASELINE["macro_f1"]),
        "macro_precision_change": float(exp4_metrics["macro_precision"] - EXP2_BASELINE["macro_precision"]),
        "macro_recall_change": float(exp4_metrics["macro_recall"] - EXP2_BASELINE["macro_recall"]),
        "weighted_f1_change": float(exp4_metrics["weighted_f1"] - EXP2_BASELINE["weighted_f1"]),
        "melanoma_recall_change": float(exp4_metrics["melanoma_recall"] - EXP2_BASELINE["melanoma_recall"]),
        "melanoma_precision_change": float(exp4_metrics["melanoma_precision"] - EXP2_BASELINE["melanoma_precision"]),
        "melanoma_f1_change": float(exp4_metrics["melanoma_f1"] - EXP2_BASELINE["melanoma_f1"]),
        "mel_to_nv_error_change": int(exp4_metrics["melanoma_to_nv_errors"] - EXP2_BASELINE["mel_to_nv_errors"]),
        "mel_to_bkl_error_change": int(exp4_metrics["melanoma_to_bkl_errors"] - EXP2_BASELINE["mel_to_bkl_errors"]),
    }
    comp2 = {
        "experiment2_class_weighted_ce": EXP2_BASELINE,
        "experiment4_class_balanced_sampling": {
            "accuracy": exp4_metrics["accuracy"],
            "balanced_accuracy": exp4_metrics["balanced_accuracy"],
            "macro_precision": exp4_metrics["macro_precision"],
            "macro_recall": exp4_metrics["macro_recall"],
            "macro_f1": exp4_metrics["macro_f1"],
            "weighted_f1": exp4_metrics["weighted_f1"],
            "melanoma_recall": exp4_metrics["melanoma_recall"],
            "melanoma_precision": exp4_metrics["melanoma_precision"],
            "melanoma_f1": exp4_metrics["melanoma_f1"],
            "mel_to_nv_errors": exp4_metrics["melanoma_to_nv_errors"],
            "mel_to_bkl_errors": exp4_metrics["melanoma_to_bkl_errors"],
        },
        "delta_exp4_minus_exp2": diff2
    }
    with open(os.path.join(results_dir, "comparison_with_experiment2.json"), "w") as f:
        json.dump(comp2, f, indent=4)

    print("\n" + "="*60)
    print("COMPARISON: EXPERIMENT 4 vs EXPERIMENT 1 & EXPERIMENT 2")
    print("="*60)
    print(f"  Test Accuracy     : Exp 1: {EXP1_BASELINE['accuracy']*100:.2f}% | Exp 2: {EXP2_BASELINE['accuracy']*100:.2f}% | Exp 4: {exp4_metrics['accuracy']*100:.2f}%")
    print(f"    vs Exp 1 Delta  : {diff1['accuracy_change']*100:+.2f}%")
    print(f"    vs Exp 2 Delta  : {diff2['accuracy_change']*100:+.2f}%")
    print(f"  Balanced Accuracy : Exp 1: {EXP1_BASELINE['balanced_accuracy']*100:.2f}% | Exp 2: {EXP2_BASELINE['balanced_accuracy']*100:.2f}% | Exp 4: {exp4_metrics['balanced_accuracy']*100:.2f}%")
    print(f"    vs Exp 1 Delta  : {diff1['balanced_accuracy_change']*100:+.2f}%")
    print(f"    vs Exp 2 Delta  : {diff2['balanced_accuracy_change']*100:+.2f}%")
    print(f"  Macro F1          : Exp 1: {EXP1_BASELINE['macro_f1']:.4f} | Exp 2: {EXP2_BASELINE['macro_f1']:.4f} | Exp 4: {exp4_metrics['macro_f1']:.4f}")
    print(f"    vs Exp 1 Delta  : {diff1['macro_f1_change']:+.4f}")
    print(f"    vs Exp 2 Delta  : {diff2['macro_f1_change']:+.4f}")
    print(f"  Melanoma Recall   : Exp 1: {EXP1_BASELINE['melanoma_recall']*100:.2f}% | Exp 2: {EXP2_BASELINE['melanoma_recall']*100:.2f}% | Exp 4: {exp4_metrics['melanoma_recall']*100:.2f}%")
    print(f"    vs Exp 1 Delta  : {diff1['melanoma_recall_change']*100:+.2f}%")
    print(f"    vs Exp 2 Delta  : {diff2['melanoma_recall_change']*100:+.2f}%")
    print(f"  Melanoma F1       : Exp 1: {EXP1_BASELINE['melanoma_f1']:.4f} | Exp 2: {EXP2_BASELINE['melanoma_f1']:.4f} | Exp 4: {exp4_metrics['melanoma_f1']:.4f}")
    print(f"    vs Exp 1 Delta  : {diff1['melanoma_f1_change']:+.4f}")
    print(f"    vs Exp 2 Delta  : {diff2['melanoma_f1_change']:+.4f}")
    print("="*60)
    return comp1, comp2


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

    # 2. Setup Class-Balanced Sampler STRICTLY from training set only
    sampler, sampling_info, distribution_info = setup_class_balanced_sampler(train_df)

    # Save sampling configuration & original training distribution
    with open(os.path.join(RESULTS_DIR, "sampling_configuration.json"), "w") as f:
        json.dump({
            "sampler_type": "torch.utils.data.WeightedRandomSampler",
            "replacement": True,
            "samples_per_epoch": len(train_df),
            "batches_per_epoch": int(np.ceil(len(train_df) / BATCH_SIZE)),
            "per_class_strategy": "1.0 / N_class inverse frequency sampling weights",
            "target_distribution": "Uniform (14.2857% expected frequency per class)",
            "classes": sampling_info
        }, f, indent=4)

    with open(os.path.join(RESULTS_DIR, "training_class_distribution.json"), "w") as f:
        json.dump(distribution_info, f, indent=4)

    # 3. LOSS: Strictly UNWEIGHTED Standard Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()
    print("Loss function: Standard Unweighted CrossEntropyLoss() [Isolating sampling effect]")

    # 4. Datasets and Loaders
    train_dataset = HAM10000Dataset(train_df, transform=get_transforms(is_train=True))
    val_dataset   = HAM10000Dataset(val_df,   transform=get_transforms(is_train=False))

    # TRAINING LOADER uses the WeightedRandomSampler (shuffle must be False when sampler is provided)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=0, pin_memory=False)
    # Validation and Test loaders remain unweighted, sequential, and fully representative
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=False)

    print(f"Diagnostic -> Device: {device}, num_workers: 0, pin_memory: False, batch_size: {BATCH_SIZE}")

    # Smoke Test
    print("\n[SMOKE TEST] Fetching one batch from train_loader (Class-Balanced) ...")
    try:
        smoke_imgs, smoke_labels = next(iter(train_loader))
        print(f"[SMOKE TEST] Image tensor shape : {smoke_imgs.shape}")
        print(f"[SMOKE TEST] Label tensor shape : {smoke_labels.shape}")
        print(f"[SMOKE TEST] Batch label distribution : {pd.Series(smoke_labels.numpy()).value_counts().to_dict()}")
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

    print(f"\nStarting Experiment 4 training for up to {EPOCHS} epochs using Class-Balanced Sampling...")
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
    axes[0].set_title('Loss (Standard CE with Balanced Sampling)'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs_ran, history['val_acc'], color='green', label='Val Accuracy')
    axes[1].set_title('Val Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True)

    axes[2].plot(epochs_ran, history['lr'], color='orange', label='LR')
    axes[2].set_title('Learning Rate'); axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log'); axes[2].legend(); axes[2].grid(True)

    plt.suptitle('Experiment 4 — Class-Balanced Sampling Training Curves', fontsize=14)
    plt.tight_layout()
    curves_path = os.path.join(RESULTS_DIR, "training_curves.png")
    plt.savefig(curves_path, dpi=150)
    plt.close()
    print(f"  Training curves -> {curves_path}")

    # 9. Single Held-Out Test Evaluation
    metrics = run_test_evaluation(model, device, test_df, RESULTS_DIR, checkpoint_path)

    # 10. Comparisons against Experiment 1 & Experiment 2
    comp1, comp2 = compute_comparisons(metrics, RESULTS_DIR)

    # 11. Experiment Summary
    summary = {
        "experiment": EXPERIMENT_NAME,
        "config": {
            "epochs_configured": EPOCHS,
            "epochs_ran": len(epochs_ran),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "loss": "standard_unweighted_cross_entropy",
            "sampler": "WeightedRandomSampler",
            "sampling_weights_source": "training_set_only",
            "sampling_info": sampling_info,
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
        "comparison_with_experiment1": comp1,
        "comparison_with_experiment2": comp2
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n{'='*60}")
    print("EXPERIMENT 4 COMPLETE")
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
    print("\nExperiment 4 finished. STOPPED — awaiting next instruction.\n")


if __name__ == "__main__":
    train()
