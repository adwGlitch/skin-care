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
RESULTS_DIR   = os.path.join(AI_DIR, "results", "experiment14")

CHECKPOINT_EXP8  = os.path.join(MODELS_DIR, "experiment8_best_model.pt")
CHECKPOINT_EXP13 = os.path.join(MODELS_DIR, "experiment13_best_model.pt")

VAL_CSV_PATH  = os.path.join(SPLITS_DIR, "val.csv")
TEST_CSV_PATH = os.path.join(SPLITS_DIR, "test.csv")

LAMBDA_GRID = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]
BATCH_SIZE  = 32

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MEL_IDX     = CLASS_MAPPING["mel"]
NV_IDX      = CLASS_MAPPING["nv"]

os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Step 1: Model Loading & Freezing ───────────────────────────────────────────
def load_and_verify_models(device):
    print("="*70)
    print("STEP 1: LOAD & VERIFY EXP8 AND EXP13 CHECKPOINTS (FROZEN EVAL MODE)")
    print("="*70)
    
    assert os.path.exists(CHECKPOINT_EXP8), f"Exp8 checkpoint missing: {CHECKPOINT_EXP8}"
    assert os.path.exists(CHECKPOINT_EXP13), f"Exp13 checkpoint missing: {CHECKPOINT_EXP13}"
    
    model8 = DermaAI_MobileNetV3(num_classes=7)
    model8.load_state_dict(torch.load(CHECKPOINT_EXP8, map_location=device))
    model8 = model8.to(device)
    model8.eval()
    for p in model8.parameters():
        p.requires_grad = False
    
    model13 = DermaAI_MobileNetV3(num_classes=7)
    model13.load_state_dict(torch.load(CHECKPOINT_EXP13, map_location=device))
    model13 = model13.to(device)
    model13.eval()
    for p in model13.parameters():
        p.requires_grad = False
        
    p8 = sum(p.numel() for p in model8.parameters())
    p13 = sum(p.numel() for p in model13.parameters())
    
    print(f"  Model A (Exp8)  : DermaAI_MobileNetV3 | Params: {p8:,} | Requires Grad: False")
    print(f"  Model B (Exp13) : DermaAI_MobileNetV3 | Params: {p13:,} | Requires Grad: False")
    assert p8 == 4326297 and p13 == 4326297, "Parameter count mismatch! Both models must have 4,326,297 parameters."
    
    # Pre-inference dummy forward check
    dummy = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        out8 = model8(dummy)
        out13 = model13(dummy)
        prob8 = torch.softmax(out8, dim=1).cpu().numpy()
        prob13 = torch.softmax(out13, dim=1).cpu().numpy()
        
    assert out8.shape == (2, 7) and out13.shape == (2, 7)
    assert np.allclose(prob8.sum(axis=1), 1.0, atol=1e-5), "Model 8 softmax must sum to 1"
    assert np.allclose(prob13.sum(axis=1), 1.0, atol=1e-5), "Model 13 softmax must sum to 1"
    print("  Forward pass and softmax checks: PASSED")
    print("="*70 + "\n")
    return model8, model13


# ── Step 2: Probability Extraction ─────────────────────────────────────────────
def extract_probabilities(model, dataloader, device, desc="Inference"):
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc, leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
            
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Assert floating-point probability normalization
    assert np.allclose(all_probs.sum(axis=1), 1.0, atol=1e-5), f"{desc} probabilities must sum to 1.0"
    return all_probs, all_labels


# ── Helper to Compute Comprehensive Metrics ────────────────────────────────────
def compute_metrics(y_true, y_pred, probs=None):
    acc = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(7)))
    
    per_class_p, per_class_r, per_class_f1, per_class_supp = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(7)), zero_division=0
    )
    
    # Melanoma specific
    mel_recall = float(per_class_r[MEL_IDX])
    mel_precision = float(per_class_p[MEL_IDX])
    mel_f1 = float(per_class_f1[MEL_IDX])
    mel_support = int(per_class_supp[MEL_IDX])
    mel_tp = int(cm[MEL_IDX, MEL_IDX])
    mel_fp = int(cm[:, MEL_IDX].sum() - mel_tp)
    mel_pred_total = int(cm[:, MEL_IDX].sum())
    
    mel_to_nv = int(cm[MEL_IDX, NV_IDX])
    mel_to_bkl = int(cm[MEL_IDX, CLASS_MAPPING["bkl"]])
    mel_to_akiec = int(cm[MEL_IDX, CLASS_MAPPING["akiec"]])
    mel_to_bcc = int(cm[MEL_IDX, CLASS_MAPPING["bcc"]])
    
    # Nevus specific
    nv_recall = float(per_class_r[NV_IDX])
    nv_precision = float(per_class_p[NV_IDX])
    nv_f1 = float(per_class_f1[NV_IDX])
    nv_support = int(per_class_supp[NV_IDX])
    nv_tp = int(cm[NV_IDX, NV_IDX])
    nv_to_mel = int(cm[NV_IDX, MEL_IDX])
    
    per_class_dict = {}
    for i, cname in enumerate(CLASS_NAMES):
        per_class_dict[cname] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1-score": float(per_class_f1[i]),
            "support": int(per_class_supp[i]),
            "correct": int(cm[i, i])
        }
        
    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bacc),
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(prec_weighted),
        "weighted_recall": float(rec_weighted),
        "weighted_f1": float(f1_weighted),
        "melanoma_precision": mel_precision,
        "melanoma_recall": mel_recall,
        "melanoma_f1": mel_f1,
        "melanoma_correct_images": mel_tp,
        "melanoma_total_images": mel_support,
        "melanoma_predicted_total": mel_pred_total,
        "melanoma_false_positives": mel_fp,
        "melanoma_to_nv_errors": mel_to_nv,
        "melanoma_to_bkl_errors": mel_to_bkl,
        "melanoma_to_akiec_errors": mel_to_akiec,
        "melanoma_to_bcc_errors": mel_to_bcc,
        "nv_precision": nv_precision,
        "nv_recall": nv_recall,
        "nv_f1": nv_f1,
        "nv_correct_images": nv_tp,
        "nv_total_images": nv_support,
        "nv_to_mel_errors": nv_to_mel,
        "bcc_recall": float(per_class_r[CLASS_MAPPING["bcc"]]),
        "bkl_recall": float(per_class_r[CLASS_MAPPING["bkl"]]),
        "per_class": per_class_dict,
        "confusion_matrix": cm.tolist()
    }


# ── Plot & Save Confusion Matrix ───────────────────────────────────────────────
def save_confusion_matrix_plot(cm, save_path, title):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, cbar=True, annot_kws={"size": 11}
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Ground-Truth Class", fontsize=11, fontweight="bold", labelpad=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# ── Main Experiment 14 Routine ─────────────────────────────────────────────────
def main():
    start_total_time = time.time()
    device = torch.device("cpu")
    print("\n" + "="*70)
    print("EXPERIMENT 14: VALIDATION-CALIBRATED PROBABILITY ENSEMBLE (EXP8 + EXP13)")
    print(f"Device : {device} | PyTorch: {torch.__version__} | torchvision: {torchvision.__version__}")
    print("="*70)
    
    # Preprocessing
    eval_transform = get_transforms(is_train=False)
    
    # ── ZERO TRAINING SET LOADING ─────────────────────────────────────────────
    # Load ONLY validation and test datasets
    assert os.path.exists(VAL_CSV_PATH), f"Val split missing: {VAL_CSV_PATH}"
    assert os.path.exists(TEST_CSV_PATH), f"Test split missing: {TEST_CSV_PATH}"
    
    val_df  = pd.read_csv(VAL_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    
    print(f"Splits loaded: Validation = {len(val_df)} images | Test = {len(test_df)} images")
    print("TRAINING SET WAS NOT LOADED (strictly maintaining leakage isolation).")
    
    val_dataset  = HAM10000Dataset(val_df, transform=eval_transform)
    test_dataset = HAM10000Dataset(test_df, transform=eval_transform)
    
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Load and freeze models
    model8, model13 = load_and_verify_models(device)
    
    # ── VALIDATION INFERENCE ───────────────────────────────────────────────────
    print("="*70)
    print("STEP 2: VALIDATION PROBABILITY EXTRACTION (N=1,010)")
    print("="*70)
    val_start_time = time.time()
    P8_val, y_val = extract_probabilities(model8, val_loader, device, desc="Val Exp8")
    P13_val, y_val_check = extract_probabilities(model13, val_loader, device, desc="Val Exp13")
    val_inference_duration = time.time() - val_start_time
    assert np.array_equal(y_val, y_val_check), "Validation label alignment error!"
    print(f"Validation inference complete in {val_inference_duration:.2f}s ({val_inference_duration/len(y_val)*1000:.2f} ms/img)")
    
    # ── 21-POINT VALIDATION SWEEP ──────────────────────────────────────────────
    print("\n" + "="*70)
    print("STEP 3: 21-POINT VALIDATION LAMBDA SWEEP & SANITY CHECKS")
    print("="*70)
    sweep_results = []
    val_metrics_per_lambda = {}
    
    for lmbda in LAMBDA_GRID:
        P_ens_val = (1.0 - lmbda) * P8_val + lmbda * P13_val
        assert np.allclose(P_ens_val.sum(axis=1), 1.0, atol=1e-5), f"P_ens sum != 1.0 at lambda={lmbda}"
        preds_val = np.argmax(P_ens_val, axis=1)
        
        # Sanity Checks
        if lmbda == 0.00:
            assert np.array_equal(preds_val, np.argmax(P8_val, axis=1)), "Sanity check FAILED: lambda=0.00 != Exp8"
            print("  Sanity check lambda=0.00 exactly reproduces Exp8: PASSED")
        elif lmbda == 1.00:
            assert np.array_equal(preds_val, np.argmax(P13_val, axis=1)), "Sanity check FAILED: lambda=1.00 != Exp13"
            print("  Sanity check lambda=1.00 exactly reproduces Exp13: PASSED")
            
        m = compute_metrics(y_val, preds_val)
        val_metrics_per_lambda[lmbda] = m
        
        sweep_results.append({
            "lambda": float(lmbda),
            "weight_exp8": float(round(1.0 - lmbda, 2)),
            "weight_exp13": float(lmbda),
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "melanoma_precision": m["melanoma_precision"],
            "melanoma_recall": m["melanoma_recall"],
            "melanoma_f1": m["melanoma_f1"],
            "melanoma_to_nv_errors": m["melanoma_to_nv_errors"],
            "melanoma_to_bkl_errors": m["melanoma_to_bkl_errors"],
            "nv_recall": m["nv_recall"],
            "nv_to_mel_errors": m["nv_to_mel_errors"]
        })
        
    sweep_df = pd.DataFrame(sweep_results)
    sweep_df.to_csv(os.path.join(RESULTS_DIR, "ensemble_weight_sweep_validation.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "ensemble_weight_sweep_validation.json"), "w") as f:
        json.dump(sweep_results, f, indent=4)
    print(f"Validation sweep saved: {os.path.join(RESULTS_DIR, 'ensemble_weight_sweep_validation.csv')}")
    
    # ── MATHEMATICALLY EXPLICIT SELECTION RULE ─────────────────────────────────
    print("\n" + "="*70)
    print("STEP 4: SELECTION RULE APPLICATION (VALIDATION ONLY)")
    print("="*70)
    f1_max = float(sweep_df["macro_f1"].max())
    print(f"  Maximum Validation Macro F1 (F1_max): {f1_max:.4f}")
    
    # Candidate thresholds: F1_max - MacroF1(lambda) <= 0.005
    candidate_mask = (f1_max - sweep_df["macro_f1"]) <= 0.005
    candidate_df = sweep_df[candidate_mask].copy()
    candidate_lambdas = candidate_df["lambda"].tolist()
    print(f"  Candidate lambdas within 0.005 tolerance ({len(candidate_lambdas)}): {candidate_lambdas}")
    
    # Criterion 1: Highest Validation Melanoma Recall
    max_mel_recall = candidate_df["melanoma_recall"].max()
    tied_mel_recall_df = candidate_df[candidate_df["melanoma_recall"] == max_mel_recall]
    print(f"  Max Melanoma Recall among candidates: {max_mel_recall*100:.2f}% (retained {len(tied_mel_recall_df)} candidates)")
    
    selection_tie_breaker = "Primary (Max Macro F1)"
    if len(tied_mel_recall_df) == 1:
        selected_row = tied_mel_recall_df.iloc[0]
        selection_tie_breaker = "Tie-break 1 (Higher Validation Melanoma Recall)"
    else:
        # Criterion 2: Highest Validation Balanced Accuracy
        max_bacc = tied_mel_recall_df["balanced_accuracy"].max()
        tied_bacc_df = tied_mel_recall_df[tied_mel_recall_df["balanced_accuracy"] == max_bacc]
        print(f"  Max Balanced Accuracy among tied candidates: {max_bacc*100:.2f}% (retained {len(tied_bacc_df)} candidates)")
        if len(tied_bacc_df) == 1:
            selected_row = tied_bacc_df.iloc[0]
            selection_tie_breaker = "Tie-break 2 (Higher Validation Balanced Accuracy)"
        else:
            # Criterion 3: Smallest lambda
            min_lmbda = tied_bacc_df["lambda"].min()
            selected_row = tied_bacc_df[tied_bacc_df["lambda"] == min_lmbda].iloc[0]
            selection_tie_breaker = "Tie-break 3 (Smallest lambda)"
            
    selected_lambda = float(selected_row["lambda"])
    weight_exp8 = float(selected_row["weight_exp8"])
    weight_exp13 = float(selected_row["weight_exp13"])
    
    print("\n" + "#"*70)
    print(f"FINAL SELECTED LAMBDA (VAL ONLY): lambda* = {selected_lambda:.2f}")
    print(f"  Exp8 Weight  : {weight_exp8:.2f}")
    print(f"  Exp13 Weight : {weight_exp13:.2f}")
    print(f"  Val Macro F1 : {selected_row['macro_f1']:.4f}")
    print(f"  Val Mel Rec  : {selected_row['melanoma_recall']*100:.2f}%")
    print(f"  Val Bal Acc  : {selected_row['balanced_accuracy']*100:.2f}%")
    print(f"  Selection Logic: {selection_tie_breaker}")
    print("#"*70 + "\n")
    
    # Save selected_lambda.json
    selected_lambda_data = {
        "selected_lambda": selected_lambda,
        "weight_exp8": weight_exp8,
        "weight_exp13": weight_exp13,
        "validation_macro_f1": float(selected_row["macro_f1"]),
        "validation_melanoma_recall": float(selected_row["melanoma_recall"]),
        "validation_balanced_accuracy": float(selected_row["balanced_accuracy"]),
        "f1_max": f1_max,
        "tolerance": 0.005,
        "candidate_lambdas": candidate_lambdas,
        "selection_rule": "F1_max - MacroF1 <= 0.005 -> Max Melanoma Recall -> Max Balanced Accuracy -> Min Lambda",
        "selection_tie_breaker": selection_tie_breaker,
        "status": "FROZEN (Pre-Test Evaluation)"
    }
    with open(os.path.join(RESULTS_DIR, "selected_lambda.json"), "w") as f:
        json.dump(selected_lambda_data, f, indent=4)
        
    # Save validation_metrics.json
    val_selected_metrics = val_metrics_per_lambda[selected_lambda]
    with open(os.path.join(RESULTS_DIR, "validation_metrics.json"), "w") as f:
        json.dump(val_selected_metrics, f, indent=4)
        
    print("="*70)
    print("LEAKAGE BOUNDARY VERIFICATION: LAMBDA IS NOW COMPLETELY FROZEN.")
    print("No further validation tuning will occur.")
    print("Evaluating held-out test set EXACTLY ONCE.")
    print("="*70 + "\n")
    
    # ── TEST INFERENCE ONCE ────────────────────────────────────────────────────
    test_start_time = time.time()
    P8_test, y_test = extract_probabilities(model8, test_loader, device, desc="Test Exp8")
    P13_test, y_test_check = extract_probabilities(model13, test_loader, device, desc="Test Exp13")
    test_inference_duration = time.time() - test_start_time
    assert np.array_equal(y_test, y_test_check), "Test label alignment error!"
    
    # Probability-level blending
    P_ens_test = (1.0 - selected_lambda) * P8_test + selected_lambda * P13_test
    assert np.allclose(P_ens_test.sum(axis=1), 1.0, atol=1e-5), "Test P_ens sum != 1.0"
    
    pred_exp8 = np.argmax(P8_test, axis=1)
    pred_exp13 = np.argmax(P13_test, axis=1)
    pred_ensemble = np.argmax(P_ens_test, axis=1)
    
    # Sanity checks on test set
    if selected_lambda == 0.0:
        assert np.array_equal(pred_ensemble, pred_exp8), "Sanity check FAILED on test: lambda=0 != Exp8"
    elif selected_lambda == 1.0:
        assert np.array_equal(pred_ensemble, pred_exp13), "Sanity check FAILED on test: lambda=1 != Exp13"
        
    # Compute Full Metrics for All Three
    m_exp8 = compute_metrics(y_test, pred_exp8)
    m_exp13 = compute_metrics(y_test, pred_exp13)
    m_ens = compute_metrics(y_test, pred_ensemble)
    
    # Save test metrics JSONs
    with open(os.path.join(RESULTS_DIR, "test_metrics_exp8.json"), "w") as f:
        json.dump(m_exp8, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "test_metrics_exp13.json"), "w") as f:
        json.dump(m_exp13, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "test_metrics_ensemble.json"), "w") as f:
        json.dump(m_ens, f, indent=4)
        
    # Classification reports TXT
    rep_exp8 = classification_report(y_test, pred_exp8, target_names=CLASS_NAMES, digits=4, zero_division=0)
    rep_exp13 = classification_report(y_test, pred_exp13, target_names=CLASS_NAMES, digits=4, zero_division=0)
    rep_ens = classification_report(y_test, pred_ensemble, target_names=CLASS_NAMES, digits=4, zero_division=0)
    
    with open(os.path.join(RESULTS_DIR, "classification_report_exp8.txt"), "w") as f:
        f.write("EXPERIMENT 8 STANDALONE ON TEST SET (N=988):\n\n" + rep_exp8)
    with open(os.path.join(RESULTS_DIR, "classification_report_exp13.txt"), "w") as f:
        f.write("EXPERIMENT 13 STANDALONE ON TEST SET (N=988):\n\n" + rep_exp13)
    with open(os.path.join(RESULTS_DIR, "classification_report_ensemble.txt"), "w") as f:
        f.write(f"EXPERIMENT 14 PROBABILITY ENSEMBLE (lambda={selected_lambda:.2f}) ON TEST SET (N=988):\n\n" + rep_ens)
        
    # Confusion Matrices JSON & PNG
    cm_exp8 = np.array(m_exp8["confusion_matrix"])
    cm_exp13 = np.array(m_exp13["confusion_matrix"])
    cm_ens = np.array(m_ens["confusion_matrix"])
    
    with open(os.path.join(RESULTS_DIR, "confusion_matrix_exp8.json"), "w") as f:
        json.dump({"classes": CLASS_NAMES, "confusion_matrix": cm_exp8.tolist()}, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "confusion_matrix_exp13.json"), "w") as f:
        json.dump({"classes": CLASS_NAMES, "confusion_matrix": cm_exp13.tolist()}, f, indent=4)
    with open(os.path.join(RESULTS_DIR, "confusion_matrix_ensemble.json"), "w") as f:
        json.dump({"classes": CLASS_NAMES, "confusion_matrix": cm_ens.tolist()}, f, indent=4)
        
    save_confusion_matrix_plot(cm_exp8, os.path.join(RESULTS_DIR, "confusion_matrix_exp8.png"),
                               "Experiment 8: Standalone Test Confusion Matrix")
    save_confusion_matrix_plot(cm_exp13, os.path.join(RESULTS_DIR, "confusion_matrix_exp13.png"),
                               "Experiment 13: Standalone Test Confusion Matrix")
    save_confusion_matrix_plot(cm_ens, os.path.join(RESULTS_DIR, "confusion_matrix_ensemble.png"),
                               f"Experiment 14: Probability Ensemble (λ={selected_lambda:.2f}) Confusion Matrix")

    # ── PREDICTION COMPLEMENTARITY ANALYSIS ────────────────────────────────────
    print("="*70)
    print("STEP 5: PREDICTION COMPLEMENTARITY & DISAGREEMENT ANALYSIS (TEST N=988)")
    print("="*70)
    
    N = len(y_test)
    correct_exp8 = (pred_exp8 == y_test)
    correct_exp13 = (pred_exp13 == y_test)
    correct_ens = (pred_ensemble == y_test)
    
    disagreement_mask = (pred_exp8 != pred_exp13)
    disagreement_count = int(disagreement_mask.sum())
    disagreement_rate = float(disagreement_count / N)
    
    # 4 mutually exclusive subsets:
    # A: Both correct
    # B: Exp8 correct, Exp13 wrong
    # C: Exp8 wrong, Exp13 correct
    # D: Both wrong
    mask_A = correct_exp8 & correct_exp13
    mask_B = correct_exp8 & (~correct_exp13)
    mask_C = (~correct_exp8) & correct_exp13
    mask_D = (~correct_exp8) & (~correct_exp13)
    
    count_A = int(mask_A.sum())
    count_B = int(mask_B.sum())
    count_C = int(mask_C.sum())
    count_D = int(mask_D.sum())
    assert count_A + count_B + count_C + count_D == N, "Subset sum != total test size!"
    
    # Ensemble behavior on each subset
    ens_on_A_correct = int((correct_ens & mask_A).sum())
    ens_on_A_corrupted = count_A - ens_on_A_correct
    
    ens_on_B_correct = int((correct_ens & mask_B).sum())
    ens_on_B_lost = count_B - ens_on_B_correct
    
    ens_on_C_correct = int((correct_ens & mask_C).sum())
    ens_on_C_lost = count_C - ens_on_C_correct
    
    ens_on_D_correct = int((correct_ens & mask_D).sum())
    ens_on_D_wrong = count_D - ens_on_D_correct
    
    # Net correct vs Exp8 and Exp13
    total_correct_exp8 = int(correct_exp8.sum())
    total_correct_exp13 = int(correct_exp13.sum())
    total_correct_ens = int(correct_ens.sum())
    
    # Melanoma specific complementarity
    mel_mask = (y_test == MEL_IDX)
    mel_N = int(mel_mask.sum())
    mel_correct_exp8 = (pred_exp8 == MEL_IDX) & mel_mask
    mel_correct_exp13 = (pred_exp13 == MEL_IDX) & mel_mask
    mel_correct_ens = (pred_ensemble == MEL_IDX) & mel_mask
    
    mel_mask_A = mel_correct_exp8 & mel_correct_exp13
    mel_mask_B = mel_correct_exp8 & (~mel_correct_exp13)
    mel_mask_C = (~mel_correct_exp8) & mel_correct_exp13
    mel_mask_D = (~mel_correct_exp8) & (~mel_correct_exp13)
    
    mel_count_A = int(mel_mask_A.sum())
    mel_count_B = int(mel_mask_B.sum())
    mel_count_C = int(mel_mask_C.sum())
    mel_count_D = int(mel_mask_D.sum())
    
    mel_ens_on_A = int((mel_correct_ens & mel_mask_A).sum())
    mel_ens_on_B = int((mel_correct_ens & mel_mask_B).sum())
    mel_ens_on_C = int((mel_correct_ens & mel_mask_C).sum())
    mel_ens_on_D = int((mel_correct_ens & mel_mask_D).sum())
    
    complementarity_data = {
        "test_total_images": N,
        "disagreement_count": disagreement_count,
        "disagreement_rate": disagreement_rate,
        "subset_breakdown": {
            "A_both_correct": {
                "count": count_A,
                "percentage": float(count_A / N * 100),
                "ensemble_retained_correct": ens_on_A_correct,
                "ensemble_corrupted_to_wrong": ens_on_A_corrupted
            },
            "B_exp8_only_correct": {
                "count": count_B,
                "percentage": float(count_B / N * 100),
                "ensemble_retained_correct": ens_on_B_correct,
                "ensemble_lost_to_wrong": ens_on_B_lost
            },
            "C_exp13_only_correct": {
                "count": count_C,
                "percentage": float(count_C / N * 100),
                "ensemble_captured_correct": ens_on_C_correct,
                "ensemble_lost_to_wrong": ens_on_C_lost
            },
            "D_both_wrong": {
                "count": count_D,
                "percentage": float(count_D / N * 100),
                "ensemble_resolved_correct": ens_on_D_correct,
                "ensemble_remained_wrong": ens_on_D_wrong
            }
        },
        "overall_correct_counts": {
            "exp8_correct": total_correct_exp8,
            "exp13_correct": total_correct_exp13,
            "ensemble_correct": total_correct_ens,
            "net_gain_vs_exp8": total_correct_ens - total_correct_exp8,
            "net_gain_vs_exp13": total_correct_ens - total_correct_exp13
        },
        "melanoma_complementarity": {
            "total_melanoma_images": mel_N,
            "A_both_detected": {
                "count": mel_count_A,
                "ensemble_retained": mel_ens_on_A
            },
            "B_exp8_only_detected": {
                "count": mel_count_B,
                "ensemble_retained": mel_ens_on_B
            },
            "C_exp13_only_detected": {
                "count": mel_count_C,
                "ensemble_retained": mel_ens_on_C
            },
            "D_neither_detected": {
                "count": mel_count_D,
                "ensemble_detected": mel_ens_on_D
            },
            "total_detected_exp8": int(mel_correct_exp8.sum()),
            "total_detected_exp13": int(mel_correct_exp13.sum()),
            "total_detected_ensemble": int(mel_correct_ens.sum())
        }
    }
    with open(os.path.join(RESULTS_DIR, "prediction_complementarity.json"), "w") as f:
        json.dump(complementarity_data, f, indent=4)

    # ── Comparisons with Previous Experiments ──────────────────────────────────
    # Exp 8 Comparison
    comp_exp8 = {
        "experiment8_baseline": {
            "accuracy": m_exp8["accuracy"],
            "balanced_accuracy": m_exp8["balanced_accuracy"],
            "macro_f1": m_exp8["macro_f1"],
            "weighted_f1": m_exp8["weighted_f1"],
            "melanoma_recall": m_exp8["melanoma_recall"],
            "melanoma_precision": m_exp8["melanoma_precision"],
            "melanoma_f1": m_exp8["melanoma_f1"],
            "mel_to_nv": m_exp8["melanoma_to_nv_errors"],
            "mel_to_bkl": m_exp8["melanoma_to_bkl_errors"],
            "nv_to_mel": m_exp8["nv_to_mel_errors"]
        },
        "experiment14_ensemble": {
            "accuracy": m_ens["accuracy"],
            "balanced_accuracy": m_ens["balanced_accuracy"],
            "macro_f1": m_ens["macro_f1"],
            "weighted_f1": m_ens["weighted_f1"],
            "melanoma_recall": m_ens["melanoma_recall"],
            "melanoma_precision": m_ens["melanoma_precision"],
            "melanoma_f1": m_ens["melanoma_f1"],
            "mel_to_nv": m_ens["melanoma_to_nv_errors"],
            "mel_to_bkl": m_ens["melanoma_to_bkl_errors"],
            "nv_to_mel": m_ens["nv_to_mel_errors"]
        },
        "delta_ensemble_minus_exp8": {
            "accuracy_change": m_ens["accuracy"] - m_exp8["accuracy"],
            "balanced_accuracy_change": m_ens["balanced_accuracy"] - m_exp8["balanced_accuracy"],
            "macro_f1_change": m_ens["macro_f1"] - m_exp8["macro_f1"],
            "weighted_f1_change": m_ens["weighted_f1"] - m_exp8["weighted_f1"],
            "melanoma_recall_change": m_ens["melanoma_recall"] - m_exp8["melanoma_recall"],
            "melanoma_precision_change": m_ens["melanoma_precision"] - m_exp8["melanoma_precision"],
            "melanoma_f1_change": m_ens["melanoma_f1"] - m_exp8["melanoma_f1"],
            "mel_to_nv_change": m_ens["melanoma_to_nv_errors"] - m_exp8["melanoma_to_nv_errors"],
            "mel_to_bkl_change": m_ens["melanoma_to_bkl_errors"] - m_exp8["melanoma_to_bkl_errors"],
            "nv_to_mel_change": m_ens["nv_to_mel_errors"] - m_exp8["nv_to_mel_errors"]
        }
    }
    with open(os.path.join(RESULTS_DIR, "comparison_with_experiment8.json"), "w") as f:
        json.dump(comp_exp8, f, indent=4)

    # Exp 13 Comparison
    comp_exp13 = {
        "experiment13_baseline": {
            "accuracy": m_exp13["accuracy"],
            "balanced_accuracy": m_exp13["balanced_accuracy"],
            "macro_f1": m_exp13["macro_f1"],
            "weighted_f1": m_exp13["weighted_f1"],
            "melanoma_recall": m_exp13["melanoma_recall"],
            "melanoma_precision": m_exp13["melanoma_precision"],
            "melanoma_f1": m_exp13["melanoma_f1"],
            "mel_to_nv": m_exp13["melanoma_to_nv_errors"],
            "mel_to_bkl": m_exp13["melanoma_to_bkl_errors"],
            "nv_to_mel": m_exp13["nv_to_mel_errors"]
        },
        "experiment14_ensemble": {
            "accuracy": m_ens["accuracy"],
            "balanced_accuracy": m_ens["balanced_accuracy"],
            "macro_f1": m_ens["macro_f1"],
            "weighted_f1": m_ens["weighted_f1"],
            "melanoma_recall": m_ens["melanoma_recall"],
            "melanoma_precision": m_ens["melanoma_precision"],
            "melanoma_f1": m_ens["melanoma_f1"],
            "mel_to_nv": m_ens["melanoma_to_nv_errors"],
            "mel_to_bkl": m_ens["melanoma_to_bkl_errors"],
            "nv_to_mel": m_ens["nv_to_mel_errors"]
        },
        "delta_ensemble_minus_exp13": {
            "accuracy_change": m_ens["accuracy"] - m_exp13["accuracy"],
            "balanced_accuracy_change": m_ens["balanced_accuracy"] - m_exp13["balanced_accuracy"],
            "macro_f1_change": m_ens["macro_f1"] - m_exp13["macro_f1"],
            "weighted_f1_change": m_ens["weighted_f1"] - m_exp13["weighted_f1"],
            "melanoma_recall_change": m_ens["melanoma_recall"] - m_exp13["melanoma_recall"],
            "melanoma_precision_change": m_ens["melanoma_precision"] - m_exp13["melanoma_precision"],
            "melanoma_f1_change": m_ens["melanoma_f1"] - m_exp13["melanoma_f1"],
            "mel_to_nv_change": m_ens["melanoma_to_nv_errors"] - m_exp13["melanoma_to_nv_errors"],
            "mel_to_bkl_change": m_ens["melanoma_to_bkl_errors"] - m_exp13["melanoma_to_bkl_errors"],
            "nv_to_mel_change": m_ens["nv_to_mel_errors"] - m_exp13["nv_to_mel_errors"]
        }
    }
    with open(os.path.join(RESULTS_DIR, "comparison_with_experiment13.json"), "w") as f:
        json.dump(comp_exp13, f, indent=4)

    # Exp 9 Comparison
    exp9_metrics_path = os.path.join(AI_DIR, "results", "experiment9", "metrics.json")
    if os.path.exists(exp9_metrics_path):
        with open(exp9_metrics_path, "r") as f:
            exp9_data = json.load(f)
        comp_exp9 = {
            "experiment9_baseline": {
                "accuracy": exp9_data.get("accuracy"),
                "balanced_accuracy": exp9_data.get("balanced_accuracy"),
                "macro_f1": exp9_data.get("macro_f1"),
                "weighted_f1": exp9_data.get("weighted_f1"),
                "melanoma_recall": exp9_data.get("melanoma_recall"),
                "melanoma_precision": exp9_data.get("melanoma_precision"),
                "melanoma_f1": exp9_data.get("melanoma_f1")
            },
            "experiment14_ensemble": {
                "accuracy": m_ens["accuracy"],
                "balanced_accuracy": m_ens["balanced_accuracy"],
                "macro_f1": m_ens["macro_f1"],
                "weighted_f1": m_ens["weighted_f1"],
                "melanoma_recall": m_ens["melanoma_recall"],
                "melanoma_precision": m_ens["melanoma_precision"],
                "melanoma_f1": m_ens["melanoma_f1"]
            },
            "delta_ensemble_minus_exp9": {
                "accuracy_change": m_ens["accuracy"] - exp9_data.get("accuracy", 0),
                "balanced_accuracy_change": m_ens["balanced_accuracy"] - exp9_data.get("balanced_accuracy", 0),
                "macro_f1_change": m_ens["macro_f1"] - exp9_data.get("macro_f1", 0),
                "weighted_f1_change": m_ens["weighted_f1"] - exp9_data.get("weighted_f1", 0),
                "melanoma_recall_change": m_ens["melanoma_recall"] - exp9_data.get("melanoma_recall", 0),
                "melanoma_precision_change": m_ens["melanoma_precision"] - exp9_data.get("melanoma_precision", 0),
                "melanoma_f1_change": m_ens["melanoma_f1"] - exp9_data.get("melanoma_f1", 0)
            }
        }
        with open(os.path.join(RESULTS_DIR, "comparison_with_experiment9.json"), "w") as f:
            json.dump(comp_exp9, f, indent=4)

    # ── Computational Cost & Ensemble Config ───────────────────────────────────
    total_duration = time.time() - start_total_time
    total_inference_images = len(y_val) + len(y_test)
    avg_inference_time_per_image = (val_inference_duration + test_inference_duration) / total_inference_images
    
    cost_data = {
        "validation_inference_duration_seconds": val_inference_duration,
        "test_inference_duration_seconds": test_inference_duration,
        "total_inference_duration_seconds": val_inference_duration + test_inference_duration,
        "total_experiment_duration_seconds": total_duration,
        "validation_images_evaluated": len(y_val),
        "test_images_evaluated": len(y_test),
        "average_inference_time_per_image_seconds": avg_inference_time_per_image,
        "retraining_time_seconds": 0.0,
        "notes": "NO RETRAINING WAS PERFORMED. Dual forward pass with probability-level weighted averaging."
    }
    
    ensemble_config = {
        "model_A": {
            "name": "Experiment 8 (Tempered Class Weights alpha=0.75)",
            "checkpoint": CHECKPOINT_EXP8,
            "architecture": "DermaAI_MobileNetV3",
            "weight": weight_exp8
        },
        "model_B": {
            "name": "Experiment 13 (Melanoma-Focused Tempered Class Weighting 1.20x)",
            "checkpoint": CHECKPOINT_EXP13,
            "architecture": "DermaAI_MobileNetV3",
            "weight": weight_exp13
        },
        "selected_lambda": selected_lambda,
        "fusion_method": "Probability-level linear blending: P_ens = (1 - lambda) * P8 + lambda * P13",
        "decision_rule": "prediction = argmax(P_ens)",
        "input_resolution": "224x224",
        "dataset_split": "HAM10000 (Val=1010, Test=988)"
    }
    with open(os.path.join(RESULTS_DIR, "ensemble_configuration.json"), "w") as f:
        json.dump(ensemble_config, f, indent=4)
        
    summary_data = {
        "experiment_name": "Experiment 14: Validation-Calibrated Probability Ensemble (Exp8 + Exp13)",
        "selected_lambda": selected_lambda,
        "weight_exp8": weight_exp8,
        "weight_exp13": weight_exp13,
        "selection_rule": selected_lambda_data["selection_rule"],
        "selection_tie_breaker": selection_tie_breaker,
        "validation_macro_f1": float(selected_row["macro_f1"]),
        "validation_melanoma_recall": float(selected_row["melanoma_recall"]),
        "validation_balanced_accuracy": float(selected_row["balanced_accuracy"]),
        "test_accuracy": m_ens["accuracy"],
        "test_balanced_accuracy": m_ens["balanced_accuracy"],
        "test_macro_f1": m_ens["macro_f1"],
        "test_weighted_f1": m_ens["weighted_f1"],
        "test_melanoma_recall": m_ens["melanoma_recall"],
        "test_melanoma_precision": m_ens["melanoma_precision"],
        "test_melanoma_f1": m_ens["melanoma_f1"],
        "test_melanoma_correct": m_ens["melanoma_correct_images"],
        "test_mel_to_nv_errors": m_ens["melanoma_to_nv_errors"],
        "test_mel_to_bkl_errors": m_ens["melanoma_to_bkl_errors"],
        "test_nv_to_mel_errors": m_ens["nv_to_mel_errors"],
        "test_disagreement_rate": disagreement_rate,
        "computational_cost": cost_data
    }
    with open(os.path.join(RESULTS_DIR, "experiment_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=4)

    # ── Final Console Output ───────────────────────────────────────────────────
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION RESULTS: EXP8 vs EXP13 vs EXP14 ENSEMBLE")
    print("="*70)
    print(f"{'Metric':<25} | {'Exp 8':<12} | {'Exp 13':<12} | {'Exp 14 (Ens)':<12} | {'Delta vs Exp8':<14} | {'Delta vs Exp13'}")
    print("-"*90)
    metrics_to_print = [
        ("Accuracy", m_exp8["accuracy"]*100, m_exp13["accuracy"]*100, m_ens["accuracy"]*100, "%"),
        ("Balanced Accuracy", m_exp8["balanced_accuracy"]*100, m_exp13["balanced_accuracy"]*100, m_ens["balanced_accuracy"]*100, "%"),
        ("Macro F1", m_exp8["macro_f1"], m_exp13["macro_f1"], m_ens["macro_f1"], "f"),
        ("Weighted F1", m_exp8["weighted_f1"], m_exp13["weighted_f1"], m_ens["weighted_f1"], "f"),
        ("Melanoma Recall", m_exp8["melanoma_recall"]*100, m_exp13["melanoma_recall"]*100, m_ens["melanoma_recall"]*100, "%"),
        ("Melanoma Precision", m_exp8["melanoma_precision"]*100, m_exp13["melanoma_precision"]*100, m_ens["melanoma_precision"]*100, "%"),
        ("Melanoma F1", m_exp8["melanoma_f1"], m_exp13["melanoma_f1"], m_ens["melanoma_f1"], "f"),
        ("Melanoma TP (/106)", m_exp8["melanoma_correct_images"], m_exp13["melanoma_correct_images"], m_ens["melanoma_correct_images"], "d"),
        ("Mel -> NV Errors", m_exp8["melanoma_to_nv_errors"], m_exp13["melanoma_to_nv_errors"], m_ens["melanoma_to_nv_errors"], "d"),
        ("Mel -> BKL Errors", m_exp8["melanoma_to_bkl_errors"], m_exp13["melanoma_to_bkl_errors"], m_ens["melanoma_to_bkl_errors"], "d"),
        ("NV Recall", m_exp8["nv_recall"]*100, m_exp13["nv_recall"]*100, m_ens["nv_recall"]*100, "%"),
        ("NV -> MEL False Pos", m_exp8["nv_to_mel_errors"], m_exp13["nv_to_mel_errors"], m_ens["nv_to_mel_errors"], "d")
    ]
    for name, v8, v13, v_ens, fmt in metrics_to_print:
        d8 = v_ens - v8
        d13 = v_ens - v13
        if fmt == "%":
            print(f"{name:<25} | {v8:>10.2f}% | {v13:>10.2f}% | {v_ens:>10.2f}% | {d8:>+12.2f}% | {d13:>+12.2f}%")
        elif fmt == "f":
            print(f"{name:<25} | {v8:>12.4f} | {v13:>12.4f} | {v_ens:>12.4f} | {d8:>+14.4f} | {d13:>+14.4f}")
        elif fmt == "d":
            print(f"{name:<25} | {int(v8):>12d} | {int(v13):>12d} | {int(v_ens):>12d} | {int(d8):>+14d} | {int(d13):>+14d}")
            
    print("="*70)
    print(f"Prediction Disagreement Rate (Exp8 vs Exp13) : {disagreement_rate*100:.2f}% ({disagreement_count}/988)")
    print(f"Subset A (Both Correct)     : {count_A} images (Ensemble correct: {ens_on_A_correct}, corrupted: {ens_on_A_corrupted})")
    print(f"Subset B (Exp8-only Correct): {count_B} images (Ensemble correct: {ens_on_B_correct}, lost: {ens_on_B_lost})")
    print(f"Subset C (Exp13-only Correct): {count_C} images (Ensemble correct: {ens_on_C_correct}, lost: {ens_on_C_lost})")
    print(f"Subset D (Both Wrong)       : {count_D} images (Ensemble correct: {ens_on_D_correct})")
    print(f"All 22 Experiment 14 artifacts saved to: {RESULTS_DIR}")
    print("="*70)
    print("Experiment 14 execution finished successfully. STOPPING as required.")


if __name__ == "__main__":
    main()
