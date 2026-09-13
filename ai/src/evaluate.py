import os
import json
import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

try:
    from .model import DermaAI_MobileNetV3
    from .preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING, REVERSE_CLASS_MAPPING
except ImportError:
    from model import DermaAI_MobileNetV3
    from preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING, REVERSE_CLASS_MAPPING

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

def evaluate():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # 1. Load Data
    test_csv = os.path.join(SPLITS_DIR, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError("test.csv not found. Run prepare_data.py first.")
        
    test_df = pd.read_csv(test_csv)
    test_dataset = HAM10000Dataset(test_df, transform=get_transforms(is_train=False))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 2. Load Model
    model = DermaAI_MobileNetV3(num_classes=7, use_attention=True, pretrained=False)
    model_path = os.path.join(MODELS_DIR, "best_model.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Train the model first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Inference
    all_preds = []
    all_targets = []
    
    print("Running inference on test set...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.numpy())

    # 4. Calculate Metrics
    class_names = [REVERSE_CLASS_MAPPING[i] for i in range(7)]
    
    acc = accuracy_score(all_targets, all_preds)
    bacc = balanced_accuracy_score(all_targets, all_preds)
    
    print(f"\nAccuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    report_text = classification_report(all_targets, all_preds, target_names=class_names)
    
    print("\nClassification Report:")
    print(report_text)
    
    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Balanced Accuracy: {bacc:.4f}\n\n")
        f.write(report_text)
        
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump({
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "report": report
        }, f, indent=4)
        
    # 5. Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix on Test Set')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    print(f"Saved evaluation artifacts to {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate()
