import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

try:
    from .model import DermaAI_MobileNetV3
    from .preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING
except ImportError:
    from model import DermaAI_MobileNetV3
    from preprocessing import HAM10000Dataset, get_transforms, CLASS_MAPPING

# Config
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
PATIENCE = 5
LOSS_TYPE = "weighted_cross_entropy" # options: "cross_entropy", "weighted_cross_entropy", "focal"
USE_ATTENTION = True

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def get_loss_function(loss_type, device):
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss()
        
    # Load class weights
    weights_path = os.path.join(SPLITS_DIR, "class_weights.json")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing {weights_path}. Run prepare_data.py first.")
        
    with open(weights_path, "r") as f:
        weight_dict = json.load(f)
    
    # Ensure ordered correctly
    weights = [weight_dict[k] for k in CLASS_MAPPING.keys()]
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    if loss_type == "weighted_cross_entropy":
        return nn.CrossEntropyLoss(weight=weights_tensor)
    elif loss_type == "focal":
        return FocalLoss(alpha=weights_tensor, gamma=2)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))

    train_dataset = HAM10000Dataset(train_df, transform=get_transforms(is_train=True))
    val_dataset = HAM10000Dataset(val_df, transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # 2. Initialize Model
    model = DermaAI_MobileNetV3(num_classes=7, use_attention=USE_ATTENTION, pretrained=True)
    model.to(device)

    # Transfer Learning: Initially freeze features (except attention and classifier)
    for param in model.features.parameters():
        param.requires_grad = False

    # 3. Setup Optimizer, Loss, Scheduler
    criterion = get_loss_function(LOSS_TYPE, device)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    # 4. Training Loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"Starting training for {EPOCHS} epochs using {LOSS_TYPE}...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Unfreeze backbone halfway through for fine-tuning
        if epoch == EPOCHS // 2:
            print("Unfreezing backbone for fine-tuning...")
            for param in model.features.parameters():
                param.requires_grad = True
            # Re-init optimizer with smaller LR for fine-tuning
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
            for inputs, targets in pbar_val:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1} Summary: Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        
        scheduler.step(epoch_val_loss)
        
        # Early Stopping & Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, "best_model.pt"))
            print("--> Saved best model")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break

    # Save final history
    with open(os.path.join(RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    print("Training complete.")

if __name__ == "__main__":
    train()
