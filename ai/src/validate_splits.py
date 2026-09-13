import os
import pandas as pd
import json
import torch

DATASET_DIR = "c:/Users/USER/OneDrive/Desktop/skin/ai/dataset"
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")

train_path = os.path.join(SPLITS_DIR, "train.csv")
val_path = os.path.join(SPLITS_DIR, "val.csv")
test_path = os.path.join(SPLITS_DIR, "test.csv")
weights_path = os.path.join(PROCESSED_DIR, "class_weights.json")

print("Checking existence...")
assert os.path.exists(train_path), "train.csv missing"
assert os.path.exists(val_path), "val.csv missing"
assert os.path.exists(test_path), "test.csv missing"
assert os.path.exists(weights_path), "class_weights.json missing"
print("All files exist.")

print("Loading DataFrames...")
train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)
test_df = pd.read_csv(test_path)

print("Checking lesion_id column...")
assert 'lesion_id' in train_df.columns, "lesion_id missing in train_df"
assert 'lesion_id' in val_df.columns, "lesion_id missing in val_df"
assert 'lesion_id' in test_df.columns, "lesion_id missing in test_df"
print("lesion_id column present.")

print("Checking 7 classes representation...")
with open(weights_path, "r") as f:
    weights = json.load(f)
assert len(weights) == 7, "class_weights.json doesn't have 7 classes"
for df, name in zip([train_df, val_df, test_df], ['train', 'val', 'test']):
    assert len(df['dx'].unique()) == 7, f"{name} split doesn't have 7 classes"
print("All 7 classes are represented.")

print("Checking split leakage (lesion_id intersection)...")
train_lesions = set(train_df['lesion_id'])
val_lesions = set(val_df['lesion_id'])
test_lesions = set(test_df['lesion_id'])

assert len(train_lesions.intersection(val_lesions)) == 0, "Leakage between train and val!"
assert len(train_lesions.intersection(test_lesions)) == 0, "Leakage between train and test!"
assert len(val_lesions.intersection(test_lesions)) == 0, "Leakage between val and test!"
print("Split leakage assertions passed.")

print(f"PyTorch Version: {torch.__version__}")
print("PyTorch import successful.")
print("Pre-training validation SUCCESSFUL.")
