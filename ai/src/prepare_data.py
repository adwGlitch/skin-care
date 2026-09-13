import os
import glob
import json
import warnings
import pandas as pd
from PIL import Image
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Suppress PIL decompression bomb warnings if any large images exist (not typical for HAM10000, but good practice)
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

# Configuration - Allow overriding via environment variables
DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "dataset")
RAW_DIR = os.environ.get("DERMAAI_DATASET_ROOT", os.path.join(DATASET_DIR, "raw"))
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")
SPLITS_DIR = os.path.join(DATASET_DIR, "splits")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "dataset")

# Using ISIC 2018 Task 3 GroundTruth for actual labels
METADATA_PATH = os.environ.get("DERMAAI_METADATA_PATH", os.path.join(RAW_DIR, "metadata.csv"))
# Using the original Harvard Dataverse HAM10000 metadata to get true lesion_ids
LESION_MAPPING_PATH = os.environ.get("DERMAAI_LESION_MAPPING_PATH", None)

EXPECTED_IMAGE_COUNT = 10015

CLASS_MAPPING = {
    'akiec': 'Actinic keratoses / intraepithelial carcinoma',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevus',
    'vasc': 'Vascular lesion'
}

def create_dirs():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def parse_metadata(df, mapping_df=None):
    """
    Adapts the metadata CSV to the expected format (image_id, dx, lesion_id).
    Handles both standard HAM10000 format and ISIC 2018 Task 3 GroundTruth format.
    """
    # If it's ISIC 2018 Task 3 GroundTruth (one-hot encoded)
    if 'image' in df.columns and 'MEL' in df.columns:
        df = df.rename(columns={'image': 'image_id'})
        
        # Get the one-hot columns (all except image_id)
        disease_cols = [c for c in df.columns if c != 'image_id']
        
        # Convert one-hot to a single 'dx' column (lowercase to match our classes)
        # Check which column has the maximum value (1.0)
        df['dx'] = df[disease_cols].idxmax(axis=1).str.lower()
        
        # Drop the original one-hot columns
        df = df.drop(columns=disease_cols)
        
    # If lesion_id is missing but we have an authoritative mapping
    if 'lesion_id' not in df.columns:
        if mapping_df is not None:
            # We assume mapping_df has 'image_id' and 'lesion_id'
            print("Applying authoritative lesion_id mapping from Harvard Dataverse metadata...")
            mapping_subset = mapping_df[['image_id', 'lesion_id']]
            df = df.merge(mapping_subset, on='image_id', how='left')
            
            missing_lesion_ids = df['lesion_id'].isna().sum()
            if missing_lesion_ids > 0:
                print(f"CRITICAL WARNING: {missing_lesion_ids} images could not be matched to a true lesion_id in the mapping file!")
        else:
            print("Notice: 'lesion_id' not found in metadata. Using 'image_id' as 'lesion_id' for splitting.")
            df['lesion_id'] = df['image_id']
            
    return df

def validate_dataset():
    print("==================================================")
    print("1. DATASET VALIDATION")
    print("==================================================")
    print(f"Dataset Root: {RAW_DIR}")
    print(f"Metadata Path: {METADATA_PATH}")
    print(f"Lesion Mapping Path: {LESION_MAPPING_PATH}")
    
    if not os.path.exists(METADATA_PATH):
        print(f"ERROR: {METADATA_PATH} not found.")
        return None, None
        
    df = pd.read_csv(METADATA_PATH)
    
    mapping_df = None
    if LESION_MAPPING_PATH and os.path.exists(LESION_MAPPING_PATH):
        mapping_df = pd.read_csv(LESION_MAPPING_PATH)
    
    df = parse_metadata(df, mapping_df=mapping_df)
    
    # Locate all images in RAW_DIR
    all_jpgs = glob.glob(os.path.join(RAW_DIR, "**", "*.jpg"), recursive=True)
    
    # Build a quick lookup for images
    image_paths = {}
    duplicate_filenames = set()
    for path in all_jpgs:
        basename = os.path.splitext(os.path.basename(path))[0]
        if basename in image_paths:
            duplicate_filenames.add(basename)
        image_paths[basename] = path
        
    print(f"\nMetadata rows: {len(df)}")
    print(f"Image files found: {len(all_jpgs)}")
    
    # True lesion statistics
    true_lesion_count = df['lesion_id'].nunique()
    print(f"True unique lesions found: {true_lesion_count}")
    
    lesion_counts = df.groupby('lesion_id').size()
    multi_image_lesions = lesion_counts[lesion_counts > 1]
    images_in_multi_lesions = multi_image_lesions.sum()
    print(f"Images belonging to multi-image lesions: {images_in_multi_lesions}")
    
    if len(df) < EXPECTED_IMAGE_COUNT:
        print(f"\nWARNING: Expected {EXPECTED_IMAGE_COUNT} images but found {len(df)} metadata rows.")
        
    # Check for missing and orphan images
    df['image_path'] = df['image_id'].map(image_paths)
    missing_images = df[df['image_path'].isna()]
    
    metadata_image_ids = set(df['image_id'])
    found_image_ids = set(image_paths.keys())
    orphan_images = found_image_ids - metadata_image_ids
    
    print(f"Missing images (in metadata but not on disk): {len(missing_images)}")
    print(f"Orphan images (on disk but not in metadata): {len(orphan_images)}")
    print(f"Duplicate filenames on disk: {len(duplicate_filenames)}")
    
    # Check for duplicates in metadata
    duplicate_metadata_ids = df[df.duplicated(['image_id'])]
    print(f"Duplicate image_ids in metadata: {len(duplicate_metadata_ids)}")
    
    # Remove missing images from dataframe to proceed
    valid_df = df.dropna(subset=['image_path']).copy()
    
    # Validate classes
    classes_found = set(valid_df['dx'])
    expected_classes = set(CLASS_MAPPING.keys())
    print("\nClass validation:")
    for cls in expected_classes:
        status = "OK" if cls in classes_found else "MISSING"
        print(f"  {cls} -> {CLASS_MAPPING[cls]}: {status}")
        
    if classes_found != expected_classes:
        print("WARNING: Not all expected classes are present in the dataset!")

    # Image corruption and dimension checks
    print("\nValidating image files (this may take a minute)...")
    corrupt_images = 0
    dimensions = []
    
    for idx, row in valid_df.iterrows():
        try:
            with Image.open(row['image_path']) as img:
                img.verify()
            # Reopen to get size since verify() might close/alter state
            with Image.open(row['image_path']) as img:
                dimensions.append(img.size)
        except Exception:
            corrupt_images += 1
            valid_df.drop(idx, inplace=True)
            
    print(f"Corrupt/unreadable images: {corrupt_images}")
    
    valid_df['dimensions'] = dimensions
    unique_dims = Counter(dimensions)
    print("Image Dimensions:")
    for dim, count in unique_dims.items():
        print(f"  {dim}: {count} images")
        
    return valid_df, {
        "expected_count": EXPECTED_IMAGE_COUNT,
        "metadata_rows": len(df),
        "true_lesion_count": true_lesion_count,
        "images_in_multi_lesions": int(images_in_multi_lesions),
        "images_on_disk": len(all_jpgs),
        "missing_images": len(missing_images),
        "orphan_images": len(orphan_images),
        "duplicate_metadata_ids": len(duplicate_metadata_ids),
        "duplicate_filenames": len(duplicate_filenames),
        "corrupt_images": corrupt_images,
        "valid_images": len(valid_df),
        "dimensions": {str(k): v for k, v in unique_dims.items()}
    }

def split_dataset(df):
    print("\n==================================================")
    print("2. LESION-LEVEL SPLITTING")
    print("==================================================")
    
    # Group by lesion_id
    lesion_df = df.groupby('lesion_id').first().reset_index()
    print(f"Total unique lesions in dataset: {len(lesion_df)}")
    
    # Stratified split at the lesion level (using the diagnosis of the first image of the lesion)
    train_lesions, temp_lesions = train_test_split(
        lesion_df['lesion_id'], test_size=0.2, random_state=42, stratify=lesion_df['dx']
    )
    # Stratify the remaining 20% into 10% Val / 10% Test
    temp_df = lesion_df[lesion_df['lesion_id'].isin(temp_lesions)]
    val_lesions, test_lesions = train_test_split(
        temp_df['lesion_id'], test_size=0.5, random_state=42, stratify=temp_df['dx']
    )
    
    # Map back to images
    train_df = df[df['lesion_id'].isin(train_lesions)].copy()
    val_df = df[df['lesion_id'].isin(val_lesions)].copy()
    test_df = df[df['lesion_id'].isin(test_lesions)].copy()
    
    # Ensure splits preserve approximately 80/10/10 structure
    print(f"\nImages in Train Split: {len(train_df)} ({len(train_lesions)} lesions)")
    print(f"Images in Validation Split: {len(val_df)} ({len(val_lesions)} lesions)")
    print(f"Images in Test Split: {len(test_df)} ({len(test_lesions)} lesions)")
    
    # Leakage Verification
    train_set = set(train_df['lesion_id'])
    val_set = set(val_df['lesion_id'])
    test_set = set(test_df['lesion_id'])
    
    overlap_tv = len(train_set.intersection(val_set))
    overlap_tt = len(train_set.intersection(test_set))
    overlap_vt = len(val_set.intersection(test_set))
    
    print(f"\nTrain <-> Val lesion overlap: {overlap_tv}")
    print(f"Train <-> Test lesion overlap: {overlap_tt}")
    print(f"Val <-> Test lesion overlap: {overlap_vt}")
    
    assert overlap_tv == 0, "LEAKAGE DETECTED between Train and Validation!"
    assert overlap_tt == 0, "LEAKAGE DETECTED between Train and Test!"
    assert overlap_vt == 0, "LEAKAGE DETECTED between Validation and Test!"
    print("Strict leakage assertions passed. 0 overlap.")
    
    return train_df, val_df, test_df

def process_and_save(train_df, val_df, test_df, report_metrics):
    print("\n==================================================")
    print("3. WEIGHTS, REPORTING, AND VISUALIZATION")
    print("==================================================")
    
    # Save splits (drop the dimensions column as it is not needed for training)
    train_df.drop(columns=['dimensions'], inplace=True, errors='ignore')
    val_df.drop(columns=['dimensions'], inplace=True, errors='ignore')
    test_df.drop(columns=['dimensions'], inplace=True, errors='ignore')
    
    train_df.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)
    
    # Calculate weights on TRAIN only
    counts = Counter(train_df['dx'])
    total_train = len(train_df)
    
    class_weights = {}
    for cls in CLASS_MAPPING.keys():
        count = counts.get(cls, 0)
        class_weights[cls] = total_train / (len(CLASS_MAPPING) * count) if count > 0 else 0.0
        
    with open(os.path.join(PROCESSED_DIR, "class_weights.json"), "w") as f:
        json.dump(class_weights, f, indent=4)
        
    # Generate Report
    report = {
        "dataset_name": "HAM10000 / ISIC 2018 Task 3",
        "dataset_root": RAW_DIR,
        "metadata_path": METADATA_PATH,
        "lesion_mapping_path": LESION_MAPPING_PATH,
        "source": "ISIC Archive / Harvard Dataverse",
        "license": "CC BY-NC 4.0",
        "citation": "Tschandl, P., Rosendahl, C. & Kittler, H. 'The HAM10000 dataset...'",
        "metrics": report_metrics,
        "splits": {
            "train_images": len(train_df),
            "validation_images": len(val_df),
            "test_images": len(test_df),
            "train_lesions": train_df['lesion_id'].nunique(),
            "validation_lesions": val_df['lesion_id'].nunique(),
            "test_lesions": test_df['lesion_id'].nunique()
        },
        "class_distributions_images": {
            "train": dict(counts),
            "validation": dict(Counter(val_df['dx'])),
            "test": dict(Counter(test_df['dx']))
        },
        "class_weights": class_weights
    }
    
    with open(os.path.join(PROCESSED_DIR, "dataset_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    with open(os.path.join(PROCESSED_DIR, "dataset_report.txt"), "w") as f:
        f.write("HAM10000 DATASET REPORT\n")
        f.write("=======================\n")
        for k, v in report.items():
            if isinstance(v, dict):
                f.write(f"\n{k}:\n")
                for sub_k, sub_v in v.items():
                    f.write(f"  {sub_k}: {sub_v}\n")
            else:
                f.write(f"{k}: {v}\n")
                
    # Visualizations
    plt.figure(figsize=(10, 6))
    classes = list(CLASS_MAPPING.keys())
    x = np.arange(len(classes))
    width = 0.25
    
    tr_counts = [report["class_distributions_images"]["train"].get(c, 0) for c in classes]
    va_counts = [report["class_distributions_images"]["validation"].get(c, 0) for c in classes]
    te_counts = [report["class_distributions_images"]["test"].get(c, 0) for c in classes]
    
    plt.bar(x - width, tr_counts, width, label='Train')
    plt.bar(x, va_counts, width, label='Validation')
    plt.bar(x + width, te_counts, width, label='Test')
    
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution Across Splits')
    plt.xticks(x, classes)
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, "class_distribution.png"))
    plt.close()
    
    # Sample Grid
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('HAM10000 Sample Images')
    axes = axes.flatten()
    
    for i, cls in enumerate(classes):
        # We need to handle classes that might not exist, though we verified them earlier
        class_samples = train_df[train_df['dx'] == cls]
        if len(class_samples) > 0:
            sample_row = class_samples.iloc[0]
            img = Image.open(sample_row['image_path'])
            axes[i].imshow(img)
            axes[i].set_title(f"{cls}\n({CLASS_MAPPING[cls][:15]}...)")
        axes[i].axis('off')
    axes[7].axis('off') # Hide 8th subplot
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sample_grid.png"))
    plt.close()
    
    print("\nFiles created:")
    print(f"- {os.path.join(SPLITS_DIR, 'train.csv')}")
    print(f"- {os.path.join(SPLITS_DIR, 'val.csv')}")
    print(f"- {os.path.join(SPLITS_DIR, 'test.csv')}")
    print(f"- {os.path.join(PROCESSED_DIR, 'class_weights.json')}")
    print(f"- {os.path.join(PROCESSED_DIR, 'dataset_report.json')}")
    print(f"- {os.path.join(PROCESSED_DIR, 'dataset_report.txt')}")
    print(f"- {os.path.join(RESULTS_DIR, 'class_distribution.png')}")
    print(f"- {os.path.join(RESULTS_DIR, 'sample_grid.png')}")

def main():
    create_dirs()
    df, metrics = validate_dataset()
    if df is not None:
        train_df, val_df, test_df = split_dataset(df)
        process_and_save(train_df, val_df, test_df, metrics)
        print("\nDataset preparation completed successfully!")

if __name__ == "__main__":
    main()
