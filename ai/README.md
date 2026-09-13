# DermaAI: AI & Machine Learning Subsystem

This directory contains the machine learning pipeline for training the DermaAI prototype on the HAM10000 dataset.

## Dataset Setup
1. **Download the HAM10000 dataset**:
   The HAM10000 dataset (dermoscopic images of common pigmented skin lesions) is provided under the CC BY-NC 4.0 license.
   
   *You must manually download the dataset from the official source and accept the terms of use.*
   
   **Official ISIC Archive Challenge Page**:
   https://challenge.isic-archive.com/data/ (ISIC 2018 Task 3)
   
   Alternatively, via Harvard Dataverse:
   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

2. **Placement**: 
   Place the extracted images (`.jpg`) and `metadata.csv` into `ai/dataset/raw/`. 
   
   **DO NOT** commit these files to Git, and **DO NOT** redistribute the raw dataset through this repository.

## Citation

If you use this model or dataset, please cite the official HAM10000 publication:

> Tschandl, P., Rosendahl, C. & Kittler, H.
> "The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions."
> Scientific Data 5, 180161 (2018).

## Folder Structure
- `dataset/`: Contains raw data, processed reports, and train/val/test splits.
- `src/`: Source code for the pipeline.
- `models/`: Saved model checkpoints (ignored by Git).
- `results/`: Evaluation metrics and dataset visualizations (ignored by Git).

## Setup Instructions

1. **Install dependencies**:
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare & Validate the Dataset**:
   This script parses the metadata, creates leak-free splits (grouping by lesion ID), computes class weights, and generates comprehensive validation reports in `results/dataset/`.
   ```bash
   python src/prepare_data.py
   ```

3. **Train the Model**:
   This trains a MobileNetV3 + Attention model. Checkpoints will be saved to `models/`.
   ```bash
   python src/train.py
   ```

4. **Evaluate the Model**:
   This runs the test set and generates a confusion matrix and classification report in `results/`.
   ```bash
   python src/evaluate.py
   ```

## Limitations
- **Dermoscopic Only**: This model is trained on dermoscopic images. It is not guaranteed to perform well on standard clinical (smartphone) photographs.
- **Preliminary AI**: The output is not a medical diagnosis. The model simply identifies patterns correlated with the 7 HAM10000 classes.
