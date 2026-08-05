# IE643_TensorTitan

# Unsupervised anomaly detection in 3d volumetric MRI/CT scan 

## Overview
This repository contains the complete implementation of our project, including code files, trained models, and the interface.

## Getting Started
### Dataset Information

#### Training Dataset: **OpenBHB**
- **Description**: OpenBHB (Open Brain Health Dataset) is used as the primary dataset for training the model.
- **Purpose**: Provides a robust dataset for learning normal patterns in brain MRI volumes.
- **Link**: [OpenBHB Dataset](https://www.kaggle.com/datasets/rahulkumargop/openbhb/data)

#### Testing Dataset: **BraTS 2020**
- **Description**: The BraTS 2020 (Brain Tumor Segmentation Challenge) dataset is utilized for anomaly detection testing.
- **Purpose**: Contains annotated brain MRIs, allowing evaluation of the model's anomaly detection capabilities, such as identifying tumors.
- **Link**: [BraTS 2020 Dataset](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

### Model Training

- **Trained Model**: [Link](https://drive.google.com/drive/folders/1ZVSV-4akGO8kbln2dH_ognXTFInPG2jH?usp=sharing)
- **To Train**: Download any provided notebook, upload the dataset, and execute the notebook to train the model.

## Hyperparameters
- `learning_rate`: Default 1e-4. Adjust based on dataset size.
- `batch_size`: Default 8. Adjust based on available memory.
- `epoch` : Default 100. With early stopping.
- `optimizer` : Adam  

## Getting Started with GUI

Follow these steps to set up and run the GUI for the MRI Viewer and Analysis Tool:

### 1. Clone the Repository
```bash
git clone <repository-link>
```
### 2. Navigate to the GUI Directory

Change your working directory to the folder containing the GUI files:
```bash
cd <repository-folder>/GUI
```

### 3. Prerequisite: Install Python 3.8
Ensure you have Python 3.8 installed on your system. 

### 4. Install Required Dependencies
Install all necessary Python packages using the requirements.txt file:
```bash
pip install -r requirements.txt

```

### 5. Run the Application
Launch the GUI by running the main application file:
```bash
python tk_app.py
```

## Classical-ML baseline (Spec 006)

`make classical` runs the honest control: engineered radiomic/texture features (first-order,
GLCM, gradient, GLRLM — 44/slice) into gradient-boosted trees, cross-validated on the identical
BraTS test split the reconstruction pipeline is scored on.

- **Operating granularity is slice-level**, not voxel-level: each 2D axial slice is one sample,
  classified tumor-bearing or not. This gives a clean ROC-AUC but **does not localize**, so it is
  **not directly comparable to the reconstruction pipeline's voxel-level Dice** — that gap is
  reported plainly, never glossed over.
- **PyRadiomics substitution.** PyRadiomics is fragile to build on some platforms, so this
  baseline uses `scikit-image` for GLCM/first-order/gradient features plus a hand-rolled GLRLM
  (skimage ships no run-length equivalent) instead. Every generated artifact records
  `features.library: scikit-image` / `pyradiomics_used: false` so the substitution is visible,
  not hidden.

