# KDC Model: Knowledge-Driven Conditional Fusion for csPCa Identification

## Overview
This repository contains the analysis code for the manuscript:
**"Knowledge-Driven Conditional Model Integrating mpMRI Radiomics with Clinical Information for Clinically Significant Prostate Cancer Identification"**

## Data Description
- **Training cohort**: 699 patients from Centers 1 and 2
- **External test cohort**: 60 patients from Center 3
- **Modalities**: T2WI, DWI, DCE, Clinical features
- **Outcome**: Clinically significant prostate cancer (csPCa) vs non-csPCa

## Directory Structure

```
Code/
├── 01-Preprocessing/                    # MRI preprocessing and registration
│   ├── dicom_to_nii_batch.py           # DICOM to NIfTI conversion
│   ├── rigid registration.py           # Rigid registration (T2WI, DWI, DCE)
│   ├── 1-mask-parameter.py            # VOI mask parameter settings
│   └── 1-bounding rectangle.py                   # Bounding box extraction
│
├── 02-Radiomics_Feature_Extraction/    # Radiomics feature extraction
│   ├── main-t2.py                     # T2WI feature extraction
│   ├── main-dce.py                    # DWI/DCE feature extraction
│   ├── 3-extract_radiomics_t2.py      # T2WI radiomics extraction
│   └── 3-extract_radiomics_adc.py     # DCE radiomics extraction
│
├── 03-Feature_Selection/              # Feature selection
│   └── feature_selection.py           # LASSO + correlation analysis
│
├── 04-Model_Training/                 # Single-modality model training
│   ├── 5-train_tz_pirads_all_clinical_model.py      # Clinical model
│   ├── 5-train_tz_pirads_align_radiomics_model.py   # Radiomics model
│   └── 5-train_tz_pirads_align_clinical_model_PLUS.py
│
├── 05-Model_Evaluation/               # Model evaluation on test set
│   ├── Clinical-test.py               # Clinical model evaluation
│   ├── Radiomics-test.py              # Radiomics model evaluation
│   └── Fusion_test.py                 # Fusion model evaluation
│
├── 06-Fusion_Models/                  # Late fusion strategies
│   ├── late_fusion_multimodal-pkl.py  # KDC, Stacking LR, etc.
│   ├── ROC-Fusion.py                  # ROC for fusion models
│   ├── ROC-Single.py                  # ROC for single models
│   └── ROC-Ex.py                      # ROC for external test
│
├── 07-Statistical_Analysis/           # Statistical analysis
│   ├── Fusion_shap_analysis_stacking.py   # SHAP for fusion
│   └── Clinical_shap_analysis_stacking.py  # SHAP for clinical
│
└── 08-Visualization/                 # Visualization scripts
    ├── Delong-hot.py                  # DeLong test heatmap
    ├── DCA-Last.py                    # Decision curve analysis
    ├── calibration_curves.py          # Calibration curves
    └── radar.py                       # Radar charts
```

## Key Models

### Single-Modality Models
- Clinical model (balanced RF)
- T2WI radiomics model (Extra Trees)
- DWI radiomics model (balanced RF)
- DCE radiomics model (SVM-RBF)

### Fusion Strategies
1. **Early Fusion** - Feature concatenation
2. **KDC (Knowledge-Driven Conditional)** - Conditional late fusion with prior knowledge weights
3. **Data-Driven Conditional** - Conditional late fusion with data-driven thresholds
4. **Stacking LR** - Logistic regression meta-learner
5. **Stacking RF** - Random forest meta-learner
6. **Heuristic** - Fixed weight averaging
7. **AUC-weighted** - Performance-weighted averaging

## Dependencies

```bash
numpy
pandas
scikit-learn
pydicom
SimpleITK
PyRadiomics
shap
matplotlib
seaborn
```

## Usage

1. **Preprocessing**: Run DICOM to NIfTI conversion, then rigid registration
2. **Feature Extraction**: Extract radiomics features using PyRadiomics
3. **Feature Selection**: Apply LASSO + correlation filtering
4. **Model Training**: Train single-modality models with nested CV
5. **Fusion**: Apply late fusion strategies
6. **Evaluation**: Evaluate on internal and external test sets

## Citation

If this code is helpful for your research, please cite our paper.
