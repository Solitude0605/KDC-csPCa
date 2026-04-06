# Data Directory

This directory contains data files used in the analysis.

## Required Data Files

### Clinical Data
- `clinical_data.csv` - Clinical features and laboratory results
  - Columns: patient_id, age, BMI, tPSA, fPSA, f/tPSA, PSAD, PV, TV, TD, PI-RADS, EPE, DRE, etc.

### Radiomics Features
- `T2WI_features.csv` - T2WI radiomics features (107 features)
- `DWI_features.csv` - DWI/DCE radiomics features (107 features)
- `DCE_features.csv` - DCE radiomics features (107 features)
- `selected_features.csv` - Final selected features after LASSO and correlation filtering

### Model Outputs
- `single_modality_predictions.csv` - Predictions from single-modality models
- `fusion_predictions.csv` - Predictions from fusion models

### Ground Truth Labels
- `labels.csv` - csPCa vs non-csPCa labels
  - 1 = csPCa (clinically significant prostate cancer)
  - 0 = non-csPCa (non-clinically significant prostate cancer)

## Data Preprocessing Steps

1. **Missing value imputation**: Median imputation for radiomics features
2. **Normalization**: Z-score normalization (scaled to 100 for MRI)
3. **Feature selection**: LASSO regression + Spearman correlation (ρ > 0.75)
4. **Harmonization**: Combat harmonization for multi-center data

## Data Split

- **Training/Internal validation**: 699 patients (Centers 1 & 2)
- **External test**: 60 patients (Center 3)

## Note

Due to patient privacy protection, raw MRI data and clinical information are not publicly available. Researchers interested in collaboration may contact the corresponding author.
