import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.linear_model import LassoCV, Lasso, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Set plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams.get('font.serif', [])

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
data_dir = r"K:\PCa_2026\Article\放射组学\Claude\radiomics_csv"
output_dir = r"K:\PCa_2026\Article\放射组学\Claude\Figure"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("Radiomics Feature Selection Process")
print("=" * 70)

# Load data
dce_df = pd.read_csv(os.path.join(data_dir, 'radiomics_features_dce.csv'))
dwi_df = pd.read_csv(os.path.join(data_dir, 'radiomics_features_dwi.csv'))
t2_df = pd.read_csv(os.path.join(data_dir, 'radiomics_features_t2.csv'))

label_col = 'csPCa'
id_col = 'patient_ids'

# Data alignment
common_ids = set(dce_df[id_col]) & set(dwi_df[id_col]) & set(t2_df[id_col])
print(f"\n📊 Common patients: {len(common_ids)}")

dce_df = dce_df[dce_df[id_col].isin(common_ids)].sort_values(id_col).reset_index(drop=True)
dwi_df = dwi_df[dwi_df[id_col].isin(common_ids)].sort_values(id_col).reset_index(drop=True)
t2_df = t2_df[t2_df[id_col].isin(common_ids)].sort_values(id_col).reset_index(drop=True)

# Extract labels
if label_col in dce_df.columns:
    y = dce_df[label_col].values
    print(f"Label distribution: Positive={sum(y == 1)}, Negative={sum(y == 0)}")
else:
    raise ValueError(f"Label column '{label_col}' not found")

# Extract features
exclude_cols = [id_col, label_col]


def extract_features(df, modality_name):
    feature_cols = [col for col in df.columns
                    if col.startswith('original_') and col not in exclude_cols]
    features = df[feature_cols].copy()
    features = features.add_prefix(f'{modality_name}_')
    print(f"  {modality_name}: {len(feature_cols)} features")
    return features


dce_features = extract_features(dce_df, 'DCE')
dwi_features = extract_features(dwi_df, 'DWI')
t2_features = extract_features(t2_df, 'T2')

# Combine features
X = pd.concat([dce_features, dwi_features, t2_features], axis=1)
feature_names = X.columns.tolist()
print(f"\n📊 Initial total features: {X.shape[1]}")

# Handle missing values
X = X.fillna(X.mean())

# ==========================================
# 2. (a) Variance Threshold Selection
# ==========================================
print("\n" + "=" * 70)
print("(a) Variance Threshold Feature Selection")
print("=" * 70)

# Calculate variances
variances = X.var()
variances_sorted = variances.sort_values(ascending=False)

# Set variance threshold
variance_threshold = 0.01
selector_variance = VarianceThreshold(threshold=variance_threshold)
X_variance_selected = selector_variance.fit_transform(X)
selected_variance_features = X.columns[selector_variance.get_support()].tolist()

print(f"📊 Variance threshold: {variance_threshold}")
print(f"📊 Features retained: {len(selected_variance_features)}/{X.shape[1]}")
print(f"📊 Retention rate: {len(selected_variance_features) / X.shape[1] * 100:.1f}%")

# Visualize variance distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Variance distribution of all features
axes[0].hist(variances, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=variance_threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold={variance_threshold}')
axes[0].set_xlabel('Variance', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('(a1) Feature Variance Distribution', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: Top 30 high-variance features
top_n = 30
top_features = variances_sorted.head(top_n)
axes[1].barh(range(top_n), top_features.values, color='steelblue')
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top_features.index, fontsize=8)
axes[1].set_xlabel('Variance', fontsize=12)
axes[1].set_title(f'(a2) Top {top_n} High-Variance Features', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(alpha=0.3)

plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'Feature_Selection_Variance.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Feature_Selection_Variance.tif'), dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 3. (b) LASSO with 5-fold CV for lambda selection
# ==========================================
print("\n" + "=" * 70)
print("(b) LASSO 5-fold Cross-Validation for Lambda Selection")
print("=" * 70)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_variance_selected)

# Use LassoCV for 5-fold cross-validation
lasso_cv = LassoCV(
    cv=5,
    random_state=42,
    max_iter=10000,
    n_alphas=100,
    verbose=False
)

lasso_cv.fit(X_scaled, y)

print(f"📊 Optimal lambda (alpha): {lasso_cv.alpha_:.4f}")
print(f"📊 Minimum MSE: {lasso_cv.mse_path_.mean(axis=1).min():.4f}")

# Get features with non-zero coefficients
lasso_coef = pd.Series(lasso_cv.coef_, index=selected_variance_features)
lasso_selected_features = lasso_coef[lasso_coef != 0].index.tolist()
print(f"📊 Features selected by LASSO: {len(lasso_selected_features)}")

# Plot LASSO cross-validation results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: MSE vs lambda
axes[0].plot(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1), 'b-', linewidth=2, label='Mean MSE')
axes[0].fill_between(lasso_cv.alphas_,
                     lasso_cv.mse_path_.mean(axis=1) - lasso_cv.mse_path_.std(axis=1),
                     lasso_cv.mse_path_.mean(axis=1) + lasso_cv.mse_path_.std(axis=1),
                     alpha=0.2, color='blue')
axes[0].axvline(x=lasso_cv.alpha_, color='red', linestyle='--', linewidth=2,
                label=f'Best λ={lasso_cv.alpha_:.4f}')
axes[0].set_xscale('log')
axes[0].set_xlabel('Lambda (log scale)', fontsize=12)
axes[0].set_ylabel('Mean Squared Error (MSE)', fontsize=12)
axes[0].set_title('(b1) LASSO Cross-Validation Error Curve', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: MSE for each fold
for i in range(5):
    axes[1].plot(lasso_cv.alphas_, lasso_cv.mse_path_[:, i], '--', alpha=0.5,
                 label=f'Fold {i + 1}' if i == 0 else "")
axes[1].plot(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=1), 'b-', linewidth=3, label='Mean')
axes[1].axvline(x=lasso_cv.alpha_, color='red', linestyle='--', linewidth=2)
axes[1].set_xscale('log')
axes[1].set_xlabel('Lambda (log scale)', fontsize=12)
axes[1].set_ylabel('Mean Squared Error (MSE)', fontsize=12)
axes[1].set_title('(b2) Cross-Validation Error by Fold', fontsize=13, fontweight='bold')
axes[1].legend(loc='upper right', fontsize=8)
axes[1].grid(alpha=0.3)

plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'Feature_Selection_LASSO_CV.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Feature_Selection_LASSO_CV.tif'), dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 4. (c) LASSO Coefficient Plot
# ==========================================
print("\n" + "=" * 70)
print("(c) LASSO Coefficient Plot")
print("=" * 70)

# Sort by absolute coefficient values
lasso_coef_sorted = lasso_coef.abs().sort_values(ascending=False)
lasso_coef_sorted_with_sign = lasso_coef[lasso_coef_sorted.index]

# Select top features
n_top = min(30, len(lasso_selected_features))
top_coef = lasso_coef_sorted_with_sign.head(n_top)

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Left: Coefficient path plot
alphas = np.logspace(-4, 1, 100)
coefs = []
for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y)
    coefs.append(lasso.coef_)

coefs = np.array(coefs)
for i in range(min(20, len(selected_variance_features))):
    axes[0].plot(alphas, coefs[:, i], linewidth=1.5, alpha=0.7,
                 label=selected_variance_features[i] if i < 5 else "")
axes[0].axvline(x=lasso_cv.alpha_, color='red', linestyle='--', linewidth=2,
                label=f'Best λ={lasso_cv.alpha_:.4f}')
axes[0].set_xscale('log')
axes[0].set_xlabel('Lambda', fontsize=12)
axes[0].set_ylabel('Coefficient Value', fontsize=12)
axes[0].set_title('(c1) LASSO Coefficient Path', fontsize=13, fontweight='bold')
axes[0].legend(loc='upper right', fontsize=8)
axes[0].grid(alpha=0.3)

# Right: Top feature coefficients bar plot
colors = ['red' if x > 0 else 'blue' for x in top_coef.values]
axes[1].barh(range(len(top_coef)), top_coef.values, color=colors, alpha=0.7)
axes[1].set_yticks(range(len(top_coef)))
axes[1].set_yticklabels(top_coef.index, fontsize=9)
axes[1].axvline(x=0, color='black', linewidth=1)
axes[1].set_xlabel('Coefficient Value', fontsize=12)
axes[1].set_title(f'(c2) Top {n_top} LASSO Coefficients (Red: Positive, Blue: Negative)',
                  fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(alpha=0.3)

plt.tight_layout()
# plt.savefig(os.path.join(output_dir, 'Feature_Selection_LASSO_Coefficients.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Feature_Selection_LASSO_Coefficients.tif'), dpi=300, bbox_inches='tight')
plt.show()

# Print LASSO selected features
print("\n📊 LASSO selected features (non-zero coefficients):")
lasso_nonzero = lasso_coef[lasso_coef != 0].sort_values(ascending=False)
for feature, coef in lasso_nonzero.items():
    print(f"  {feature}: {coef:.4f}")

# ==========================================
# 5. (d) RFECV with 5-fold CV
# ==========================================
print("\n" + "=" * 70)
print("(d) Recursive Feature Elimination with 5-fold Cross-Validation (RFECV)")
print("=" * 70)

# Use Logistic Regression as estimator
estimator = LogisticRegression(
    max_iter=5000,
    random_state=42,
    class_weight='balanced'
)

# Create RFECV object
rfecv = RFECV(
    estimator=estimator,
    step=1,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring='roc_auc',
    min_features_to_select=1,
    n_jobs=-1,
    verbose=0
)

# Fit on standardized data
scaler_rfe = StandardScaler()
X_scaled_rfe = scaler_rfe.fit_transform(X_variance_selected)

print("Running RFECV, this may take some time...")
rfecv.fit(X_scaled_rfe, y)

print(f"📊 Optimal number of features from RFECV: {rfecv.n_features_}")

# Get the cross-validation scores - fixed attribute name
cv_scores = rfecv.cv_results_['mean_test_score']  # Changed from grid_scores_
print(f"📊 Best cross-validation score (AUC): {cv_scores.max():.4f}")

# Get selected features
rfecv_selected_indices = rfecv.support_
rfecv_selected_features = [selected_variance_features[i] for i in range(len(selected_variance_features))
                          if rfecv_selected_indices[i]]
print(f"📊 Features retained by RFECV: {len(rfecv_selected_features)}")

# Plot RFECV results
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: CV score vs number of features - fixed to use cv_results_
n_features = len(cv_scores)
axes[0].plot(range(1, n_features + 1), cv_scores, 'b-o', linewidth=2, markersize=4)
axes[0].axvline(x=rfecv.n_features_, color='red', linestyle='--', linewidth=2,
                label=f'Optimal features={rfecv.n_features_}')
axes[0].set_xlabel('Number of Features', fontsize=12)
axes[0].set_ylabel('Cross-Validation AUC', fontsize=12)
axes[0].set_title('(d1) RFECV Feature Selection Curve', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Right: Feature importance ranking
feature_ranking = pd.Series(rfecv.ranking_, index=selected_variance_features)
top_features_rfe = feature_ranking.sort_values().head(20)

axes[1].barh(range(len(top_features_rfe)), 1/top_features_rfe.values, color='green', alpha=0.7)
axes[1].set_yticks(range(len(top_features_rfe)))
axes[1].set_yticklabels(top_features_rfe.index, fontsize=9)
axes[1].set_xlabel('Importance Score (1/rank)', fontsize=12)
axes[1].set_title('(d2) Top 20 Important Features (Based on RFECV Ranking)', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Feature_Selection_RFECV.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'Feature_Selection_RFECV.tif'), dpi=300, bbox_inches='tight')
plt.show()

# ==========================================
# 6. Feature Selection Results Summary
# ==========================================
print("\n" + "=" * 70)
print("Feature Selection Results Summary")
print("=" * 70)

# Create summary table
summary_data = {
    'Method': ['Original Features', 'Variance Filtering', 'LASSO', 'RFECV'],
    'Feature Count': [X.shape[1], len(selected_variance_features),
                      len(lasso_selected_features), len(rfecv_selected_features)],
    'Retention Rate (%)': [100,
                           len(selected_variance_features) / X.shape[1] * 100,
                           len(lasso_selected_features) / X.shape[1] * 100,
                           len(rfecv_selected_features) / X.shape[1] * 100]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save results
summary_df.to_csv(os.path.join(output_dir, 'Feature_Selection_Summary.csv'), index=False)

# Save selected feature lists
feature_selection_results = {
    'variance_selected': selected_variance_features,
    'lasso_selected': lasso_selected_features,
    'rfecv_selected': rfecv_selected_features
}

# Find common features selected by all methods
common_features = set(selected_variance_features) & set(lasso_selected_features) & set(rfecv_selected_features)
print(f"\n📊 Features common to all three methods: {len(common_features)}")
if len(common_features) > 0:
    print("Common features:")
    for feat in sorted(list(common_features))[:10]:
        print(f"  {feat}")
    if len(common_features) > 10:
        print(f"  ... and {len(common_features) - 10} more")

# Plot Venn diagram comparing three methods
try:
    from matplotlib_venn import venn3

    plt.figure(figsize=(10, 8))
    venn3([set(selected_variance_features),
           set(lasso_selected_features),
           set(rfecv_selected_features)],
          ('Variance Filtering', 'LASSO', 'RFECV'))
    plt.title('Venn Diagram of Three Feature Selection Methods', fontsize=14, fontweight='bold')
    # plt.savefig(os.path.join(output_dir, 'Feature_Selection_Venn.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'Feature_Selection_Venn.tif'), dpi=300, bbox_inches='tight')
    plt.show()
except ImportError:
    print("\nTip: Install matplotlib-venn to plot Venn diagram: pip install matplotlib-venn")

# ==========================================
# 7. Save Final Selected Features
# ==========================================
# Choose final feature set (using RFECV as an example)
final_selected_features = rfecv_selected_features
X_final = X[final_selected_features]

print(f"\n✅ Final selected features count: {X_final.shape[1]}")
print(f"✅ Final feature set saved for subsequent modeling")

# Save final feature list
with open(os.path.join(output_dir, 'Final_Selected_Features.txt'), 'w') as f:
    f.write(f"Final Selected Features (n={len(final_selected_features)}):\n")
    for i, feat in enumerate(final_selected_features, 1):
        f.write(f"{i}. {feat}\n")

print(f"\n✅ All results saved to: {output_dir}")
print("=" * 70)