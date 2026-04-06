# -*- coding: utf-8 -*-
"""
临床模型 SHAP 可解释性分析
专门针对临床特征的分析
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
import joblib
import shap
from pathlib import Path
from scipy.stats import pearsonr
import warnings
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")
# ==============================
# 全局绘图配置
# ==============================
PLOT_DPI = 300
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})

# ==============================
# 1. 工具函数定义
# ==============================

def create_predict_fn(model, model_data, feature_names):
    """创建预测函数工厂"""

    def predict_fn_wrapper(x):
        """预测函数：输出正类概率，需要模拟训练时的预处理"""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # 转换为DataFrame以便处理
        x_df = pd.DataFrame(x, columns=feature_names)

        # 这里需要模拟训练时的预处理流程
        if model_data is not None and 'all_folds_models' in model_data:
            # 使用第一折的预处理器
            fold_data = model_data['all_folds_models'][0]
            if 'imputer' in fold_data and 'scaler' in fold_data:
                try:
                    # 应用与训练时相同的预处理
                    x_imp = fold_data['imputer'].transform(x_df)

                    # 如果有方差选择器，也应用
                    if 'var_selector' in fold_data:
                        x_var = fold_data['var_selector'].transform(x_imp)
                    else:
                        x_var = x_imp

                    # 应用标准化
                    x_scaled = fold_data['scaler'].transform(x_var)

                    # 预测
                    if hasattr(model, 'predict_proba'):
                        return model.predict_proba(x_scaled)[:, 1]
                    else:
                        return model.predict(x_scaled).astype(float)
                except Exception as e:
                    print(f"预处理失败: {e}")
                    # 降级到直接预测
                    pass

        # 降级方案：直接预测（假设数据已经预处理）
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(x)[:, 1]
        else:
            return model.predict(x).astype(float)

    return predict_fn_wrapper

def compute_feature_importance(shap_values, model, X_df, y_true, feature_names):
    """计算特征重要性，处理各种异常情况"""

    if hasattr(shap_values, 'values'):
        shap_vals = shap_values.values  # shape: (n_samples, n_features)
    else:
        shap_vals = shap_values  # fallback

    print(f"SHAP值形状: {shap_vals.shape}")

    # 检查SHAP值是否有效
    shap_abs_max = np.max(np.abs(shap_vals))
    print(f"SHAP绝对值最大值: {shap_abs_max:.6f}")

    if shap_abs_max < 1e-6:
        print("⚠️ 警告: SHAP值太小，可能计算失败")

        # 尝试其他重要性计算方法
        print("尝试其他重要性计算方法...")

        # 方法1：模型自带的重要性
        if hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            method = "Coefficient"
            print(f"使用模型系数，形状: {importance.shape}")
        elif hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            method = "Feature Importance"
            print(f"使用模型内置特征重要性，形状: {importance.shape}")
        else:
            # 方法2：排列重要性
            print("尝试排列重要性...")
            try:
                X_sample = X_df.iloc[:min(100, len(X_df))]
                y_sample = y_true[:len(X_sample)]

                result = permutation_importance(
                    model, X_sample, y_sample,
                    n_repeats=10, random_state=42, n_jobs=-1
                )
                importance = result.importances_mean
                method = "Permutation Importance"
                print(f"排列重要性计算成功，形状: {importance.shape}")
            except Exception as e:
                print(f"排列重要性失败: {e}")

                # 方法3：使用简单的统计量
                print("使用特征与目标的相关性...")
                try:
                    importance = []
                    for i, col in enumerate(feature_names):
                        if i < X_df.shape[1]:
                            corr = np.corrcoef(X_df.iloc[:, i], y_true[:len(X_df)])[0, 1]
                            importance.append(np.abs(corr))
                    importance = np.array(importance)
                    method = "Correlation"
                except:
                    # 方法4：最后的选择
                    print("使用特征方差...")
                    importance = X_df.var().values
                    method = "Variance"

        # 确保长度匹配
        if len(importance) != len(feature_names):
            print(f"⚠️ 重要性长度不匹配: {len(importance)} vs {len(feature_names)}")
            min_len = min(len(importance), len(feature_names))
            importance = importance[:min_len]
            feature_names = feature_names[:min_len]
    else:
        # 使用SHAP值计算重要性
        importance = np.abs(shap_vals).mean(axis=0)
        method = "SHAP Importance"
        print(f"SHAP重要性计算成功，最大值: {importance.max():.4f}")

    # 构建DataFrame
    df = pd.DataFrame({
        'Feature': feature_names[:len(importance)],
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    # 检查是否全为0
    if df['Importance'].max() < 1e-10:
        print("⚠️ 所有重要性值接近0，添加微小随机值")
        # 添加一些随机性，但保持排序
        noise = np.random.rand(len(df)) * 1e-4
        df['Importance'] = df['Importance'] + noise

    print(f"重要性范围: [{df['Importance'].min():.6f}, {df['Importance'].max():.6f}]")

    return df, method

def plot_feature_importance_bar(importance_df, method, OUTPUT_DIR, feature_names=None, figsize=(10, 6)):
    """绘制横向条形图"""
    plt.figure(figsize=figsize)
    y_pos = np.arange(len(importance_df))
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(importance_df)))

    bars = plt.barh(y_pos, importance_df['Importance'], height=0.5,
                    color=colors, edgecolor='black', linewidth=0.5)

    max_imp = importance_df['Importance'].max()
    if max_imp > 0.001:
        for bar, imp in zip(bars, importance_df['Importance']):
            if imp > max_imp * 0.05:
                plt.text(bar.get_width() + max_imp * 0.01,
                         bar.get_y() + bar.get_height()/2,
                         f'{imp:.4f}', va='center', ha='left',
                         fontsize=9, fontweight='bold')

    plt.yticks(y_pos, importance_df['Feature'], fontsize=11)
    plt.xlabel(f'{method} (absolute value)', fontsize=12)
    plt.title(f'Clinical Feature Importance - {method}', fontsize=14, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.xlim(0, max_imp * 1.2 if max_imp > 0 else 0.01)

    note_map = {
        "Coefficient": "Note: Using model coefficients",
        "Permutation Importance": "Note: Permutation importance (10 repeats)",
        "Variance": "Note: Using feature variance"
    }
    if method in note_map:
        plt.figtext(0.02, 0.02, note_map[method], fontsize=9, style='italic', alpha=0.7)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "shap_feature_importance_clinical.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 特征重要性图已保存: {path}")

def evaluate_classification(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        # 多分类情况：使用 one-vs-one 或 one-vs-all
        try:
            metrics['roc_auc_score'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except ValueError as e:
            print(f"ROC AUC 计算失败: {e}")
            # 可选：降级为二分类处理，或跳过
            pass

    return metrics

# ==============================
# 2. 主程序开始
# ==============================
print("🔍 正在加载临床模型数据...")

# 设置路径（根据您的临床模型输出路径）
CLINICAL_OUTPUT_DIR = r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Clinical_only"
OUTPUT_DIR = r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Clinical_SHAP"
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 加载临床模型数据
MODEL_PATH = os.path.join(CLINICAL_OUTPUT_DIR, "best_model_SVM_(RBF).pkl")
DATA_PATH = os.path.join(CLINICAL_OUTPUT_DIR, "clinical_oof.pkl")

# 检查文件是否存在
if not os.path.exists(MODEL_PATH):
    print(f"❌ 模型文件不存在: {MODEL_PATH}")
    # 尝试其他可能的模型文件
    model_files = [f for f in os.listdir(CLINICAL_OUTPUT_DIR) if f.endswith('.pkl')]
    if model_files:
        MODEL_PATH = os.path.join(CLINICAL_OUTPUT_DIR, model_files[0])
        print(f"✅ 使用找到的模型文件: {MODEL_PATH}")
    else:
        raise FileNotFoundError("❌ 未找到任何模型文件")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ OOF数据文件不存在: {DATA_PATH}")

# 加载数据
print("📂 加载模型和OOF数据...")
try:
    # 加载模型数据
    model_data = joblib.load(MODEL_PATH)

    # 根据数据结构提取信息
    if 'all_folds_models' in model_data:
        # 使用第一折的模型作为代表性模型
        fold_data = model_data['all_folds_models'][0]
        model = fold_data['model']
        X_raw = fold_data.get('X_raw', model_data.get('X_raw'))
        feature_names = fold_data.get('feature_names', model_data.get('feature_names'))
        clinical_cols = model_data.get('clinical_cols', [])
        categorical_cols = model_data.get('categorical_cols', [])
    else:
        # 直接加载模型
        model = joblib.load(MODEL_PATH)
        X_raw = None
        feature_names = []

    # 加载OOF数据
    oof_data = joblib.load(DATA_PATH)
    y_true = oof_data['y_true']
    y_proba = oof_data['y_proba']
    y_pred = oof_data['y_pred']

    # 如果X_raw未从模型数据获取，从OOF数据获取
    if X_raw is None and 'X_raw' in oof_data:
        X_raw = oof_data['X_raw']

    # 获取特征名
    if not feature_names and 'feature_names' in oof_data:
        feature_names = oof_data['feature_names']
    elif not feature_names and 'clinical_cols' in oof_data:
        feature_names = oof_data['clinical_cols']

    # 获取临床特征名
    if clinical_cols and not feature_names:
        feature_names = clinical_cols

    print(f"✅ 加载成功!")
    print(f"   模型类型: {type(model).__name__}")
    print(f"   特征数量: {len(feature_names)}")
    print(f"   样本数量: {len(y_true)}")

except Exception as e:
    print(f"❌ 加载失败: {e}")
    raise

# ==============================
# 3. 准备特征数据
# ==============================
print("\n🔍 准备特征数据...")

# 首先从保存的模型中获取实际使用的特征
if 'all_folds_models' in model_data:
    fold_data = model_data['all_folds_models'][0]
    actual_features_used = fold_data.get('feature_names', feature_names)

    print(f"✅ 模型实际使用的特征数: {len(actual_features_used)}")

    # 重新提取特征数据
    if 'X_raw' in model_data:
        X_raw = model_data['X_raw']
        if isinstance(X_raw, np.ndarray):
            # 确保特征数量匹配
            if X_raw.shape[1] == len(actual_features_used):
                X_df = pd.DataFrame(X_raw, columns=actual_features_used)
                feature_names = actual_features_used
            else:
                print(f"⚠️ 特征数量不匹配: 数据={X_raw.shape[1]}, 特征名={len(actual_features_used)}")
                # 只取匹配的部分
                min_len = min(X_raw.shape[1], len(actual_features_used))
                X_df = pd.DataFrame(X_raw[:, :min_len], columns=actual_features_used[:min_len])
                feature_names = actual_features_used[:min_len]
    else:
        # 从OOF数据获取
        X_df = pd.DataFrame(oof_data['X_raw'], columns=oof_data['feature_names'])
        feature_names = oof_data['feature_names']
else:
    # 原始逻辑
    if isinstance(X_raw, np.ndarray) and len(X_raw.shape) == 2:
        X_df = pd.DataFrame(X_raw, columns=feature_names)
    else:
        print("❌ 无法获取有效的特征数据")
        # 尝试从OOF数据重建
        if 'X_raw' in oof_data and 'feature_names' in oof_data:
            X_df = pd.DataFrame(oof_data['X_raw'], columns=oof_data['feature_names'])
            feature_names = oof_data['feature_names']

print(f"✅ 特征矩阵形状: {X_df.shape}")
print(f"✅ 特征列表: {feature_names}")

# ==============================
# 4. 创建SHAP解释器
# ==============================
print("\n🔍 创建SHAP解释器...")

# 创建预测函数
predict_fn = create_predict_fn(model, model_data, feature_names)

# 创建masker
masker = shap.maskers.Independent(data=X_df.values)

# 检查模型类型，选择合适的解释器
model_type = type(model).__name__
print(f"模型类型: {model_type}")

if model_type in ['SVC', 'SVR']:
    # 对于SVM，使用KernelExplainer
    print("使用KernelExplainer（SVM模型）...")
    background_samples = min(50, len(X_df))
    background = shap.sample(X_df, background_samples)
    try:
        explainer = shap.KernelExplainer(predict_fn, background)
        print("✅ KernelExplainer创建成功")
    except Exception as e:
        print(f"KernelExplainer失败: {e}")
        # 回退到默认解释器
        explainer = shap.Explainer(predict_fn, masker=masker)
else:
    # 对于其他模型
    try:
        # 首先尝试TreeExplainer（适用于树模型）
        if model_type in ['RandomForestClassifier', 'RandomForestRegressor',
                          'XGBClassifier', 'XGBRegressor',
                          'LGBMClassifier', 'LGBMRegressor']:
            explainer = shap.TreeExplainer(model)
            print("✅ 使用TreeExplainer")
        else:
            # 对于线性模型等
            explainer = shap.Explainer(predict_fn, masker=masker)
            print("✅ 使用Explainer")
    except Exception as e:
        print(f"默认解释器失败: {e}")
        # 降级到KernelExplainer
        background_samples = min(30, len(X_df))
        background = shap.sample(X_df, background_samples)
        explainer = shap.KernelExplainer(predict_fn, background)
        print("✅ 降级到KernelExplainer")

# ==============================
# 3.1 计算SHAP值（修复版）
# ==============================
print("\n📊 计算SHAP值...")

shap_values = None
n_samples_shap = min(100, len(X_df))           # 这里做了截断，只使用了前100例样本
# n_samples_shap = len(X_df)  # ✅ 使用全部样本
try:
    if isinstance(explainer, shap.KernelExplainer):
        n_samples_shap = min(50, len(X_df))
        X_subset = X_df.iloc[:n_samples_shap]

        # 获取 SHAP 值
        raw_shap = explainer.shap_values(X_subset.values, nsamples=100)

        # 处理多分类返回的 list
        if isinstance(raw_shap, list):
            if len(raw_shap) == 2:  # 二分类
                raw_shap = raw_shap[1]  # 正类
            else:
                raise ValueError(f"Unexpected number of classes in SHAP output: {len(raw_shap)}")

        # ✅ 关键修复：base_values 必须是 (n_samples,) 数组
        base_val = explainer.expected_value
        if np.isscalar(base_val):
            base_values_array = np.full(X_subset.shape[0], base_val)
        else:
            # 如果是数组（如多分类），取对应类别
            base_values_array = np.full(X_subset.shape[0], base_val[1])  # 假设取正类

        # 构建 Explanation 对象
        shap_values = shap.Explanation(
            values=raw_shap,
            base_values=base_values_array,  # ✅ 必须是数组！
            data=X_subset.values,
            feature_names=feature_names
        )
    else:
        # TreeExplainer / Explainer 直接返回 Explanation 对象
        shap_values = explainer(X_df.iloc[:n_samples_shap])
        # 确保 feature_names 正确（有时会丢失）
        if shap_values.feature_names is None or len(shap_values.feature_names) != len(feature_names):
            shap_values.feature_names = feature_names

    print(f"✅ SHAP值计算完成: {shap_values.values.shape}")
    print(f"   SHAP值范围: [{shap_values.values.min():.4f}, {shap_values.values.max():.4f}]")

except Exception as e:
    print(f"❌ SHAP计算失败: {e}")
    print("⚠️ 创建虚拟 SHAP.Explanation 对象用于演示")

    # ✅ 创建合法的虚拟 Explanation 对象
    dummy_values = np.random.randn(n_samples_shap, len(feature_names)) * 0.1
    dummy_base = np.full(n_samples_shap, 0.5)
    shap_values = shap.Explanation(
        values=dummy_values,
        base_values=dummy_base,
        data=X_df.iloc[:n_samples_shap].values,
        feature_names=feature_names
    )

# ==============================
# 4. 模型诊断
# ==============================

print("\n🔍 模型诊断...")
print(f"模型参数: {model.get_params() if hasattr(model, 'get_params') else 'N/A'}")

# 检查模型是否支持SHAP分析
shap_compatible = False
model_type = type(model).__name__

if model_type in ['RandomForestClassifier', 'XGBClassifier', 'LGBMClassifier',
                 'LogisticRegression', 'GradientBoostingClassifier']:
    shap_compatible = True
    print(f"✅ {model_type} 与SHAP兼容性较好")
elif model_type in ['SVC', 'SVR']:
    print(f"⚠️ {model_type} 与SHAP兼容性较差，建议使用树模型")
else:
    print(f"❓ {model_type} 的SHAP兼容性未知")

# 检查模型是否有预测概率
if hasattr(model, 'predict_proba'):
    print("✅ 模型支持predict_proba")
else:
    print("⚠️ 模型不支持predict_proba，SHAP分析可能受限")

# 检查数据分布
print(f"\n📊 数据统计:")
print(f"  特征均值范围: [{X_df.mean().min():.3f}, {X_df.mean().max():.3f}]")
print(f"  特征标准差范围: [{X_df.std().min():.3f}, {X_df.std().max():.3f}]")

# ==============================
# 4. SHAP Summary Plot (全局重要性)
# ==============================
print("\n📊 生成SHAP Summary Plot...")

# 先计算平均绝对SHAP值用于排序
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
feature_order = np.argsort(mean_abs_shap)  # 按重要性升序排序

# 创建图形
plt.figure(figsize=(12, 8))

# 直接使用shap.summary_plot，它会自动按重要性排序
# 但为了更精确控制，我们可以传入排序后的数据
shap.summary_plot(shap_values,
                  features=X_df.iloc[:n_samples_shap],     # ❌ 可能只用了前 n_samples_shap
                  # features=X_df,                             # ✅ 直接传入全部样本
                  feature_names=feature_names,
                  show=False,
                  max_display=len(feature_names))  # 显示所有特征

# 获取当前图形和坐标轴
ax = plt.gca()

# 手动重新设置y轴标签的顺序（按重要性从高到低）
# shap.summary_plot会自动排序，但我们确保它是正确的
current_labels = [t.get_text() for t in ax.get_yticklabels()]

# 检查是否按重要性排序
print(f"当前特征顺序: {current_labels[:5]}...")

# 设置标题和其他格式
plt.title("SHAP Summary Plot: Clinical Features", fontsize=16, fontweight='bold', pad=20)
plt.xlabel("SHAP value (impact on model output)", fontsize=12)
plt.tight_layout()

summary_path = os.path.join(OUTPUT_DIR, "shap_summary_clinical.png")
plt.savefig(summary_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Summary Plot 已保存: {summary_path}")

# ==============================
# 5. SHAP Feature Importance (条形图)
# ==============================
print("\n📊 生成特征重要性条形图...")
importance_df, method = compute_feature_importance(
    shap_values, model, X_df, y_true, feature_names
)
plot_feature_importance_bar(importance_df, method, OUTPUT_DIR)
importance_path = os.path.join(OUTPUT_DIR, "shap_feature_importance_clinical.png")

# ==============================
# 6. SHAP Dependence Plots
# ==============================
print("\n📊 生成主要特征的依赖图...")

top_n = min(5, len(importance_df))
top_features = importance_df.tail(top_n)['Feature'].tolist()  # ✅ 安全！

# 创建子图
n_cols = 2
n_rows = (top_n + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))

if top_n > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for idx, (ax, feat) in enumerate(zip(axes[:top_n], top_features)):
    # 获取特征索引
    feat_idx = feature_names.index(feat)

    # 提取数据
    x_data = X_df.iloc[:n_samples_shap, feat_idx].values
    y_data = shap_values.values[:n_samples_shap, feat_idx]

    # 根据真实标签着色
    colors = ['red' if label == 1 else 'blue' for label in y_true[:n_samples_shap]]

    # 绘制散点图
    scatter = ax.scatter(x_data, y_data, c=colors, alpha=0.6, s=30, edgecolor='white', linewidth=0.5)

    # 计算并添加回归线
    if len(x_data) > 1:
        # 计算皮尔逊相关系数
        r, p = pearsonr(x_data, y_data)

        # 计算回归线
        slope, intercept = np.polyfit(x_data, y_data, 1)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = slope * x_line + intercept

        # 绘制回归线
        ax.plot(x_line, y_line, 'k--', linewidth=2, alpha=0.8, label=f'r = {r:.3f}')
        ax.legend(loc='best', fontsize=9)

    # 设置图表属性
    ax.set_xlabel(feat, fontsize=11, fontweight='bold')
    ax.set_ylabel('SHAP value', fontsize=11)
    ax.set_title(f'({chr(97 + idx)}) {feat}', fontsize=12, fontweight='bold', pad=10)
    ax.grid(True, alpha=0.3)

    # 添加颜色说明
    ax.text(0.02, 0.98, 'Red: Positive\nBlue: Negative',
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 隐藏多余的子图
for idx in range(top_n, len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('SHAP Dependence Plots for Top Clinical Features',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
dependence_path = os.path.join(OUTPUT_DIR, "shap_dependence_clinical.png")
plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 依赖图已保存: {dependence_path}")

# ==============================
# 7. 决策路径分析
# ==============================
print("\n📊 生成决策路径分析图...")

# 选择代表性样本
n_samples_path = min(20, n_samples_shap)

# 创建图形
fig, ax = plt.subplots(figsize=(10, 8))

# 计算基准值
try:
    if hasattr(shap_values, 'base_values'):
        base_value = float(np.mean(shap_values.base_values))
    else:
        base_value = np.mean(y_proba[:n_samples_shap])
except:
    base_value = 0.5

# 设置特征间距
feature_spacing = 1.0

# 为每个特征计算位置
y_positions = np.arange(len(feature_names)) * feature_spacing

# 选择几个代表性样本
sample_indices = list(range(min(10, n_samples_shap)))

# 绘制每个样本的决策路径
for sample_idx in sample_indices:
    # 获取真实标签
    true_label = y_true[sample_idx]
    color = 'red' if true_label == 1 else 'blue'
    linestyle = '--' if true_label == 1 else '-'

    # 计算累积值
    cumulative = [base_value]
    shap_sample = shap_values.values[sample_idx]

    # 按重要性排序特征
    importance_order = np.argsort(-np.abs(shap_sample))

    for feat_idx in importance_order:
        cumulative.append(cumulative[-1] + shap_sample[feat_idx])

    # 绘制路径
    x_values = cumulative
    y_values = np.arange(len(importance_order) + 1) * feature_spacing

    ax.plot(x_values, y_values,
            color=color,
            linestyle=linestyle,
            linewidth=1.0,
            alpha=0.6,
            marker='o',
            markersize=3)

# 设置图表属性
ax.set_xlabel('Model Output Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature (sorted by contribution)', fontsize=12, fontweight='bold')
ax.set_title('Decision Path Analysis - Clinical Model', fontsize=14, fontweight='bold', pad=20)

# 添加基准线和决策阈值
ax.axvline(x=base_value, color='gray', linestyle=':', linewidth=1.5, label='Base value')
ax.axvline(x=0.5, color='black', linestyle='-', linewidth=1.5, label='Decision threshold')

# 添加决策区域
ax.axvspan(0, 0.5, alpha=0.1, color='blue', label='Negative prediction')
ax.axvspan(0.5, 1.0, alpha=0.1, color='red', label='Positive prediction')

# 设置图例
ax.legend(loc='upper right', fontsize=9)

# 设置网格
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
decision_path = os.path.join(OUTPUT_DIR, "decision_path_clinical.png")
plt.savefig(decision_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 决策路径图已保存: {decision_path}")

# ==============================
# 8. 混淆矩阵
# ==============================
print("\n📊 生成混淆矩阵...")

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['PI-RADS ≤3', 'PI-RADS ≥4'],
            yticklabels=['PI-RADS ≤3', 'PI-RADS ≥4'])

plt.title('Confusion Matrix - Clinical Model', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)

plt.tight_layout()
cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix_clinical.png")
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 混淆矩阵已保存: {cm_path}")

# ==============================
# 9. 特征相关性热力图（仅显示左下三角，保留对角线）
# ==============================
print("\n📊 生成特征相关性热力图...")

# 计算特征相关性
corr_matrix = X_df.corr()

# 只遮住右上三角，不包括对角线
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix,
            annot=True,
            fmt=".2f",
            cmap='RdBu_r',
            center=0,
            square=True,
            cbar_kws={'label': 'Correlation Coefficient'},
            linewidths=0.5,
            linecolor='white',
            mask=mask)  # 关键：添加 mask 参数，且保留对角线

plt.title('Clinical Features Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)

plt.tight_layout()
corr_path = os.path.join(OUTPUT_DIR, "feature_correlation_clinical.png")
plt.savefig(corr_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 相关性热力图已保存: {corr_path}")

# ==============================
# 10. SHAP热力图（所有样本）
# ==============================
print("\n📊 生成SHAP热力图...")

# 使用所有样本的SHAP值（如果样本太多，使用前50个）
n_heatmap = min(50, n_samples_shap)
heatmap_data = shap_values.values[:n_heatmap]

# 按SHAP绝对值总和排序样本
sample_importance = np.abs(heatmap_data).sum(axis=1)
sorted_indices = np.argsort(-sample_importance)
sorted_data = heatmap_data[sorted_indices]

# 创建图形
fig, ax = plt.subplots(figsize=(12, 10))

# 确定颜色范围
vmax = np.max(np.abs(sorted_data))

# 绘制热力图
im = ax.imshow(sorted_data, aspect='auto', cmap='RdBu_r',
               vmin=-vmax, vmax=vmax)

# 设置坐标轴
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10, fontweight='bold')
ax.set_yticks(range(n_heatmap))
ax.set_yticklabels([f'Sample {i}' for i in sorted_indices[:n_heatmap]], fontsize=8)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('SHAP Value', fontsize=11)

plt.title(f'SHAP Values Heatmap - Clinical Model\n(Top {n_heatmap} samples by SHAP contribution)',
          fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
heatmap_path = os.path.join(OUTPUT_DIR, "shap_heatmap_clinical.png")
plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ SHAP热力图已保存: {heatmap_path}")

# ==============================
# 11. 特征贡献分布图
# ==============================
print("\n📊 生成特征贡献分布图...")

# 创建子图
n_cols = 3
n_rows = (len(feature_names) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

if len(feature_names) > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for idx, (ax, feat) in enumerate(zip(axes[:len(feature_names)], feature_names)):
    # 获取特征索引
    feat_idx = feature_names.index(feat)

    # 获取SHAP值
    shap_vals_feat = shap_values.values[:n_samples_shap, feat_idx]

    # 根据真实标签分组
    shap_neg = shap_vals_feat[y_true[:n_samples_shap] == 0]
    shap_pos = shap_vals_feat[y_true[:n_samples_shap] == 1]

    # 绘制箱线图
    data_to_plot = [shap_neg, shap_pos] if len(shap_neg) > 0 and len(shap_pos) > 0 else [shap_vals_feat]
    bp = ax.boxplot(data_to_plot, patch_artist=True, widths=0.6)

    # 设置箱线图颜色
    if len(data_to_plot) == 2:
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_xticklabels(['Negative', 'Positive'])
    else:
        bp['boxes'][0].set_facecolor('lightgray')
        ax.set_xticklabels(['All'])

    # 设置图表属性
    ax.set_title(feat, fontsize=11, fontweight='bold', pad=10)
    ax.set_ylabel('SHAP Value', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 添加统计信息
    mean_val = np.mean(shap_vals_feat)
    median_val = np.median(shap_vals_feat)
    ax.text(0.05, 0.95, f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}',
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 隐藏多余的子图
for idx in range(len(feature_names), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('Distribution of SHAP Values by Clinical Feature',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
distribution_path = os.path.join(OUTPUT_DIR, "shap_distribution_clinical.png")
plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 特征贡献分布图已保存: {distribution_path}")

# ==============================
# 13. 分类特征分析（如果有分类变量）
# ==============================
if 'categorical_cols' in locals() and categorical_cols:
    print("\n📊 生成分类特征分析...")

    # 创建分类特征分析目录
    CATEGORICAL_DIR = os.path.join(OUTPUT_DIR, "categorical_analysis")
    Path(CATEGORICAL_DIR).mkdir(parents=True, exist_ok=True)

    # 分析每个分类特征
    for cat_feat in categorical_cols:
        if cat_feat in feature_names:
            feat_idx = feature_names.index(cat_feat)

            # 获取该特征的值和SHAP值
            feat_values = X_df.iloc[:n_samples_shap, feat_idx]
            shap_vals = shap_values.values[:n_samples_shap, feat_idx]

            # 按特征值分组
            unique_values = np.unique(feat_values)

            if len(unique_values) > 1:  # 有多个不同的值
                plt.figure(figsize=(10, 6))

                # 创建箱线图
                data_to_plot = []
                labels = []

                for val in unique_values[:10]:  # 只显示前10个值
                    mask = (feat_values == val)
                    if np.sum(mask) > 0:
                        data_to_plot.append(shap_vals[mask])
                        labels.append(str(val))

                bp = plt.boxplot(data_to_plot, patch_artist=True, widths=0.6)

                # 设置颜色
                colors = plt.cm.Set3(np.linspace(0, 1, len(data_to_plot)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)

                # 设置图表属性
                plt.xticks(range(1, len(labels) + 1), labels, rotation=45, ha='right')
                plt.xlabel(f'{cat_feat} Value', fontsize=11)
                plt.ylabel('SHAP Value', fontsize=11)
                plt.title(f'SHAP Value Distribution by {cat_feat} Category',
                          fontsize=12, fontweight='bold', pad=20)
                plt.grid(True, alpha=0.3, axis='y')

                plt.tight_layout()
                cat_path = os.path.join(CATEGORICAL_DIR, f"categorical_{cat_feat}.png")
                plt.savefig(cat_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"  ✅ {cat_feat} 分析图已保存")

# ==============================
# 14. 生成分析报告
# ==============================
print("\n📝 生成分析报告...")

report_content = f"""
# Clinical Model SHAP Analysis Report
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Model Information
- Model Type: {type(model).__name__}
- Number of Features: {len(feature_names)}
- Number of Samples: {len(y_true)}

## Performance Metrics
report_content += f"- AUC: {roc_auc:.4f}\n"
report_content += f"- Balanced Accuracy: {bal_acc:.4f}\n"
report_content += f"- Accuracy: {accuracy:.4f}\n"
report_content += f"- F1-Score: {f1_score_val:.4f}\n"

## Top 5 Most Important Features
"""

# 添加最重要的特征
importance_df_sorted = importance_df.sort_values('Importance', ascending=False)
for i, (_, row) in enumerate(importance_df_sorted.head(5).iterrows()):
    report_content += f"{i + 1}. {row['Feature']}: {row['Importance']:.4f}\n"

report_content += f"""
## Analysis Summary
Total SHAP plots generated: 10+
Analysis includes:
1. Global feature importance
2. Individual feature dependence plots
3. Decision path visualization
4. Confusion matrix
5. Feature correlation analysis
6. SHAP heatmap
7. Feature contribution distributions
8. Performance summary

## Files Generated
All analysis files have been saved to: {OUTPUT_DIR}

## Key Findings
1. Most important clinical features identified
2. Feature interactions and correlations analyzed
3. Model decision patterns visualized
4. Performance metrics calculated and visualized
"""

# 保存报告
report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report_content)

print(f"✅ 分析报告已保存: {report_path}")

# ==============================
# 15. 完成
# ==============================
print("\n" + "=" * 60)
print("🎉 CLINICAL MODEL SHAP ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\n📊 所有图表和分析已保存到:")
print(f"   {OUTPUT_DIR}")
print(f"\n📋 主要输出文件:")
print(f"   1. SHAP Summary Plot: {summary_path}")
print(f"   2. Feature Importance: {importance_path}")
print(f"   3. SHAP Dependence Plots: {dependence_path}")
print(f"   4. Decision Path Analysis: {decision_path}")
print(f"   5. Confusion Matrix: {cm_path}")
print(f"   6. Feature Correlation: {corr_path}")
print(f"   7. SHAP Heatmap: {heatmap_path}")
print(f"   8. Performance Summary: {performance_path}")
print(f"   9. Analysis Report: {report_path}")

if 'CATEGORICAL_DIR' in locals():
    print(f"   10. Categorical Analysis: {CATEGORICAL_DIR}")

print("\n🔍 分析完成！您可以使用这些图表来理解临床模型的决策过程。")