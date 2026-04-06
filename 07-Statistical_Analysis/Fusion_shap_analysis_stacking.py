
# -*- coding: utf-8 -*-
"""
增强版 SHAP 可解释性分析：Stacking LR 模型
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
import joblib
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list

plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")



# 设置路径
OUTPUT_DIR = r"K:\PCa_2026\Article\放射组学\图表\shap-Fusion"
MODEL_PATH = os.path.join(OUTPUT_DIR, r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\ML\3rd\LateFusion\stacking_lr_model.pkl")
DATA_PATH = os.path.join(OUTPUT_DIR, r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\ML\3rd\LateFusion\late_fusion_oof.pkl")

# 检查目录是否存在，如果不存在则创建
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 加载数据
print("🔍 正在加载融合结果...")
fusion_data = joblib.load(DATA_PATH)
probas = np.column_stack([
    fusion_data['Clinical_proba'],
    fusion_data['T2_proba'],
    fusion_data['dce_proba'],
    fusion_data['DWI_proba']
])
y_true = fusion_data['y_true']

# 加载模型
print("🔍 正在加载 Stacking LR 模型...")
model = joblib.load(MODEL_PATH)

# 定义预测函数：输出正类概率
predict_fn = lambda x: model.predict_proba(x)[:, 1]

# 创建 masker（关键！）
masker = shap.maskers.Independent(data=probas)

# 定义特征名
feature_names = ['Clinical', 'T2', 'DCE', 'DWI']

# 创建 SHAP 解释器
explainer = shap.Explainer(predict_fn, masker=masker)

# 计算 SHAP 值
shap_values = explainer(probas)

# 🔑 绑定特征名到 SHAP 对象（这是关键！）
shap_values.feature_names = feature_names

# ==============================
# 1. Summary Plot (全局重要性)
# ==============================
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, features=probas, feature_names=feature_names, show=False)
plt.title("SHAP Summary Plot: Conditional_Clinical_T2_dce/DWI")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ SHAP Summary Plot 已保存")

# ==============================
# 2. SHAP Feature Importance (Bar Plot with Gradient Colors)
# ==============================
# 获取特征重要性（按 |SHAP| 平均值）
shap_importance = np.abs(shap_values.values).mean(axis=0)
feature_names = ['Clinical', 'T2', 'DCE', 'DWI']  # 确保顺序正确

# 创建从深粉到浅粉的渐变色板
colors = [(0.894, 0.365, 0.573), (1, 0.8, 0.9)]  # 深粉 → 浅粉
pink_cmap = LinearSegmentedColormap.from_list("custom_pink", colors, N=len(feature_names))

y_pos = np.arange(len(feature_names)) * 0.5

# 手动绘制条形图
plt.figure(figsize=(8, 3))
bars = plt.barh(feature_names,
                shap_importance,
                height=0.4,
                color=pink_cmap(np.linspace(0, 1, len(feature_names))),
                edgecolor='black',
                linewidth=0.1)

# 添加数值标签（右侧）
for i, bar in enumerate(bars):
    width = bar.get_width()
    plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
             f'+{width:.4f}', va='center', ha='left', fontsize=10, color='red')

# 设置坐标轴
plt.xlabel('mean(|SHAP value|)', fontsize=12)
plt.title('SHAP Feature Importance', fontsize=14, fontweight='bold')
plt.xlim(0, max(shap_importance) * 1.1)
plt.gca().invert_yaxis()  # 使 Clinical 在最上方
plt.grid(True, axis='x', alpha=0.3)

# 调整布局并保存
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_feature_importance_gradient.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ SHAP Feature Importance (Gradient Colors) 已保存")


# ==============================
# 3. Bar Plot (特征重要性)
# ==============================
# 计算各特征SHAP值的绝对值均值
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

plt.figure(figsize=(8, 5))
sns.barplot(x=feature_names, y=mean_abs_shap)
plt.title("Mean Absolute SHAP Values by Feature")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar_plot_mean_abs.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Mean Absolute SHAP Bar Plot 已保存")

# ==============================
# 4. 增强版 SHAP Dependence Plots (仿论文风格)
# ==============================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
ensemble_proba = model.predict_proba(probas)[:, 1]

for i, name in enumerate(feature_names):
    ax = axes[i]

    x = probas[:, i]  # 特征值（概率）
    y = shap_values.values[:, i]  # SHAP 值

    # 计算皮尔逊相关系数
    r, p = pearsonr(x, y)

    # 颜色编码真实标签（红=阳性，蓝=阴性）
    # alpha为蓝点和红点的透明度
    scatter = ax.scatter(x[y_true == 1], y[y_true == 1], c='blue', alpha=1.0, s=20, label='Positive')
    scatter = ax.scatter(x[y_true == 0], y[y_true == 0], c='red', alpha=0.3, s=20, label='Negative')

    # 添加线性拟合趋势线
    slope, intercept = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', linewidth=1.5, label=f'r={r:.2f}')

    # 设置标题和标签
    ax.set_title(f"({chr(97 + i)}) {name}", fontsize=12, fontweight='bold')
    ax.set_xlabel(name, fontsize=10)
    ax.set_ylabel("SHAP Value", fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.legend(fontsize=8, frameon=True, fancybox=True, edgecolor='gray', facecolor='white', loc='upper left')

    # 调整坐标轴范围（可选）
    ax.set_xlim(x.min() - 0.02, x.max() + 0.02)
    ax.set_ylim(y.min() - 0.02, y.max() + 0.02)

# 添加总标题（根据您提供的图片内容）
plt.suptitle('SHAP Dependence Plots for Stacking Ensemble',
             fontsize=16, fontweight='bold', y=0.98)

# 调整子图间距，为总标题留出空间
plt.tight_layout(pad=1.0, rect=[0, 0, 1, 1])  # rect参数：左、下、右、上

plt.savefig(os.path.join(OUTPUT_DIR, "shap_dependence_combined_plots.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Enhanced SHAP Dependence Plots (Paper Style) 已保存")


# ==============================
# 5. Coefficients vs SHAP Importance (对比分析)
# ==============================
# 获取模型系数（假设是 LogisticRegression 或其子模型）
if hasattr(model, 'coef_'):
    coef = model.coef_[0]  # 提取第一层系数（正类）
else:
    raise AttributeError("Model does not have coef_ attribute. Please check if it's a linear model.")

# 计算 SHAP 值的均值绝对值（作为特征重要性）
shap_importance = np.abs(shap_values.values).mean(axis=0)

# 归一化系数和 SHAP 重要性
coef_norm = coef / np.max(np.abs(coef))
shap_norm = shap_importance / np.max(shap_importance)

# 计算皮尔逊相关系数
r_coef_shap, _ = pearsonr(coef_norm, shap_norm)

# 创建双子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 左图：条形图
x_pos = np.arange(len(feature_names))
width = 0.35

ax1.bar(x_pos - width/2, coef_norm, width, label='Model Coefficients', color='skyblue')
ax1.bar(x_pos + width/2, shap_norm, width, label='SHAP Importance', color='lightcoral')

ax1.set_xlabel('Normalized Importance')
ax1.set_ylabel('Normalized Feature Importance')
ax1.set_title('Feature Importance: Coefficients vs SHAP')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(feature_names, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：散点图
scatter = ax2.scatter(coef_norm, shap_norm, c='green', s=80, edgecolors='black', linewidth=0.5)

# 添加对角线（完美相关）
ax2.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Perfect Correlation')

# 添加文本标签
for i, name in enumerate(feature_names):
    ax2.annotate(name, (coef_norm[i], shap_norm[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# 设置右图标签
ax2.set_xlabel('Normalized Model Coefficients')
ax2.set_ylabel('Normalized SHAP Importance')
ax2.set_title(f'Coefficient vs SHAP Correlation\nCorrelation: {r_coef_shap:.3f}')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "coefficients_vs_shap.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Coefficients vs SHAP Importance Plot 已保存")

# ==============================
# 6. Confusion Matrix (Stacking Ensemble)
# ==============================

# 预测标签
y_pred = model.predict(probas)

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)

# 创建热力图（不自动标注）
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm, annot=False, cmap='Blues', cbar=True,
                 square=True, cbar_kws={'label': 'Count'})

# 手动添加计数和百分比（两行）
total = cm.sum()
percentages = cm.astype('float') / total

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        percentage = percentages[i, j]

        # 设置文本颜色：深色背景用白色，浅色用黑色
        text_color = 'white' if cm[i, j] > 150 else 'black'

        # 绘制主文本（计数）
        ax.text(j + 0.5, i + 0.5, str(count), ha='center', va='center',
                color=text_color, fontsize=16, weight='bold')

        # 绘制副文本（百分比）——在下方偏移一点
        ax.text(j + 0.5, i + 0.6, f"\n({percentage:.1%})", ha='center', va='center',
                color='black', fontsize=14)

plt.title("Confusion Matrix - Stacking Ensemble\n(5-Fold Cross-Validation Combined)",
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel("Predicted Label", fontsize=11)
plt.ylabel("True Label", fontsize=11)
plt.xticks([0, 1], ['Low Suspicion', 'High Suspicion'], fontsize=10)
plt.yticks([0, 1], ['Low Suspicion', 'High Suspicion'], fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()
print("✅ Confusion Matrix 已保存")


# ==============================
# 8. 决策路径分析 (Decision Path Plot) - 优化版
# ==============================
print("📊 正在生成决策路径分析图（优化版）...")

# 提取数据
shap_vals = shap_values.values  # shape: (n_samples, n_features)
pred_proba = model.predict_proba(probas)[:, 1]
y_true = fusion_data['y_true']

# 获取特征名
feature_names = ['Clinical', 'T2', 'DCE', 'DWI']

# 筛选高质量样本（预测正确 + 高置信度）
correct_mask = (model.predict(probas) == y_true)
high_confidence_mask = (pred_proba <= 0.2) | (pred_proba >= 0.8)
good_indices = np.where(correct_mask & high_confidence_mask)[0]

# 取前30个作为展示
n_samples_to_show = 30
selected_indices = good_indices[:n_samples_to_show] if len(good_indices) >= n_samples_to_show else good_indices

if len(selected_indices) < n_samples_to_show:
    print(f"⚠️ 只有 {len(selected_indices)} 个高质量样本，使用所有高质量样本")
    if len(selected_indices) < 5:
        print("⚠️ 高质量样本太少，使用随机样本补充")
        all_indices = np.arange(len(y_true))
        remaining = np.setdiff1d(all_indices, selected_indices)
        additional = np.random.choice(remaining, min(n_samples_to_show - len(selected_indices), len(remaining)),
                                      replace=False)
        selected_indices = np.concatenate([selected_indices, additional])

print(f"选择的样本数: {len(selected_indices)}")

# 计算基准值
try:
    if hasattr(shap_values, 'base_values'):
        base_value = float(np.mean(shap_values.base_values))
    else:
        base_value = np.mean(pred_proba)
except:
    base_value = 0.5

print(f"基准值 (base value): {base_value:.4f}")

# 创建图形
fig, ax = plt.subplots(figsize=(6, 8))

# 定义颜色：蓝色为阴性(0)，红色为阳性(1)
colors = ['blue' if y_true[i] == 0 else 'red' for i in selected_indices]

# 定义线型：实线为阴性，虚线为阳性
linestyles = ['-' if y_true[i] == 0 else '--' for i in selected_indices]

# 设置特征位置的偏移
feature_spacing = 0.8

# 绘制每条样本的决策路径
for idx, sample_idx in enumerate(selected_indices):
    # 计算累计SHAP值（从基准值开始）
    cumulative_values = [base_value]
    for j in range(len(feature_names)):
        cumulative_values.append(cumulative_values[-1] + shap_vals[sample_idx, j])

    # 绘制路径线
    x_values = cumulative_values
    y_values = np.arange(len(feature_names) + 1) * feature_spacing

    # 绘制路径线（使用倒三角标记）
    ax.plot(x_values, y_values,
            color=colors[idx],
            linestyle=linestyles[idx],
            linewidth=1.0,
            alpha=0.6,
            marker='v',  # 倒三角标记
            markersize=4,
            markevery=1,
            markerfacecolor=colors[idx],
            markeredgecolor='white',
            markeredgewidth=0.5)

    # 在终点添加最终预测值标记
    final_pred = pred_proba[sample_idx]
    ax.plot(final_pred, y_values[-1],
            marker='o',
            color=colors[idx],
            markersize=6,
            markeredgecolor='black',
            markeredgewidth=0.5,
            zorder=5)

# 设置坐标轴
ax.set_xlabel('Model output value', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title('Decision path analysis - Stacking Ensemble', fontsize=16, fontweight='bold', pad=20)

# 设置y轴刻度（特征名）
y_tick_positions = np.arange(len(feature_names)) * feature_spacing + (feature_spacing / 2)
ax.set_yticks(y_tick_positions)
ax.set_yticklabels(feature_names, fontsize=11, fontweight='bold')
ax.invert_yaxis()  # 倒序显示，Clinical在最上面

# 添加基准值线
ax.axvline(x=base_value, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

# 添加决策阈值线（0.5）
ax.axvline(x=0.5, color='black', linestyle='-', linewidth=1.5, alpha=0.8)

# 修改决策区域背景色
ax.axvspan(0, 0.5, alpha=0.08, color='blue')  # 从0开始
ax.axvspan(0.5, 1.0, alpha=0.08, color='red')  # 到1.0结束

# 修改区域标签位置
ax.text(0.25, -0.3, 'Negative', ha='center', va='center', fontsize=10, color='darkblue', fontweight='bold')
ax.text(0.75, -0.3, 'Positive', ha='center', va='center', fontsize=10, color='darkred', fontweight='bold')

# 设置x轴范围和刻度
ax.set_xlim(0, 1.0)  # 改为从0开始
ax.set_xticks(np.arange(0, 1.1, 0.2))  # 修改刻度
ax.set_xticklabels([f'{x:.1f}' for x in np.arange(0, 1.1, 0.2)], fontsize=10)

# 设置y轴范围
ax.set_ylim(-0.5, len(feature_names) * feature_spacing + 0.5)

# 添加浅色网格
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='both')

# 添加样本统计信息 - 放在左下角
negative_count = colors.count('blue')
positive_count = colors.count('red')
stats_text = f'Total samples: {len(selected_indices)}\nNegative (blue): {negative_count}\nPositive (red): {positive_count}'

# 使用data coordinates放在左下角（不是在axes坐标）
# 计算左下角的坐标：x = 图形左边，y = 图形底部
x_pos = 0.05  # 稍微向左移动一点
y_pos = 0.3  # 在图形底部下方一点

ax.text(x_pos, y_pos, stats_text,
        fontsize=10,
        verticalalignment='top',  # 文本顶部对齐坐标点
        horizontalalignment='left',  # 文本左对齐坐标点
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.5),
        zorder=10)  # 确保在最上层

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "decision_path_plot_30_samples.png"), dpi=300, bbox_inches='tight')
plt.close()

print("✅ 决策路径分析图（30个样本）已保存")

# ==============================
# 9. 交互式SHAP依赖图
# ==============================
print("📊 正在生成交互式SHAP依赖图...")

# 创建交互式依赖图（使用颜色表示另一个特征的交互）
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, (ax, feat_name) in enumerate(zip(axes, feature_names)):
    # 选择交互特征（选择与当前特征相关性最低的特征）
    corr_matrix = np.corrcoef(probas.T)
    corr_with_current = corr_matrix[i, :]
    # 排除自身相关性
    corr_with_current[i] = 1
    # 找到相关性最低的特征
    interaction_idx = np.argmin(np.abs(corr_with_current))
    interaction_feat = feature_names[interaction_idx]

    # 获取数据
    x_data = probas[:, i]
    y_data = shap_vals[:, i]
    color_data = probas[:, interaction_idx]  # 用于着色的交互特征

    # 绘制散点图，颜色表示交互特征的值
    scatter = ax.scatter(x_data, y_data,
                         c=color_data,
                         cmap='viridis',  # 使用viridis颜色映射
                         alpha=0.7,
                         s=40,
                         edgecolor='white',
                         linewidth=0.5)

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(f'{interaction_feat} value', fontsize=10)

    # 计算相关系数和回归线
    if len(x_data) > 1:
        # 计算皮尔逊相关系数
        r_value, p_value = pearsonr(x_data, y_data)

        # 计算回归线
        slope, intercept = np.polyfit(x_data, y_data, 1)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = slope * x_line + intercept

        # 绘制回归线，并添加图例标签
        ax.plot(x_line, y_line, 'r-', linewidth=2, alpha=0.8,
                label=f'Trend line (r={r_value:.3f})')

        # 格式化P值文本
        if p_value < 0.001:
            p_text = 'p < 0.001'
        else:
            p_text = f'p = {p_value:.3f}'
        stats_text = f'r = {r_value:.3f}\n{p_text}'

        # 在右下角添加统计信息
        ax.text(0.98, 0.02, stats_text,
                transform=ax.transAxes,
                fontsize=11,
                fontweight='bold',
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round',
                          facecolor='white',
                          alpha=0.9,
                          edgecolor='gray',
                          linewidth=1))

        # 添加图例（显示红线）
        ax.legend(loc='best', fontsize=9)

    # 设置标题和标签
    ax.set_title(f'{feat_name} (colored by {interaction_feat})',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel(f'{feat_name} value', fontsize=11)
    ax.set_ylabel('SHAP value', fontsize=11)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # 设置坐标轴范围
    padding_x = (x_data.max() - x_data.min()) * 0.05
    padding_y = (y_data.max() - y_data.min()) * 0.1
    ax.set_xlim(x_data.min() - padding_x, x_data.max() + padding_x)
    ax.set_ylim(y_data.min() - padding_y, y_data.max() + padding_y)

    # 添加子图标签（左上角）
    ax.text(0.02, 0.98, f'({chr(97 + i)})',
            transform=ax.transAxes,
            fontsize=11, fontweight='bold',
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('SHAP dependency plot analysis (interactive feature shading) - Stacking Ensemble',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, "shap_dependence_interaction.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 交互式SHAP依赖图已保存")

# ==============================
# 10. SHAP Force Plots（力导图）- 单个样本解释
# ==============================
print("📊 正在生成SHAP Force Plots...")

# 创建专门的force plots目录
FORCE_PLOT_DIR = os.path.join(OUTPUT_DIR, "force_plots")
Path(FORCE_PLOT_DIR).mkdir(parents=True, exist_ok=True)

# 方法1：使用SHAP内置的force plot（HTML格式）
print("正在生成HTML格式的force plots...")

# 选择几个代表性样本
sample_indices = []

# 1. 选择预测概率接近0.5的样本（决策边界）
mid_proba_indices = np.argsort(np.abs(pred_proba - 0.5))[:5]
sample_indices.extend(mid_proba_indices[:2])

# 2. 选择高置信度阳性样本
high_pos_indices = np.argsort(-pred_proba)[:5]  # 预测概率最高的
sample_indices.extend(high_pos_indices[:2])

# 3. 选择高置信度阴性样本
high_neg_indices = np.argsort(pred_proba)[:5]  # 预测概率最低的
sample_indices.extend(high_neg_indices[:2])

# 去重
sample_indices = list(set(sample_indices))
sample_indices = sample_indices[:6]  # 最多6个样本

# 生成force plot HTML文件
for idx in sample_indices:
    # 创建force plot
    plt.figure(figsize=(12, 4))

    # 使用SHAP的force plot
    shap.force_plot(
        explainer.expected_value if hasattr(explainer, 'expected_value') else base_value,
        shap_values.values[idx, :],
        probas[idx, :],
        feature_names=feature_names,
        matplotlib=True,  # 使用matplotlib而不是HTML
        show=False,
        text_rotation=0
    )

    # 添加标题
    true_label = "Positive" if y_true[idx] == 1 else "Negative"
    pred_label = "Positive" if pred_proba[idx] > 0.5 else "Negative"
    plt.title(f"SHAP Force Plot - Sample {idx}\n"
              f"True: {true_label}, Predicted: {pred_label} (Proba: {pred_proba[idx]:.3f})",
              fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(FORCE_PLOT_DIR, f"force_plot_sample_{idx}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

print(f"✅ HTML格式force plots已保存到: {FORCE_PLOT_DIR}")

# ==============================
# 11. 瀑布图（替代force plot的静态版本）
# ==============================
print("📊 正在生成SHAP瀑布图...")

# 选择几个关键样本
key_samples = []

# 找到不同预测概率区间的样本
probability_bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

for prob_range in probability_bins:
    mask = (pred_proba >= prob_range[0]) & (pred_proba < prob_range[1])
    if np.sum(mask) > 0:
        # 选择这个区间中预测正确的样本
        correct_mask = mask & (model.predict(probas) == y_true)
        if np.sum(correct_mask) > 0:
            sample_idx = np.where(correct_mask)[0][0]
            key_samples.append(sample_idx)
        else:
            # 如果没有预测正确的，选择任意样本
            sample_idx = np.where(mask)[0][0]
            key_samples.append(sample_idx)

key_samples = key_samples[:5]  # 最多5个样本

# 生成瀑布图
for idx in key_samples:
    plt.figure(figsize=(10, 6))

    # 创建瀑布图
    shap.plots.waterfall(shap_values[idx],
                         max_display=10,  # 显示前10个最重要的特征
                         show=False)

    # 自定义标题
    true_label = "Positive" if y_true[idx] == 1 else "Negative"
    pred_label = "Positive" if pred_proba[idx] > 0.5 else "Negative"
    plt.title(f"SHAP Waterfall Plot - Sample {idx}\n"
              f"True Label: {true_label} | Predicted: {pred_label} | Probability: {pred_proba[idx]:.3f}",
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(FORCE_PLOT_DIR, f"waterfall_plot_sample_{idx}.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

print("✅ 瀑布图已保存")

# ==============================
# 12. SHAP热力图 - 仿文献风格
# ==============================
print("📊 正在生成SHAP热力图...")

# 提取数据
shap_vals = shap_values.values  # (n_samples, 4)
feature_names = ['Clinical', 'T2', 'DCE', 'DWI']

# 选择代表性的样本（按SHAP绝对值总和排序）
# 计算每个样本的SHAP绝对值总和
sample_shap_sum = np.abs(shap_vals).sum(axis=1)

# 选择SHAP贡献最大的30个样本
n_samples_heatmap = min(30, len(shap_vals))
top_sample_indices = np.argsort(-sample_shap_sum)[:n_samples_heatmap]

# 提取这些样本的SHAP值
heatmap_data = shap_vals[top_sample_indices, :]

# 同时提取这些样本的真实标签和预测概率
sample_labels = y_true[top_sample_indices]
sample_probas = pred_proba[top_sample_indices]

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10),
                                gridspec_kw={'width_ratios': [4, 1]})

# ===== 左图：SHAP热力图 =====
# 创建自定义颜色映射（红蓝渐变色）
cmap = plt.cm.RdBu_r

# 确定颜色范围（对称，以0为中心）
vmax = np.max(np.abs(heatmap_data))
vmin = -vmax

# 绘制热力图
im = ax1.imshow(heatmap_data, aspect='auto', cmap=cmap,
                vmin=vmin, vmax=vmax)

# 设置y轴标签（样本信息）
y_tick_labels = []
for idx in top_sample_indices:
    true_label = 'Pos' if y_true[idx] == 1 else 'Neg'
    pred_label = 'Pos' if pred_proba[idx] > 0.5 else 'Neg'
    label = f'Sample {idx}\n({true_label}→{pred_label})'
    y_tick_labels.append(label)

ax1.set_yticks(range(len(top_sample_indices)))
ax1.set_yticklabels(y_tick_labels, fontsize=8)

# 设置x轴标签（特征名）
ax1.set_xticks(range(len(feature_names)))
ax1.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10, fontweight='bold')

# 添加网格线
ax1.set_xticks(np.arange(-0.5, len(feature_names), 1), minor=True)
ax1.set_yticks(np.arange(-0.5, len(top_sample_indices), 1), minor=True)
ax1.grid(which='minor', color='gray', linestyle='-', linewidth=0.2, alpha=0.5)

# 添加标题
ax1.set_title('SHAP Values Heatmap - Stacking Ensemble',
              fontsize=14, fontweight='bold', pad=20)

# 添加x轴标签
ax1.set_xlabel('Features', fontsize=12, fontweight='bold')

# ===== 右图：样本标签和统计 =====
# 创建颜色映射用于样本标签
label_colors = ['red' if label == 1 else 'blue' for label in sample_labels]

# 绘制样本标签条
for i, (label, color) in enumerate(zip(sample_labels, label_colors)):
    ax2.barh(i, 1, color=color, alpha=0.7, height=0.8)
    label_text = 'Positive' if label == 1 else 'Negative'
    ax2.text(0.5, i, label_text, ha='center', va='center',
             fontsize=9, color='white', fontweight='bold')

# 设置右图属性
ax2.set_xlim(0, 1)
ax2.set_ylim(-0.5, len(sample_labels) - 0.5)
ax2.set_title('True Labels', fontsize=12, fontweight='bold', pad=20)
ax2.set_xticks([])  # 隐藏x轴刻度
ax2.set_yticks([])  # 隐藏y轴刻度
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)

# 添加颜色条（放在热力图下方）
cbar_ax = fig.add_axes([0.15, 0.05, 0.3, 0.02])  # [left, bottom, width, height]
cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
cbar.set_label('SHAP Value', fontsize=11, fontweight='bold')

# 添加统计信息文本框
stats_text = f'Total samples: {len(top_sample_indices)}\n'
stats_text += f'Positive samples: {np.sum(sample_labels == 1)}\n'
stats_text += f'Negative samples: {np.sum(sample_labels == 0)}'

fig.text(0.75, 0.95, stats_text, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # 为颜色条留出空间
plt.savefig(os.path.join(OUTPUT_DIR, "./shap_heatmap/shap_heatmap_Stacking_Ensemble.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ SHAP热力图（仿文献风格）已保存")

# ==============================
# 13. 最终版热力图（展示所有样本）
# ==============================
print("📊 正在生成包含所有样本的热力图...")

# 获取所有样本
all_indices = np.arange(len(shap_vals))
heatmap_data_all = shap_vals[all_indices, :]

print(f"总样本数: {len(all_indices)}")

# 创建图形（根据样本数量调整尺寸）
# 动态调整图形高度：每10个样本增加1英寸
fig_height = max(8, len(all_indices) * 0.15)  # 最小8英寸，每样本0.15英寸
fig_width = 12

fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# 确定颜色范围
# 找到所有负值中的最大值（最负）
negative_values = heatmap_data_all[heatmap_data_all < 0]
positive_values = heatmap_data_all[heatmap_data_all > 0]

if len(negative_values) > 0:
    max_negative = abs(negative_values.min())  # 最负的值的绝对值
else:
    max_negative = 0.01

if len(positive_values) > 0:
    max_positive = positive_values.max()
else:
    max_positive = 0.01

# 使用两者中较大的作为对称范围
vmax_all = max(max_negative, max_positive, 0.01)

print(f"颜色范围: [-{vmax_all:.4f}, {vmax_all:.4f}]")
print(f"负值数量: {len(negative_values)} / 总值数量: {len(heatmap_data_all.flatten())}")
print(f"负值比例: {len(negative_values) / len(heatmap_data_all.flatten()) * 100:.1f}%")

# 绘制热力图
im_all = ax.imshow(heatmap_data_all, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax_all, vmax=vmax_all)

# 设置y轴标签（样本太多时只显示部分标签）
if len(all_indices) <= 50:
    # 样本较少时显示所有标签
    y_labels = [f'Sample {idx}' for idx in all_indices]
    ax.set_yticks(range(len(all_indices)))
    ax.set_yticklabels(y_labels, fontsize=8)
else:
    # 样本较多时每10个显示一个标签
    y_ticks = np.arange(0, len(all_indices), max(1, len(all_indices) // 20))
    y_labels = [f'Sample {all_indices[i]}' for i in y_ticks]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    # 添加y轴标签说明
    ax.set_ylabel(f'Sample Index (Total: {len(all_indices)} samples)', fontsize=10)

# 设置x轴
ax.set_xticks(range(len(feature_names)))
ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=11, fontweight='bold')
ax.set_xlabel('Features', fontsize=12, fontweight='bold')

# 添加颜色条
cbar_all = fig.colorbar(im_all, ax=ax, orientation='vertical',
                        fraction=0.023, pad=0.04)  # 调整颜色条大小
cbar_all.set_label('SHAP Value\n(Red: Positive, Blue: Negative)',
                   fontsize=11, fontweight='bold')

# 添加特征统计信息
stats_height = 1.0  # 统计信息区域高度
stats_y_pos = len(all_indices) + stats_height * 0.5

for j, name in enumerate(feature_names):
    col_data = heatmap_data_all[:, j]
    mean_val = np.mean(col_data)
    std_val = np.std(col_data)
    neg_count = np.sum(col_data < 0)
    pos_count = np.sum(col_data > 0)
    zero_count = np.sum(col_data == 0)

    # 绘制特征统计条形
    total = len(col_data)
    neg_ratio = neg_count / total
    pos_ratio = pos_count / total

    # 绘制堆叠条形图显示正负比例
    ax.bar(j, pos_ratio * stats_height, bottom=len(all_indices),
           color='red', alpha=0.6, width=0.8)
    ax.bar(j, neg_ratio * stats_height, bottom=len(all_indices),
           color='blue', alpha=0.6, width=0.8)

    # 添加统计文本
    stats_text = f'{mean_val:.3f}'
    ax.text(j, len(all_indices) + stats_height + 0.1, stats_text,
            ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 添加计数文本
    count_text = f'(-{neg_count}/+{pos_count})'
    color = 'blue' if mean_val < 0 else 'red'
    ax.text(j, len(all_indices) - 1, count_text,
            ha='center', va='top', fontsize=8, color=color,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# 设置y轴范围（为统计信息留出空间）
ax.set_ylim(-1, len(all_indices) + stats_height + 1)

# 添加图例说明统计区域
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='red', alpha=0.6, label='Positive SHAP samples'),
    Patch(facecolor='blue', alpha=0.6, label='Negative SHAP samples')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

# 添加网格
ax.set_xticks(np.arange(-0.5, len(feature_names), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(all_indices), 1), minor=True)
ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.3, alpha=0.2)

# 添加标题
title = f'SHAP Values Heatmap - All Samples (n={len(all_indices)})\n'
title += 'Stacking Ensemble Model'
ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

# 添加总体统计信息
overall_stats = f'Total samples: {len(all_indices)} | '
overall_stats += f'Negative SHAP values: {len(negative_values)} ({len(negative_values) / len(heatmap_data_all.flatten()) * 100:.1f}%)'
ax.text(0.5, -0.05, overall_stats, transform=ax.transAxes,
        ha='center', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
output_path = os.path.join(OUTPUT_DIR, "./shap_heatmap/shap_heatmap_all_samples.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 包含所有样本的热力图已保存: {output_path}")

# ==============================
# 14. 分页热力图（样本太多时使用）
# ==============================
if len(all_indices) > 50:
    print("📊 样本较多，正在生成分页热力图...")

    # 每页显示的样本数
    samples_per_page = 50
    n_pages = (len(all_indices) + samples_per_page - 1) // samples_per_page

    for page in range(n_pages):
        start_idx = page * samples_per_page
        end_idx = min((page + 1) * samples_per_page, len(all_indices))
        page_indices = all_indices[start_idx:end_idx]

        fig, ax = plt.subplots(figsize=(12, 10))

        page_data = shap_vals[page_indices, :]
        im_page = ax.imshow(page_data, aspect='auto', cmap='RdBu_r',
                            vmin=-vmax_all, vmax=vmax_all)

        # 设置y轴
        y_labels = [f'Sample {idx}' for idx in page_indices]
        ax.set_yticks(range(len(page_indices)))
        ax.set_yticklabels(y_labels, fontsize=9)

        # 设置x轴
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=11, fontweight='bold')

        # 添加颜色条
        cbar_page = fig.colorbar(im_page, ax=ax)
        cbar_page.set_label('SHAP Value', fontsize=11)

        # 添加标题
        title = f'SHAP Values Heatmap - Page {page + 1}/{n_pages}\n'
        title += f'Samples {start_idx}-{end_idx - 1} (Total: {len(all_indices)})'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # 添加网格
        ax.grid(True, alpha=0.1, which='both')

        plt.tight_layout()
        page_path = os.path.join(OUTPUT_DIR, f"./shap_heatmap/shap_heatmap_page_{page + 1}.png")
        plt.savefig(page_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ 第{page + 1}页热力图已保存: {page_path}")

# ==============================
# 15. 汇总统计热力图
# ==============================
print("📊 正在生成汇总统计热力图...")

# 计算每个特征的统计信息
feature_stats = []
for j, name in enumerate(feature_names):
    col_data = shap_vals[:, j]
    stats = {
        'name': name,
        'mean': np.mean(col_data),
        'std': np.std(col_data),
        'min': np.min(col_data),
        'max': np.max(col_data),
        'neg_count': np.sum(col_data < 0),
        'pos_count': np.sum(col_data > 0),
        'q25': np.percentile(col_data, 25),
        'q50': np.percentile(col_data, 50),
        'q75': np.percentile(col_data, 75)
    }
    feature_stats.append(stats)

# 创建统计热力图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# 1. 均值热力图
ax1 = axes[0]
mean_data = np.array([s['mean'] for s in feature_stats]).reshape(1, -1)
im1 = ax1.imshow(mean_data, aspect='auto', cmap='RdBu_r',
                 vmin=-vmax_all, vmax=vmax_all)
ax1.set_title('Mean SHAP Values by Feature', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(feature_names)))
ax1.set_xticklabels(feature_names, rotation=45, ha='right')
ax1.set_yticks([0])
ax1.set_yticklabels(['Mean'], fontsize=10)

# 添加数值标签
for j, stats in enumerate(feature_stats):
    ax1.text(j, 0, f'{stats["mean"]:.3f}', ha='center', va='center',
             color='white' if abs(stats['mean']) > vmax_all / 2 else 'black',
             fontsize=10, fontweight='bold')

# 2. 标准差热力图
ax2 = axes[1]
std_data = np.array([s['std'] for s in feature_stats]).reshape(1, -1)
im2 = ax2.imshow(std_data, aspect='auto', cmap='YlOrRd')
ax2.set_title('Std Dev of SHAP Values', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(feature_names)))
ax2.set_xticklabels(feature_names, rotation=45, ha='right')
ax2.set_yticks([0])
ax2.set_yticklabels(['Std'], fontsize=10)

for j, stats in enumerate(feature_stats):
    ax2.text(j, 0, f'{stats["std"]:.3f}', ha='center', va='center',
             color='white' if stats['std'] > np.max(std_data) / 2 else 'black',
             fontsize=10, fontweight='bold')

# 3. 正负样本数量热力图
ax3 = axes[2]
neg_data = np.array([s['neg_count'] for s in feature_stats]).reshape(1, -1)
pos_data = np.array([s['pos_count'] for s in feature_stats]).reshape(1, -1)

# 堆叠条形图
x_pos = range(len(feature_names))
ax3.bar(x_pos, pos_data.flatten(), color='red', alpha=0.6, label='Positive')
ax3.bar(x_pos, neg_data.flatten(), bottom=pos_data.flatten(),
        color='blue', alpha=0.6, label='Negative')

ax3.set_title('Positive/Negative SHAP Counts', fontsize=12, fontweight='bold')
ax3.set_xlabel('Features', fontsize=10)
ax3.set_ylabel('Count', fontsize=10)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(feature_names, rotation=45, ha='right')
ax3.legend()

# 4. 分位数热力图
ax4 = axes[3]
quartile_data = np.array([[s['q25'], s['q50'], s['q75']] for s in feature_stats]).T
im4 = ax4.imshow(quartile_data, aspect='auto', cmap='RdBu_r',
                 vmin=-vmax_all, vmax=vmax_all)
ax4.set_title('SHAP Value Quartiles (25%, 50%, 75%)', fontsize=12, fontweight='bold')
ax4.set_xlabel('Features', fontsize=10)
ax4.set_ylabel('Quartile', fontsize=10)
ax4.set_xticks(range(len(feature_names)))
ax4.set_xticklabels(feature_names, rotation=45, ha='right')
ax4.set_yticks(range(3))
ax4.set_yticklabels(['Q25', 'Q50', 'Q75'], fontsize=10)

plt.suptitle('SHAP Values Statistical Summary - All Samples',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, "./shap_heatmap/shap_statistical_summary.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 汇总统计热力图已保存")

print(f"\n🎉 所有热力图分析完成！共分析了 {len(all_indices)} 个样本。")

# ==============================
# 17. 特征交互分析图
# ==============================
print("📊 正在生成特征交互分析图...")

# 创建特征交互分析的目录
INTERACTION_DIR = os.path.join(OUTPUT_DIR, "feature_interactions")
Path(INTERACTION_DIR).mkdir(parents=True, exist_ok=True)

# 方法1：使用SHAP的交互值（如果模型支持）
try:
    # 尝试计算SHAP交互值
    print("正在计算SHAP交互值...")

    # 创建解释器用于计算交互值
    explainer_interaction = shap.Explainer(model, probas[:100])  # 使用部分样本加快计算

    # 计算SHAP交互值（只计算部分样本，因为计算量大）
    n_samples_interaction = min(100, len(probas))
    shap_interaction_values = explainer_interaction.shap_interaction_values(
        probas[:n_samples_interaction]
    )

    # 如果是二分类，可能有多个输出
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[1]  # 取正类的交互值

    print(f"SHAP交互值形状: {shap_interaction_values.shape}")

    # 创建交互值热力图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    # 对每个特征绘制其与其他特征的交互
    for idx, (ax, feat_name) in enumerate(zip(axes, feature_names)):
        # 提取该特征与其他特征的交互值
        interaction_with_others = shap_interaction_values[:, idx, :]

        # 计算平均交互强度
        mean_interaction = np.mean(np.abs(interaction_with_others), axis=0)

        # 绘制条形图
        x_pos = np.arange(len(feature_names))
        colors = ['red' if val > 0 else 'blue' for val in mean_interaction]

        bars = ax.bar(x_pos, mean_interaction, color=colors, alpha=0.7, edgecolor='black')

        # 设置图表属性
        ax.set_title(f'Interaction Strength: {feat_name} with other features',
                     fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel('Features', fontsize=10)
        ax.set_ylabel('Mean |Interaction|', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)

        # 添加数值标签
        for bar, value in zip(bars, mean_interaction):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.001,
                    f'{value:.4f}', ha='center', va='bottom', fontsize=8)

        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Feature Interaction Analysis - Stacking Ensemble\n(Average Interaction Strength)',
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(INTERACTION_DIR, "shap_interaction_strength.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ SHAP交互强度图已保存")

except Exception as e:
    print(f"⚠️ 无法计算SHAP交互值: {e}")
    print("正在使用替代方法...")

# 方法2：使用配对散点图分析特征交互
print("正在生成特征配对交互图...")

# 选择几对重要的特征进行交互分析
# 基于相关性选择特征对
feature_pairs = []
for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
        corr_value = corr_matrix[i, j]
        if abs(corr_value) > 0.1:  # 只分析相关性较强的特征对
            feature_pairs.append((i, j, corr_value))

# 按相关性绝对值排序
feature_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
feature_pairs = feature_pairs[:6]  # 只分析前6对

# 创建配对交互图
n_pairs = len(feature_pairs)
n_cols = 3
n_rows = (n_pairs + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
if n_pairs > 1:
    axes = axes.flatten()
else:
    axes = [axes]

for ax_idx, (i, j, corr_value) in enumerate(feature_pairs):
    if ax_idx >= len(axes):
        break

    ax = axes[ax_idx]
    feat1_name = feature_names[i]
    feat2_name = feature_names[j]

    # 获取特征数据
    x_data = probas[:, i]
    y_data = probas[:, j]

    # 根据SHAP值着色（用两个特征的SHAP值之和）
    interaction_color = shap_vals[:, i] + shap_vals[:, j]

    # 绘制散点图
    scatter = ax.scatter(x_data, y_data,
                         c=interaction_color,
                         cmap='RdBu_r',
                         alpha=0.6,
                         s=30,
                         edgecolor='white',
                         linewidth=0.5)

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label(f'{feat1_name}+{feat2_name} SHAP', fontsize=9)

    # 添加回归线
    if len(x_data) > 1:
        slope, intercept = np.polyfit(x_data, y_data, 1)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.7)

    # 设置图表属性
    ax.set_xlabel(feat1_name, fontsize=11, fontweight='bold')
    ax.set_ylabel(feat2_name, fontsize=11, fontweight='bold')
    ax.set_title(f'{feat1_name} vs {feat2_name}\nCorrelation: {corr_value:.3f}',
                 fontsize=12, fontweight='bold', pad=10)

    ax.grid(True, alpha=0.3)

# 隐藏多余的子图
for ax_idx in range(len(feature_pairs), len(axes)):
    axes[ax_idx].set_visible(False)

plt.suptitle('Feature Interaction Analysis - Scatter Plots\n(Color: Combined SHAP Value)',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(INTERACTION_DIR, "feature_interaction_scatter.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 特征交互散点图已保存")

# ==============================
# 18. 三维特征交互图
# ==============================
print("📊 正在生成三维特征交互图...")

from mpl_toolkits.mplot3d import Axes3D

# 选择相关性最强的三个特征
if len(feature_names) >= 3:
    # 找到相关性最强的三元组
    top_triplets = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            for k in range(j + 1, len(feature_names)):
                # 计算平均相关性
                avg_corr = (abs(corr_matrix[i, j]) +
                            abs(corr_matrix[i, k]) +
                            abs(corr_matrix[j, k])) / 3
                top_triplets.append((i, j, k, avg_corr))

    # 按平均相关性排序
    top_triplets.sort(key=lambda x: x[3], reverse=True)

    # 创建三维图
    fig = plt.figure(figsize=(14, 10))

    for plot_idx in range(min(2, len(top_triplets))):  # 最多两个三维图
        i, j, k, avg_corr = top_triplets[plot_idx]

        ax = fig.add_subplot(1, 2, plot_idx + 1, projection='3d')

        # 获取数据
        x_data = probas[:100, i]  # 只使用前100个样本避免太拥挤
        y_data = probas[:100, j]
        z_data = probas[:100, k]

        # 根据预测概率着色
        colors = pred_proba[:100]

        # 绘制散点图
        scatter = ax.scatter(x_data, y_data, z_data,
                             c=colors,
                             cmap='viridis',
                             alpha=0.7,
                             s=20)

        # 设置标签
        ax.set_xlabel(feature_names[i], fontsize=10, labelpad=10)
        ax.set_ylabel(feature_names[j], fontsize=10, labelpad=10)
        ax.set_zlabel(feature_names[k], fontsize=10, labelpad=10)

        ax.set_title(f'{feature_names[i]}-{feature_names[j]}-{feature_names[k]}\nAvg Corr: {avg_corr:.3f}',
                     fontsize=11, fontweight='bold', pad=20)

        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1)
        cbar.set_label('Prediction Probability', fontsize=9)

    plt.suptitle('3D Feature Interaction Analysis\n(Color: Model Prediction Probability)',
                 fontsize=16, fontweight='bold', y=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.8])
    plt.savefig(os.path.join(INTERACTION_DIR, "3d_feature_interaction.png"),
                dpi=300, bbox_inches='tight')
    plt.close()

    print("✅ 三维特征交互图已保存")

# ==============================
# 19. 交互效应热力图
# ==============================
print("📊 正在生成交互效应热力图...")

# 计算特征间的交互效应（基于SHAP值的协方差）
interaction_matrix = np.zeros((len(feature_names), len(feature_names)))

for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        if i == j:
            # 对角线：特征的自身重要性
            interaction_matrix[i, j] = np.mean(np.abs(shap_vals[:, i]))
        else:
            # 交互效应：两个特征SHAP值的协方差
            cov_value = np.cov(shap_vals[:, i], shap_vals[:, j])[0, 1]
            interaction_matrix[i, j] = cov_value

# 创建图形
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：交互效应热力图
im1 = ax1.imshow(interaction_matrix, cmap='RdBu_r', aspect='auto')

# 设置坐标轴
ax1.set_xticks(range(len(feature_names)))
ax1.set_yticks(range(len(feature_names)))
ax1.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=11, fontweight='bold')
ax1.set_yticklabels(feature_names, fontsize=11, fontweight='bold')

# 添加数值标签
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        value = interaction_matrix[i, j]
        color = 'white' if abs(value) > np.max(np.abs(interaction_matrix)) / 2 else 'black'
        ax1.text(j, i, f'{value:.4f}',
                 ha='center', va='center',
                 color=color, fontsize=9)

ax1.set_title('Feature Interaction Matrix\n(Diagonal: Self-importance, Off-diagonal: Covariance)',
              fontsize=13, fontweight='bold', pad=20)
plt.colorbar(im1, ax=ax1, label='Interaction Strength')

# 右图：网络图表示交互关系
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal')
ax2.axis('off')

# 计算节点位置（圆形排列）
angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False)
node_positions = np.column_stack([np.cos(angles), np.sin(angles)])

# 绘制节点
node_sizes = np.diag(interaction_matrix) * 500  # 节点大小表示自身重要性
for i, (pos, size, name) in enumerate(zip(node_positions, node_sizes, feature_names)):
    ax2.scatter(pos[0], pos[1], s=size, color='skyblue', alpha=0.7, edgecolor='black')
    ax2.text(pos[0] * 1.2, pos[1] * 1.2, name,
             ha='center', va='center', fontsize=12, fontweight='bold')

# 绘制边（表示交互）
for i in range(len(feature_names)):
    for j in range(i + 1, len(feature_names)):
        interaction_strength = abs(interaction_matrix[i, j])
        if interaction_strength > 0.001:  # 只绘制显著的交互
            color = 'red' if interaction_matrix[i, j] > 0 else 'blue'
            width = interaction_strength * 10  # 线宽表示交互强度
            ax2.plot([node_positions[i, 0], node_positions[j, 0]],
                     [node_positions[i, 1], node_positions[j, 1]],
                     color=color, linewidth=width, alpha=0.5)

ax2.set_title('Feature Interaction Network\n(Node size: Self-importance, Edge: Interaction)',
              fontsize=13, fontweight='bold', pad=20)

plt.suptitle('Feature Interaction Analysis - Stacking Ensemble',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(os.path.join(INTERACTION_DIR, "feature_interaction_network.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 交互效应热力图已保存")

print(f"\n🎉 特征交互分析完成！所有图表已保存到: {INTERACTION_DIR}")

# ==============================
# 33. 基于特征重要性的聚类相关性热力图
# ==============================
print("📊 正在生成基于特征重要性的聚类相关性热力图...")

# 获取 SHAP 值的绝对均值作为特征重要性
shap_importance = np.abs(shap_values.values).mean(axis=0)

# 获取所有特征的名称
feature_names = ['Clinical', 'T2', 'DCE', 'DWI']

# 创建DataFrame
data_df = pd.DataFrame(probas, columns=feature_names)

# 计算特征间的相关系数矩阵
corr_matrix = data_df.corr()

# 使用层次聚类对特征进行聚类，并根据结果对相关系数矩阵重新排序
linkage_matrix = linkage(corr_matrix, method='ward')

# 获取聚类顺序
from scipy.cluster.hierarchy import leaves_list
order = leaves_list(linkage_matrix)

# 根据聚类结果对相关系数矩阵重新排序
corr_clustered = corr_matrix.iloc[order, order]
clustered_features = [feature_names[i] for i in order]

# 绘制聚类后的相关系数热图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_clustered,
            cmap='RdBu_r',
            annot=True,
            fmt=".2f",
            square=True,
            center=0,
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8},
            linewidths=0.5,
            linecolor='white')

# 设置坐标轴标签
plt.xticks(range(len(clustered_features)), clustered_features,
           rotation=45, ha='right', fontsize=11, fontweight='bold')
plt.yticks(range(len(clustered_features)), clustered_features,
           rotation=0, fontsize=11, fontweight='bold')

# 添加特征重要性信息到标签
for i, feat in enumerate(clustered_features):
    feat_idx = feature_names.index(feat)
    importance = shap_importance[feat_idx]
    # 在y轴标签中添加重要性
    plt.gca().get_yticklabels()[i].set_text(f'{feat}\n({importance:.4f})')

plt.title('Clustered Correlation Heatmap\nFeatures grouped by similarity (SHAP importance in parentheses)',
          fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "clustered_correlation_heatmap.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 聚类相关性热力图已保存")

# ==============================
# 35. 特征重要性与相关性结合分析
# ==============================
print("📊 正在生成特征重要性与相关性结合分析图...")

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 特征重要性条形图
ax1 = axes[0, 0]
# 按重要性排序
sorted_indices = np.argsort(-shap_importance)
sorted_features = [feature_names[i] for i in sorted_indices]
sorted_importance = shap_importance[sorted_indices]

bars = ax1.barh(range(len(sorted_features)), sorted_importance,
                color='lightcoral', edgecolor='black', alpha=0.8)

ax1.set_yticks(range(len(sorted_features)))
ax1.set_yticklabels(sorted_features, fontsize=10)
ax1.invert_yaxis()  # 最重要的在最上面
ax1.set_xlabel('Mean |SHAP Value|', fontsize=11)
ax1.set_title('Feature Importance Ranking', fontsize=12, fontweight='bold')

# 添加数值标签
for bar, importance in zip(bars, sorted_importance):
    width = bar.get_width()
    ax1.text(width + 0.001, bar.get_y() + bar.get_height()/2,
             f'{importance:.4f}', va='center', ha='left', fontsize=9)

# 2. 相关系数热力图
ax2 = axes[0, 1]
im2 = ax2.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)

ax2.set_xticks(range(len(feature_names)))
ax2.set_yticks(range(len(feature_names)))
ax2.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=10)
ax2.set_yticklabels(feature_names, fontsize=10)

# 添加数值
for i in range(len(feature_names)):
    for j in range(len(feature_names)):
        corr_value = corr_matrix.iloc[i, j]
        color = 'white' if abs(corr_value) > 0.5 else 'black'
        ax2.text(j, i, f'{corr_value:.2f}',
                ha='center', va='center',
                color=color, fontsize=9)

ax2.set_title('Correlation Matrix', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# 3. 重要性与相关性散点图
ax3 = axes[1, 0]
# 计算每个特征的平均相关性（与其他特征）
mean_correlations = []
for i in range(len(feature_names)):
    other_corrs = []
    for j in range(len(feature_names)):
        if i != j:
            other_corrs.append(abs(corr_matrix.iloc[i, j]))
    mean_correlations.append(np.mean(other_corrs))

scatter = ax3.scatter(shap_importance, mean_correlations,
                     s=200, alpha=0.7, edgecolor='black')

# 添加标签
for i, feat in enumerate(feature_names):
    ax3.annotate(feat, (shap_importance[i], mean_correlations[i]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, fontweight='bold')

ax3.set_xlabel('Feature Importance (Mean |SHAP|)', fontsize=11)
ax3.set_ylabel('Mean Absolute Correlation with other features', fontsize=11)
ax3.set_title('Importance vs Mean Correlation', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. 特征关系网络图
ax4 = axes[1, 1]
ax4.set_xlim(-1.2, 1.2)
ax4.set_ylim(-1.2, 1.2)
ax4.set_aspect('equal')
ax4.axis('off')

# 计算节点位置
angles = np.linspace(0, 2*np.pi, len(feature_names), endpoint=False)
node_pos = np.column_stack([np.cos(angles), np.sin(angles)])

# 绘制节点（大小表示重要性）
max_importance = shap_importance.max()
node_sizes = shap_importance / max_importance * 1000 + 300

for i, (pos, size, feat) in enumerate(zip(node_pos, node_sizes, feature_names)):
    ax4.scatter(pos[0], pos[1], s=size, color='skyblue',
                alpha=0.8, edgecolor='black', linewidth=2)
    ax4.text(pos[0]*1.15, pos[1]*1.15, feat,
             ha='center', va='center', fontsize=11, fontweight='bold')

# 绘制边（表示相关性）
for i in range(len(feature_names)):
    for j in range(i+1, len(feature_names)):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > 0.2:  # 只绘制相关性较强的边
            color = 'red' if corr_value > 0 else 'blue'
            width = abs(corr_value) * 3
            ax4.plot([node_pos[i, 0], node_pos[j, 0]],
                    [node_pos[i, 1], node_pos[j, 1]],
                    color=color, linewidth=width, alpha=0.5)

ax4.set_title('Feature Relationship Network\n(Node size: Importance, Edge: Correlation)',
              fontsize=12, fontweight='bold')

plt.suptitle('Comprehensive Feature Analysis: Importance & Correlation\nStacking Ensemble Model',
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance_correlation_comprehensive.png"),
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 特征重要性与相关性结合分析图已保存")

print("\n🎉 所有特征相关性分析完成！")

print("\n🎉 所有SHAP分析完成！图表已保存至:", OUTPUT_DIR)