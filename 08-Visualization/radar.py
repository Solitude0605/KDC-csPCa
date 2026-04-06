import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from math import pi

# -----------------------------
# 配置路径
# -----------------------------
data_dir = r'K:\PCa_2026\Article\放射组学\图表\早期融合\cs'
output_dir = os.path.join(data_dir, r'K:\PCa_2026\Article\放射组学\图表\雷达图\Early_radar_plots')
os.makedirs(output_dir, exist_ok=True)

# 新的指标名称（适用于正负样本分析）
metrics_names = ['Accuracy', 'Sensitivity (Recall+)', 'Specificity (Recall-)', 'PPV (Precision+)', 'NPV (Precision-)']
num_vars = len(metrics_names)
angles = np.linspace(0, 2 * pi, num_vars, endpoint=False).tolist()
angles += angles[:1]


def calculate_clinical_metrics(y_true, y_pred, y_proba):
    """
    计算临床常用指标：
    - Accuracy: 全局
    - Sensitivity = Recall of class 1
    - Specificity = Recall of class 0
    - PPV = Precision of class 1
    - NPV = Precision of class 0
    """
    acc = accuracy_score(y_true, y_pred)

    # 敏感性（召回率+）
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    # 特异性（召回率-）
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)

    # 阳性预测值（PPV）
    ppv = precision_score(y_true, y_pred, pos_label=1, zero_division=0)

    # 阴性预测值（NPV）
    npv = precision_score(y_true, y_pred, pos_label=0, zero_division=0)

    # AUC 是全局指标，这里不放入 per-class 雷达图
    auc = roc_auc_score(y_true, y_proba)  # 保留用于打印或后续分析

    return [acc, sensitivity, specificity, ppv, npv], auc


# -----------------------------
# 1. 扫描并加载所有模型结果
# -----------------------------
files = [f for f in os.listdir(data_dir) if f.startswith('predictions_') and f.endswith('.csv')]
models = []

print("🔍 正在扫描模型文件...")
for file in files:
    base_name = file.replace('.csv', '')
    parts = base_name.split('_', 2)
    if len(parts) < 3:
        print(f"⚠️ 跳过无法解析的文件: {file}")
        continue

    model_name_underscore = parts[2]
    model_name_space = model_name_underscore.replace('_', ' ')

    prob_col = f"{model_name_space}_prob"
    pred_col = f"{model_name_space}_pred"

    df_path = os.path.join(data_dir, file)
    df = pd.read_csv(df_path)

    if 'y_true' not in df.columns:
        print(f"❌ 错误：{file} 缺少 'y_true' 列")
        continue
    if prob_col not in df.columns or pred_col not in df.columns:
        print(f"❌ 错误：{file} 缺少 '{prob_col}' 或 '{pred_col}' 列")
        print(f"   可用列: {list(df.columns)}")
        continue

    y_true = df['y_true'].values
    y_proba = df[prob_col].values
    y_pred = df[pred_col].values

    # 如果 y_true 只有一个类别，跳过（无法计算有意义的指标）
    if len(np.unique(y_true)) < 2:
        print(f"⚠️ 跳过 {file}：y_true 只有一个类别")
        continue

    # 计算指标（单组，不是分正负子集！）
    metrics, auc = calculate_clinical_metrics(y_true, y_pred, y_proba)

    models.append({
        'display_name': model_name_space,
        'save_name': model_name_underscore,
        'metrics': metrics,
        'auc': auc
    })

    print(f"✅ 加载模型: {model_name_space} ← AUC={auc:.3f}")

print(f"\n📊 共加载 {len(models)} 个模型。\n")

# -----------------------------
# 2. 为每个模型绘制雷达图（一个模型一张图，一种颜色）
# -----------------------------
colors = plt.cm.tab10.colors  # 10种鲜明颜色，足够11个模型（会循环）

for idx, model in enumerate(models):
    values = model['metrics'] + [model['metrics'][0]]  # 闭合
    color = colors[idx % len(colors)]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, values, 'o-', linewidth=2, markersize=6, color=color)
    ax.fill(angles, values, alpha=0.25, color=color)

    ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], metrics_names)
    ax.set_ylim(0, 1)
    plt.title(f"Performance Radar – {model['display_name']}\n(AUC = {model['auc']:.3f})", fontsize=20, pad=20)

    output_path = os.path.join(output_dir, f"radar_{model['save_name']}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"💾 已保存: {output_path}")

# -----------------------------
# 2. 为所有模型绘制雷达图（所有模型绘制成一张图）
# -----------------------------

# 创建一个新的图形
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# 设置颜色循环
colors = plt.cm.get_cmap('tab10').colors

# 绘制每个模型的雷达图
for idx, model in enumerate(models):
    values = model['metrics'] + [model['metrics'][0]]  # 闭合
    color = colors[idx % len(colors)]

    ax.plot(angles, values, 'o-', linewidth=2, markersize=6, color=color, label=model['display_name'])
    ax.fill(angles, values, alpha=0.1, color=color)  # 减少透明度以便于比较

# 设置图表属性
ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], metrics_names)
ax.set_ylim(0, 1)
plt.title("Performance Radar of All Models", fontsize=14, pad=20)

# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

# 保存汇总图
output_path_all = os.path.join(output_dir, "radar_all_models.png")
plt.savefig(output_path_all, dpi=300, bbox_inches='tight')
plt.close()

print(f"🎉 所有模型的汇总雷达图已保存至:\n{output_path_all}")

print(f"\n🎉 所有雷达图已保存至:\n{output_dir}")