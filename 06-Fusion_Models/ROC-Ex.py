import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.metrics import roc_auc_score  # 用于 bootstrap
import seaborn as sns
import os
from scipy.interpolate import interp1d  # 正确的导入方式

# ==============================
# 全局字体设置（支持中文）
# ==============================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# 路径配置
# ==============================
CSV_DIR = r"K:\PCa_2026\Article\放射组学\图表\roc\外部测试\Pre-csv"
OUTPUT_PNG = r"K:\PCa_2026\Article\放射组学\图表\roc\外部测试\Pre-csv\单模态\Top3_ROC_with_CI_Band.tif"
METRICS_CSV = r"K:\PCa_2026\Article\放射组学\图表\roc\外部测试\Pre-csv\单模态\Model_Metrics.csv"

# ==============================
# 颜色映射（必须与文件名前缀一致）
# ==============================
COLORS = {
    'DCE': '#8A2BE2',
    'DWI': '#7DD2F6',
    'Clinical': '#FF8C00',
    'T2': '#00FF00',
    'Our_Model': '#468BCA',
    'Stacking_RF': '#B384BA',
    'Heuristic': '#D9C2DD'
}

# ==============================
# Bootstrap 计算 AUC 的 95% 置信区间
# ==============================
def compute_auc_ci(y_true, y_score, n_bootstraps=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    bootstrapped_aucs = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true.iloc[indices])) < 2:
            continue
        try:
            auc_val = roc_auc_score(y_true.iloc[indices], y_score.iloc[indices])
            bootstrapped_aucs.append(auc_val)
        except:
            continue

    if len(bootstrapped_aucs) < 2:
        return np.nan, np.nan

    lower = np.percentile(bootstrapped_aucs, 2.5)
    upper = np.percentile(bootstrapped_aucs, 97.5)
    return lower, upper

# ==============================
# 主流程：收集结果 + 计算指标
# ==============================
results = []
metrics_list = []

for filename in os.listdir(CSV_DIR):
    if not filename.endswith('_predictions.csv'):
        continue

    method_name = filename[:-len('_predictions.csv')]
    if method_name not in COLORS:
        print(f"⚠️ 跳过未定义颜色的方法: {method_name}")
        continue

    filepath = os.path.join(CSV_DIR, filename)
    df = pd.read_csv(filepath)

    if 'y_true' not in df.columns or ('y_proba' not in df.columns and 'y_pred' not in df.columns):
        print(f"❌ 文件 {filename} 缺少必要列")
        continue

    y_true = df['y_true']
    y_score = df['y_proba'] if 'y_proba' in df.columns else df['y_pred']

    # 二值预测（阈值=0.5）
    y_pred = (y_score >= 0.5).astype(int)

    # === 分类指标 ===
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # === ROC & AUC ===
    fpr_main, tpr_main, _ = roc_curve(y_true, y_score)
    auc_val = auc(fpr_main, tpr_main)

    # === Bootstrap AUC 95% CI ===
    ci_lower, ci_upper = compute_auc_ci(y_true, y_score)

    # === 存储绘图数据 ===
    results.append({
        'method': method_name,
        'fpr_main': fpr_main,
        'tpr_main': tpr_main,
        'auc': auc_val,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'color': COLORS[method_name]
    })

    # === 保存混淆矩阵 ===
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Predicted Negative', 'Predicted Positive'],
        yticklabels=['Actual Negative', 'Actual Positive'],
        annot_kws={"size": 18}  # 调大数字字体
    )
    plt.title(f'{method_name}\nAUC = {auc_val:.3f}', fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(CSV_DIR, '..', 'confusion_matrices', f"{method_name}_confusion_matrix.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {cm_path}")

    # === 保存指标 ===
    metrics_list.append({
        'Method': method_name,
        'Accuracy': round(acc, 4),
        'Kappa': round(kappa, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1_Score': round(f1, 4),
        'Sensitivity': round(rec, 4),
        'AUC': round(auc_val, 4),
        'AUC_95%CI_Lower': round(ci_lower, 4),
        'AUC_95%CI_Upper': round(ci_upper, 4)
    })

# ==============================
# 排序并保存指标 CSV
# ==============================
if not metrics_list:
    raise ValueError("❌ 没有成功加载任何有效模型！请检查文件路径和格式。")

metrics_df = pd.DataFrame(metrics_list)
metrics_df = metrics_df.sort_values(by='AUC', ascending=False)
metrics_df.to_csv(METRICS_CSV, index=False, encoding='utf-8-sig')
print(f"✅ 模型指标已保存至: {METRICS_CSV}")

from scipy.interpolate import interp1d

# 假设results已经填充完毕
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, alpha=0.7, label='Random')

method_to_result = {r['method']: r for r in results}
highlight_models = ['DCE', 'DWI', 'T2', 'Clinical', 'Knowledge-Driven Conditional']

# 定义新的FPR数组，用于插值
mean_fpr = np.linspace(0, 1, 100)

for _, row in metrics_df.iterrows():
    method = row['Method']
    res = method_to_result[method]
    color = res['color']
    auc_val = res['auc']
    ci_lower = res['ci_lower']
    ci_upper = res['ci_upper']

    fpr = res['fpr_main']
    tpr = res['tpr_main']

    # 使用线性插值生成更多的点
    interp_tpr = interp1d(fpr, tpr, kind='linear', fill_value="extrapolate")
    tpr_smooth = interp_tpr(mean_fpr)
    tpr_smooth[0] = 0.0  # 在最左边起点设置为(0,0)
    tpr_smooth[-1] = 1.0  # 在最右边终点设置为(1,1)

    # 图例包含 AUC 和 95% CI
    label_text = f"{method} (AUC={auc_val:.3f}, 95% CI [{ci_lower:.3f}–{ci_upper:.3f}])"

    if method in highlight_models:
        plt.plot(mean_fpr, tpr_smooth, color=color, lw=2.5, label=label_text)
    else:
        plt.plot(mean_fpr, tpr_smooth, color=color, lw=1.5, linestyle=':', label=label_text)

# 其余绘图代码保持不变...


# # ==============================
# # 绘制 ROC 曲线（无置信区间带，但图例含 CI）
# # ==============================
# plt.figure(figsize=(10, 8))
# plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, alpha=0.7, label='Random')
#
# method_to_result = {r['method']: r for r in results}
# highlight_models = ['DCE', 'DWI', 'T2', 'Clinical', 'Knowledge-Driven Conditional']
#
# for _, row in metrics_df.iterrows():
#     method = row['Method']
#     res = method_to_result[method]
#     color = res['color']
#     auc_val = res['auc']
#     ci_lower = res['ci_lower']
#     ci_upper = res['ci_upper']
#
#     # 图例包含 AUC 和 95% CI
#     label_text = f"{method} (AUC={auc_val:.3f}, 95% CI [{ci_lower:.3f}–{ci_upper:.3f}])"
#
#     if method in highlight_models:
#         plt.plot(res['fpr_main'], res['tpr_main'], color=color, lw=2.5, label=label_text)
#     else:
#         plt.plot(res['fpr_main'], res['tpr_main'], color=color, lw=1.5, linestyle=':', label=label_text)
#
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate', fontsize=14)
# plt.ylabel('True Positive Rate', fontsize=14)
# plt.title('Last Fusion ROC Curves for External Testing (AUC and 95% CI)', fontsize=23)
# plt.legend(loc="lower right", fontsize=14, frameon=True, fancybox=True, shadow=True)
# plt.grid(True, alpha=0.3, color='gray')
# plt.tight_layout()
# plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
# print(f"✅ ROC 图已保存至: {OUTPUT_PNG}")
# plt.show()

# ==============================
# 绘制 ROC 曲线（无置信区间带，但图例含 CI），采用插值方法使曲线更平滑
# ==============================
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, alpha=0.7, label='Random')

method_to_result = {r['method']: r for r in results}
highlight_models = ['DCE', 'DWI', 'T2', 'Clinical', 'Our_Model']

# 定义新的FPR数组，用于插值
mean_fpr = np.linspace(0, 1, 100)

for _, row in metrics_df.iterrows():
    method = row['Method']
    res = method_to_result[method]
    color = res['color']
    auc_val = res['auc']
    ci_lower = res['ci_lower']
    ci_upper = res['ci_upper']

    fpr = res['fpr_main']
    tpr = res['tpr_main']

    # 使用线性插值生成更多的点
    interp_tpr = interp1d(fpr, tpr, kind='linear', fill_value="extrapolate")
    tpr_smooth = interp_tpr(mean_fpr)
    tpr_smooth[0] = 0.0  # 确保起点为(0,0)
    tpr_smooth[-1] = 1.0  # 确保终点为(1,1)

    # 图例包含 AUC 和 95% CI
    label_text = f"{method} (AUC={auc_val:.3f}, 95% CI [{ci_lower:.3f}–{ci_upper:.3f}])"

    if method in highlight_models:
        plt.plot(mean_fpr, tpr_smooth, color=color, lw=2.5, label=label_text)
    else:
        plt.plot(mean_fpr, tpr_smooth, color=color, lw=1.5, linestyle=':', label=label_text)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('Last Fusion ROC Curves for External Testing \n (AUC and 95% CI)', fontsize=23)
plt.legend(loc="lower right", fontsize=16, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3, color='gray')
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"✅ ROC 图已保存至: {OUTPUT_PNG}")
plt.show()