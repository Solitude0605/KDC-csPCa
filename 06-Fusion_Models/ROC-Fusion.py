import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    accuracy_score, cohen_kappa_score,
    precision_score, recall_score, f1_score
)
from scipy.interpolate import interp1d
import os

# ==============================
# 全局字体设置
# ==============================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Calibri', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# 路径配置
# ==============================
CSV_DIR = r"K:\PCa_2026\Article\放射组学\图表\roc\内部测试\后融合-ROC\csv"
OUTPUT_PNG = r"K:\PCa_2026\Article\放射组学\图表\roc\内部测试\后融合-ROC\Top3_ROC_with_CI_Band.tif"
METRICS_CSV = r"K:\PCa_2026\Article\放射组学\图表\roc\内部测试\后融合-ROC\Model_Metrics.csv"

COLORS = {
    'Knowledge-Driven Conditional': '#8A2BE2',
    'Stacking_LR': '#7DD2F6',
    'Data-Driven Conditional': '#FF8C00',
    # 'External_test': '#00FF00',
    'AUC_weighted': '#468BCA',
    'Stacking_RF': '#B384BA',
    'Heuristic': '#D9C2DD'
    # 'Conditional_T2_LR': '#DEAA82',
}

# ==============================
# Bootstrap 计算 ROC 置信带
# ==============================
def bootstrap_roc_ci(y_true, y_score, n_bootstraps=500):
    rng = np.random.RandomState(42)
    base_fpr = np.linspace(0, 1, 101)
    tprs = []
    aucs = []

    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true.iloc[indices])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true.iloc[indices], y_score.iloc[indices])
        interp_func = interp1d(fpr, tpr, bounds_error=False, fill_value=(0.0, 1.0))
        tprs.append(interp_func(base_fpr))
        aucs.append(auc(fpr, tpr))

    tprs = np.array(tprs)
    ci_lower = np.percentile(tprs, 2.5, axis=0)
    ci_upper = np.percentile(tprs, 97.5, axis=0)
    auc_ci = np.percentile(aucs, [2.5, 97.5])

    return base_fpr, ci_lower, ci_upper, auc_ci

# ==============================
# 收集结果 + 计算指标
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

    # === 计算分类指标 ===
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)  # = Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # 主 ROC & AUC
    fpr_main, tpr_main, _ = roc_curve(y_true, y_score)
    auc_val = auc(fpr_main, tpr_main)

    # Bootstrap CI
    fpr_grid, tpr_lower, tpr_upper, auc_ci = bootstrap_roc_ci(y_true, y_score)

    # 存储绘图数据
    results.append({
        'method': method_name,
        'fpr_main': fpr_main,
        'tpr_main': tpr_main,
        'auc': auc_val,
        'auc_ci': auc_ci,
        'fpr_grid': fpr_grid,
        'tpr_lower': tpr_lower,
        'tpr_upper': tpr_upper,
        'color': COLORS[method_name]
    })

    # === 保存混淆矩阵图 ===
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                annot_kws={'size': 15})
    plt.title(f'{method_name}\nAUC = {auc_val:.3f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(CSV_DIR, '..', 'confusion_matrices', f"{method_name}_confusion_matrix.png")
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存: {cm_path}")

    # 存储指标（用于 CSV）
    metrics_list.append({
        'Method': method_name,
        'Accuracy': round(acc, 4),
        'Kappa': round(kappa, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1_Score': round(f1, 4),
        'Sensitivity': round(rec, 4),  # 和 Recall 相同
        'AUC': round(auc_val, 4),
        'AUC_95%CI_Lower': round(auc_ci[0], 4),
        'AUC_95%CI_Upper': round(auc_ci[1], 4)
    })

# 按 AUC 排序（影响绘图顺序）
results.sort(key=lambda x: x['auc'], reverse=True)
# 同时对 metrics_list 按 AUC 排序（保持一致）
metrics_df = pd.DataFrame(metrics_list)
metrics_df = metrics_df.sort_values(by='AUC', ascending=False)
metrics_df.to_csv(METRICS_CSV, index=False, encoding='utf-8-sig')  # utf-8-sig 支持 Excel 中文
print(f"✅ 模型指标已保存至: {METRICS_CSV}")

# ==============================
# 绘图
# ==============================
plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, alpha=0.7, label='Random')

# 创建映射
method_to_result = {r['method']: r for r in results}

# 指定哪些模型用实线（加粗）
highlight_models = ['Knowledge-Driven Conditional', 'Stacking_LR',
                    'Data-Driven Conditional', 'External_test']

# 指定哪些模型要画置信区间（仅前3个，不含 External_test）
ci_models = ['Knowledge-Driven Conditional', 'Stacking_LR', 'Data-Driven Conditional']

for _, row in metrics_df.iterrows():
    method = row['Method']
    res = method_to_result[method]
    color = res['color']
    auc_val = res['auc']
    auc_ci = res['auc_ci']

    # 决定线条样式
    if method in highlight_models:
        plt.plot(res['fpr_main'], res['tpr_main'], color=color, lw=2.5,
                 label=f'{method} (AUC={auc_val:.3f}, 95% CI [{auc_ci[0]:.3f}–{auc_ci[1]:.3f}])')
    else:
        plt.plot(res['fpr_main'], res['tpr_main'], color=color, lw=1.5, linestyle=':',
                 label=f'{method} (AUC={auc_val:.3f}, 95% CI [{auc_ci[0]:.3f}–{auc_ci[1]:.3f}])')

    # 仅对指定的三个模型添加置信区间带
    if method in ci_models:
        plt.fill_between(res['fpr_grid'], res['tpr_lower'], res['tpr_upper'],
                         color=color, alpha=0.2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Last Fusion ROC Curves for Internal Testing \n (AUC and 95% CI)', fontsize=20)
plt.legend(loc="lower right", fontsize=15, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3, color='gray')
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"✅ ROC 图已保存至: {OUTPUT_PNG}")
plt.show()