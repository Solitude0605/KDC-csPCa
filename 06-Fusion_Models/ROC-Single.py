import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.interpolate import interp1d
import os
from matplotlib import cm

# 字体设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Calibri', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径配置
CSV_DIR = r"K:\PCa_2026\Article\放射组学\图表\roc\内部测试\单模态-ROC\DWI"
OUTPUT_PNG = r"K:\PCa_2026\Article\放射组学\图表\roc\内部测试\单模态-ROC\699\DWI_11_Models_ROC.tif"
CM_OUTPUT = r"K:\PCa_2026\Article\放射组学\图表\roc\内部测试\单模态-混淆矩阵\699\DWI_Best_Model_Confusion_Matrix.tif"

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

# 收集结果（和之前一样）
results = []

for filename in os.listdir(CSV_DIR):
    if not (filename.startswith('predictions_rank') and filename.endswith('.csv')):
        continue

    filepath = os.path.join(CSV_DIR, filename)
    try:
        df = pd.read_csv(filepath)

        prob_cols = [col for col in df.columns if col.endswith('_prob')]
        if len(prob_cols) != 1:
            continue
        prob_col = prob_cols[0]
        model_name = prob_col[:-5]

        if 'y_true' not in df.columns:
            continue

        y_true = df['y_true']
        y_score = df[prob_col]

        if len(np.unique(y_true)) < 2:
            continue

        # 计算二值预测（用于混淆矩阵）
        y_pred = (y_score >= 0.5).astype(int)

        fpr_main, tpr_main, _ = roc_curve(y_true, y_score)
        auc_val = auc(fpr_main, tpr_main)
        fpr_grid, tpr_lower, tpr_upper, auc_ci = bootstrap_roc_ci(y_true, y_score)

        results.append({
            'method': model_name,
            'fpr_main': fpr_main,
            'tpr_main': tpr_main,
            'auc': auc_val,
            'auc_ci': auc_ci,
            'fpr_grid': fpr_grid,
            'tpr_lower': tpr_lower,
            'tpr_upper': tpr_upper,
            'y_true': y_true,      # 👈 保存原始标签
            'y_pred': y_pred       # 👈 保存二值预测（用于混淆矩阵）
        })

        print(f"✅ 加载: {model_name} | AUC = {auc_val:.4f}")

    except Exception as e:
        print(f"❌ 错误处理 {filename}: {e}")

if len(results) == 0:
    raise RuntimeError("未加载任何模型！")

# 按 AUC 排序
results.sort(key=lambda x: x['auc'], reverse=True)

# ==============================
# 🎯 绘制最优模型的混淆矩阵
# ==============================
best = results[0]
cm = confusion_matrix(best['y_true'], best['y_pred'])

plt.figure(figsize=(6, 5))
sns.set(font_scale=1.2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Negative', 'Predicted Positive'],
            yticklabels=['Actual Negative', 'Actual Positive'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 15})
plt.title(f'DWI – {best["method"]} (AUC = {best["auc"]:.3f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(CM_OUTPUT, dpi=300, bbox_inches='tight')
print(f"✅ 最优模型混淆矩阵已保存至: {CM_OUTPUT}")
plt.show()

# ==============================
# 绘图（白底 + 黑色边框）
# ==============================
plt.figure(figsize=(12, 9))
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1, alpha=0.7, label='Random')

colors = plt.cm.tab20(np.linspace(0, 1, len(results)))

for i, (res, color) in enumerate(zip(results, colors)):
    method = res['method']
    auc_val = res['auc']
    auc_ci = res['auc_ci']

    line_color = 'gray' if i >= 3 else color
    linewidth = 2.8 if i < 3 else 1.8
    linestyle = '-' if i < 3 else ':'

    if i < 3:
        plt.plot(res['fpr_main'], res['tpr_main'], color=color, lw=linewidth,
                 label=f'{method} (AUC={auc_val:.3f}, 95% CI [{auc_ci[0]:.3f}–{auc_ci[1]:.3f}])')
        plt.fill_between(res['fpr_grid'], res['tpr_lower'], res['tpr_upper'],
                         color=color, alpha=0.2)
    else:
        plt.plot(res['fpr_main'], res['tpr_main'], color=line_color, lw=linewidth, linestyle=linestyle,
                 label=f'{method} (AUC={auc_val:.3f}, 95% CI [{auc_ci[0]:.3f}–{auc_ci[1]:.3f}])')

# 设置坐标轴范围和标签
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=15)
plt.ylabel('True Positive Rate', fontsize=15)
plt.title('DWI - ROC Curves of 11 Models (Top-3 with 95% CI)', fontsize=25)
plt.legend(loc="lower right", fontsize=14, frameon=True, fancybox=True, shadow=True)
plt.grid(True, alpha=0.3, color='gray')

# ✅ 关键修复：设置白色背景
plt.gca().set_facecolor('white')

# ✅ 添加黑色外边框
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_color('black')
    spine.set_linewidth(1.0)

# 确保所有设置完成后再保存
plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
print(f"✅ ROC 图已保存至: {OUTPUT_PNG}")
plt.show()