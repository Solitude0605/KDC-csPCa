import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ==========================================
# 设置字体
# ==========================================
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Calibri']
# plt.rcParams['font.sans-serif'] = ['Calibri', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 11

# ==========================================
# 1. 读取 CSV 数据
# ==========================================
df = pd.read_csv(r'K:\PCa_2026\Article\放射组学\图表\Delong\多模态\内部测试\result\delong_pairwise_comparison.csv')

# ==========================================
# 2. 提取所有唯一模型，并确定每个模型的 AUC
# ==========================================
model_auc = {}
for _, row in df.iterrows():
    m1, m2 = row['Model1'], row['Model2']
    auc1, auc2 = row['AUC1'], row['AUC2']
    if pd.notna(auc1):
        model_auc[m1] = auc1
    if pd.notna(auc2):
        model_auc[m2] = auc2

models = []
seen = set()
for m in df['Model1'].tolist() + df['Model2'].tolist():
    if m not in seen:
        models.append(m)
        seen.add(m)

print("Models detected:", models)
print("AUC values:", {m: round(v, 4) for m, v in model_auc.items()})

# ==========================================
# 3. 模型名称缩写映射
# ==========================================
short_name_map = {
    'Knowledge-Driven Conditional': 'Our Model',
    'Data-Driven Conditional': 'Data-Driven',
    'DCE': 'DCE',
    'Clinical': 'Clinical',
    'DWI': 'DWI',
    'T2': 'T2'
}
short_models = [short_name_map.get(m, m) for m in models]

# ==========================================
# 4. 构建索引映射
# ==========================================
n = len(models)
idx_map = {m: i for i, m in enumerate(models)}

# ==========================================
# 5. 按 AUC 从高到低对模型排序（新增步骤）
# ==========================================
sorted_models = sorted(model_auc.items(), key=lambda x: x[1], reverse=True)
models = [m[0] for m in sorted_models]  # 重新赋值 models，按 AUC 降序
short_models = [short_name_map.get(m, m) for m in models]
n = len(models)
idx_map = {m: i for i, m in enumerate(models)}  # 重建 idx_map

# ==========================================
# 6. 初始化矩阵（原第4步内容，现在使用排序后的 models）
# ==========================================
auc_diff_matrix = np.zeros((n, n))
p_matrix = np.ones((n, n))
np.fill_diagonal(p_matrix, 1.0)

# 填充 AUC 差异（基于排序后的 models）
for i, m1 in enumerate(models):
    for j, m2 in enumerate(models):
        auc_diff_matrix[i, j] = model_auc[m1] - model_auc[m2]

# 填充 P 值（注意：df 中的 Model1/Model2 顺序不变，但 idx_map 已更新）
for _, row in df.iterrows():
    m1, m2, p = row['Model1'], row['Model2'], row['P_value']
    if m1 in idx_map and m2 in idx_map:  # 防御性检查
        i, j = idx_map[m1], idx_map[m2]
        p_matrix[i, j] = p
        p_matrix[j, i] = p

# ==========================================
# 5. 构建标注矩阵（含对角线文本）
# ==========================================
annot_matrix = np.empty((n, n), dtype='U30')
for i in range(n):
    for j in range(n):
        if i == j:
            # 对角线：显示 "模型名\n(AUC=xxx)"
            model_label = short_models[i]
            auc_val = model_auc[models[i]]
            annot_matrix[i, j] = f'{model_label}\n(AUC={auc_val:.3f})'
        elif i > j:
            diff = auc_diff_matrix[i, j]
            annot_matrix[i, j] = f'Diff_AUC\n {diff:.3f}'
        else:
            p = p_matrix[i, j]
            if p < 0.001:
                sig = '***'
            elif p < 0.01:
                sig = '**'
            elif p < 0.05:
                sig = '*'
            else:
                sig = 'ns'
            annot_matrix[i, j] = f'$\\mathit{{p}}$={p:.3f}\n({sig})'

# ==========================================
# 6. 创建自定义颜色映射
# ==========================================
cmap_lower = mcolors.LinearSegmentedColormap.from_list("custom_lower", ["#DEB887", "#DEB887"])
cmap_upper = sns.color_palette("Blues", as_cmap=True)
cmap_diag_color = "#006B10"

# ==========================================
# 7. 绘图：使用 pcolormesh + invert_yaxis 实现左下/右上布局
# ==========================================
# 动态计算 figsize，确保格子是正方形，并为图例留空间
cell_size = 1.6          # 可调：1.4~1.8 之间
heatmap_size = n * cell_size
fig_width = heatmap_size + 3.2   # 额外宽度给图例和颜色条
fig_height = heatmap_size

fig, ax = plt.subplots(figsize=(fig_width, fig_height + 1))

# 准备 p_matrix_clean
p_matrix_clean = np.nan_to_num(p_matrix, nan=1.0)
p_matrix_clean = np.clip(p_matrix_clean, 0, 1)

# 构建颜色矩阵（逻辑不变）
color_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i == j:
            color_matrix[i, j] = 0.5
        elif i < j:  # 数学上三角（将显示在视觉右上）
            color_matrix[i, j] = p_matrix_clean[i, j]
        else:        # 数学下三角（将显示在视觉左下）
            color_matrix[i, j] = 0.5

# 绘图
im = ax.pcolormesh(color_matrix, cmap=cmap_upper, vmin=0, vmax=1)

# 设置坐标和标签
ax.set_xlim(0, n)
ax.set_ylim(0, n)
ax.set_xticks([i + 0.5 for i in range(n)])
ax.set_yticks([i + 0.5 for i in range(n)])
ax.set_xticklabels(short_models, rotation=0, ha='center')
ax.set_yticklabels(short_models, rotation=90, ha='center', va='center')

# 👇 关键：让 (0,0) 在左下角
ax.invert_yaxis()

# 手动覆盖下三角（现在视觉左下）
for i in range(n):
    for j in range(i):  # i > j → 视觉左下
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor='#DEB887', edgecolor='none'))

# 对角线
for i in range(n):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, facecolor='#006B10', edgecolor='white', linewidth=1.5))

# 添加文字（坐标不变，invert_yaxis 自动映射）
for i in range(n):
    for j in range(n):
        if i == j:
            model_label = short_models[i]
            auc_val = model_auc[models[i]]
            ax.text(j + 0.5, i + 0.5,
                    f'{model_label}\n(AUC={auc_val:.3f})',
                    ha='center', va='center',
                    color='white', fontsize=15, weight='bold')
        elif i > j:
            diff = auc_diff_matrix[i, j]
            ax.text(j + 0.5, i + 0.5,
                    f'Diff_AUC\n{diff:.3f}',
                    ha='center', va='center',
                    color='black', fontsize=14)
        else:
            p = p_matrix[i, j]
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            ax.text(j + 0.5, i + 0.5,
                    f'$\\mathit{{p}}$={p:.3f}\n({sig})',
                    ha='center', va='center',
                    color='black', fontsize=14)

# # 标题放在热图上方
# ax.set_title(
#     'Delong Test for Internal Test Cohort: Our modal vs Other Multi-Modal Models\n'
#     '($\\mathit{p}$<0.001, $\\mathit{p}$<0.01, $\\mathit{p}$<0.05, ns = not significant)',
#     fontsize=20, fontweight='bold', pad=5
# )
ax.set_xlabel('Models', fontsize=17, fontweight='bold')
ax.set_ylabel('Models', fontsize=17, fontweight='bold')

# 颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Delong Test P-value')
cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.set_ticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])

# 图例放在热图上方，标题下方
legend_ax = fig.add_axes([0.2, 0.97, 0.6, 0.03])

# 添加图例
legend_elements = [
    Patch(facecolor='#DEB887', label='AUC Difference (Lower Triangle)'),
    Patch(facecolor='#006B10', edgecolor='white', linewidth=1.5, label='AUC Value (Diagonal)'),
    Patch(facecolor=sns.color_palette("Blues", as_cmap=True)(0.5), label='Delong Test P-value (Upper Triangle)')
]
legend_ax.legend(handles=legend_elements, loc='center', ncol=3, frameon=False, prop={'size': 14})  # 使用ncol调整列数
legend_ax.axis('off')  # 关闭图例坐标轴显示

# 设置整体布局
plt.subplots_adjust(top=0.88, bottom=0.10)  # 热图往下移，图例在顶部不重叠

ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('delong_pairwise_heatmap_Internal——1_test_multi_model.tif', dpi=300, bbox_inches='tight')
plt.show()