"""
使用 T2WI + DCE + DWI 多模态 Radiomics 特征融合预测 csPCa（临床显著性前列腺癌）
输入：
  - radiomics_features_t2.csv: T2WI 特征
  - radiomics_features_dce.csv: DCE 特征
  - radiomics_features_dwi.csv: DWI 特征
  - GT-20260104.csv: 金标准标签（含 patient_id 映射和 csPCa 标签）
输出：
  - 控制台打印所有模型性能...
任务：non-csPCa (0) vs csPCa (1)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, cohen_kappa_score, balanced_accuracy_score,
    roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
)
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
import warnings

plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore")


def safe_evaluate_model(X_raw, y, model, name, random_state=42):
    """安全评估单个模型（5折CV，每折内 Impute → VarianceFilter → Scale）"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    y_true_all, y_proba_all, auc_list = [], [], []

    for train_idx, val_idx in cv.split(X_raw, y):
        X_tr, X_val = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # 1. Imputation (median)
        imp = SimpleImputer(strategy='median')
        X_tr_imp = imp.fit_transform(X_tr)
        X_val_imp = imp.transform(X_val)

        # 2. Variance filter
        var_sel = VarianceThreshold(threshold=0.01)
        X_tr_var = var_sel.fit_transform(X_tr_imp)
        X_val_var = var_sel.transform(X_val_imp)

        # 3. Scaling
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_var)
        X_val_sc = scaler.transform(X_val_var)

        # 4. Train & Predict
        model.fit(X_tr_sc, y_tr)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val_sc)[:, 1]
        else:
            decision = model.decision_function(X_val_sc)
            proba = 1 / (1 + np.exp(-np.clip(decision, -100, 100)))

        y_true_all.append(y_val)
        y_proba_all.append(proba)
        auc_list.append(roc_auc_score(y_val, proba))

    # Aggregate
    y_true = np.concatenate(y_true_all)
    y_proba = np.concatenate(y_proba_all)
    y_pred = (y_proba > 0.5).astype(int)

    bal_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    auc_mean, auc_std = np.mean(auc_list), np.std(auc_list)

    print(f"\n📊 {name}: BalAcc={bal_acc:.3f}, Kappa={kappa:.3f}, AUC={auc_mean:.3f}±{auc_std:.3f}")

    return {
        'model': name,
        'Balanced Acc': bal_acc,
        'Kappa': kappa,
        'AUC': auc_mean,
        'AUC_std': auc_std,
        'y_true': y_true,
        'y_proba': y_proba,
        'y_pred': y_pred
    }


def main():
    # ==============================
    # 1. 加载 T2WI, DCE, DWI Radiomics 特征 和 GT 标签
    # ==============================
    print("🔍 正在加载 T2WI, DCE, DWI Radiomics 特征和 GT 标签...")

    import os
    import pandas as pd

    # === 1. 加载 Radiomics 特征 ===
    base_dir = r"K:\PCa_2025\8-Radiomics-PCa\data\radiomics_features_9th\radiomics_csv"
    modality_files = {
        'T2WI': "radiomics_features_t2.csv",
        'DCE': "radiomics_features_dce.csv",
        'DWI': "radiomics_features_dwi.csv"
    }

    dfs = {}
    for modality, filename in modality_files.items():
        path = os.path.join(base_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ {modality} 特征文件不存在: {path}")
        df = pd.read_csv(path)
        if 'patient_id' not in df.columns:
            raise ValueError(f"{modality} 文件缺少 'patient_id' 列")
        dfs[modality] = df
        print(f"✅ 成功加载 {len(df)} 行 {modality} 特征")

    # === 2. 加载 GT 标签 ===
    gt_path = r"K:\PCa_2025\8-Radiomics-PCa\data\GT-20260104.csv"
    df_gt = pd.read_csv(gt_path)

    required_cols = ['T2WI_patient_id', 'dce_patient_id', 'DWI_patient_id']
    # 自动识别标签列
    label_col = None
    for col in ['label', 'csPCa', 'gt', 'GT']:
        if col in df_gt.columns:
            label_col = col
            break
    if label_col is None:
        raise ValueError(f"❌ GT 文件中未找到标签列 (label/csPCa/gt/GT)")
    print(f"✅ 检测到标签列: {label_col}")
    for col in required_cols:
        if col not in df_gt.columns:
            raise ValueError(f"❌ GT 文件缺少必要列: {col}")

    print(f"✅ 成功加载 {len(df_gt)} 行 GT 标签")

    # === 3. 分别合并每个模态 ===
    df_combined = df_gt.copy()

    for modality in ['T2WI', 'DCE', 'DWI']:
        # GT 中对应的 patient ID 列名
        gt_id_col = f"{modality}_patient_id"
        # 特征表中的 patient_id 列
        feat_df = dfs[modality][['patient_id']]  # 先只取 ID，避免重复 merge 太多列
        # 合并：用 GT 的 XXX_patient_id 匹配特征表的 patient_id
        df_combined = pd.merge(
            df_combined,
            feat_df,
            left_on=gt_id_col,
            right_on='patient_id',
            how='inner'
        )
        # 删除临时的 'patient_id' 列（保留 GT 的原始 ID 列）
        df_combined = df_combined.drop(columns=['patient_id'])

    print(f"✅ 三模态 ID 对齐后剩余样本数: {len(df_combined)}")

    if len(df_combined) == 0:
        raise ValueError("❌ 无任何样本在三模态中同时存在！")

    # === 4. 再次合并完整的特征（现在确保 ID 一致）===
    # 重新 merge 完整特征（包含所有 radiomics 列）
    X_full = df_gt[['T2WI_patient_id', 'dce_patient_id', 'DWI_patient_id', 'TZ_5_score']].copy()

    for modality in ['T2WI', 'DCE', 'DWI']:
        gt_id_col = f"{modality}_patient_id"
        feat_df = dfs[modality]
        X_full = pd.merge(
            X_full,
            feat_df,
            left_on=gt_id_col,
            right_on='patient_id',
            how='inner'
        )
        X_full = X_full.drop(columns=['patient_id'])

    # 提取标签
    y = X_full[label_col].astype(int).values  # 0: non-csPCa, 1: csPCa
    print(f"✅ 标签分布:\n{pd.Series(y).value_counts().sort_index()}")

    # 提取 Radiomics 特征（排除非特征列）
    non_feature_cols = ['T2WI_patient_id', 'dce_patient_id', 'DWI_patient_id', 'label', 'csPCa', 'PZ_5_score', 'TZ_5_score', 'clinical_id']
    feature_cols = [col for col in X_full.columns if col not in non_feature_cols and col.startswith('original_')]

    if not feature_cols:
        raise ValueError("❌ 未找到任何以 'original_' 开头的 Radiomics 特征！")

    X_raw = X_full[feature_cols].copy()
    print(f"✅ 融合后总特征数: {X_raw.shape[1]}")

    # === 5. 选方差 Top-60 ===
    variances = X_raw.var()
    top_features = variances.nlargest(60).index.tolist()
    X_selected = X_raw[top_features].copy()
    print(f"✅ 选取方差最高的 {len(top_features)} 个特征")

    # # 方案 A：直接选全局方差最高的 Top-60（推荐，避免模态偏倚）
    # variances = X_temp.var()
    # top_features = variances.nlargest(60).index.tolist()  # 可调：60 或 100
    # X_raw = X_temp[top_features].copy()
    # print(f"✅ 选取方差最高的 {len(top_features)} 个多模态 Radiomics 特征")

    # ==============================
    # 2. 定义所有模型（与 clinical 脚本完全一致）
    # ==============================
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        "SVM (RBF)": SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, eval_metric='logloss',
            use_label_encoder=False, scale_pos_weight=len(y[y == 0]) / len(y[y == 1])
        ),
        "LightGBM": LGBMClassifier(n_estimators=200, class_weight='balanced', random_state=42, verbose=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        "Balanced Random Forest": BalancedRandomForestClassifier(n_estimators=200, random_state=42),
        "Naive Bayes": GaussianNB()
    }

    # ==============================
    # 3. 训练并评估所有模型
    # ==============================
    print("\n🧠 正在训练多个模型（5折交叉验证，安全流程）...")
    results = []
    model_items = list(models.items())
    for name, model in tqdm(model_items, desc="Running models", unit="model"):
        tqdm.write(f"▶️ Evaluating: {name}")
        try:
            res = safe_evaluate_model(X_raw, y, model, name)
            results.append(res)
        except Exception as e:
            print(f"❌ {name} 训练失败: {e}")

    if not results:
        raise RuntimeError("所有模型均训练失败！")

    # ==============================
    # 4. 输出最佳模型结果
    # ==============================
    best_model = max(results, key=lambda x: x['AUC'])
    print(f"\n🏆 最佳模型: {best_model['model']} (AUC = {best_model['AUC']:.3f})")

    # 分类报告
    print(f"\n📋 {best_model['model']} 分类报告:")
    target_names = ["non-csPCa", "csPCa"]
    print(classification_report(best_model['y_true'], best_model['y_pred'], target_names=target_names))

    # ==============================
    # 5. 绘制多模型 ROC 曲线
    # ==============================
    plt.figure(figsize=(9, 7))
    sorted_results = sorted(results, key=lambda x: x['AUC'], reverse=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_results)))
    for i, res in enumerate(sorted_results):
        fpr, tpr, _ = roc_curve(res['y_true'], res['y_proba'])
        label_text = f"{res['model']} (AUC={res['AUC']:.3f} ± {res['AUC_std']:.3f})"
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=label_text)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC=0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison of Radiomics Models (Sorted by AUC)')
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    roc_path = r"K:\PCa_2025\8-Radiomics-PCa\output\ROC\roc_curve_radiomics_only_sorted.tif"
    os.makedirs(os.path.dirname(roc_path), exist_ok=True)
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"\n📈 多模型 ROC 对比图已保存至: {roc_path}")
    plt.close()

    # ==============================
    # 6. 模型性能对比图（柱状图）
    # ==============================
    try:
        print("\n📊 正在生成模型性能对比图...")
        for res in results:
            y_true = res['y_true']
            y_pred = res['y_pred']
            res['Accuracy'] = accuracy_score(y_true, y_pred)
            res['Precision'] = precision_score(y_true, y_pred, zero_division=0)
            res['Recall'] = recall_score(y_true, y_pred, zero_division=0)
            res['F1-Score'] = f1_score(y_true, y_pred, zero_division=0)

        metrics = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#B7B7EB', '#9D9EA3', '#EAB883', '#9BBBE1', '#F09BA0']

        sorted_results = sorted(results, key=lambda x: x['AUC'])  # 升序
        model_names = [res['model'] for res in sorted_results]
        data = np.array([[res[metric] for metric in metrics] for res in sorted_results])

        fig, ax = plt.subplots(figsize=(14, 8))
        n_models = len(model_names)
        width = 0.15
        x = np.arange(n_models)
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            ax.bar(x - width * (len(metrics) // 2 - i), data[:, i], width, label=metric, color=color, edgecolor='black', linewidth=0.8)

        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylabel('Performance score', fontsize=12)
        ax.set_title('Model Performance Comparison (Radiomics, Sorted by AUC ↑)', fontsize=14, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=11)
        ax.set_ylim(0.5, 0.9)
        ax.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.8)

        for i, row in enumerate(data):
            for j, val in enumerate(row):
                ax.text(i - width * (len(metrics) // 2 - j), val + 0.01, f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10, frameon=True, fancybox=True, shadow=True)
        plt.tight_layout()

        perf_path = r"K:\PCa_2025\8-Radiomics-PCa\output\Performance\performance_comparison_radiomics_sorted.tif"
        os.makedirs(os.path.dirname(perf_path), exist_ok=True)
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✅ 模型性能对比图已保存至:\n{perf_path}")
    except Exception as e:
        print(f"\n❌ 性能对比图生成失败: {e}")
        import traceback
        traceback.print_exc()

    # ==============================
    # 7. 决策曲线分析（DCA）—— 按净获益面积排序
    # ==============================
    def calculate_net_benefit(y_true, y_proba, thresholds):
        n = len(y_true)
        if n == 0:
            return np.zeros_like(thresholds), 0
        net_benefits = []
        for t in thresholds:
            if t <= 0 or t >= 1:
                net_benefits.append(0.0)
                continue
            pred_positive = (y_proba >= t).astype(int)
            tp = np.sum((pred_positive == 1) & (y_true == 1))
            fp = np.sum((pred_positive == 1) & (y_true == 0))
            nb = (tp - fp * (t / (1 - t))) / n
            net_benefits.append(nb)
        net_benefit_area = np.trapz(net_benefits, thresholds)
        return np.array(net_benefits), net_benefit_area

    try:
        print("\n📊 正在生成决策曲线分析图...")
        thresholds = np.linspace(0.01, 0.99, 100)
        net_benefits = []
        net_benefit_areas = []
        for res in results:
            nb, area = calculate_net_benefit(res['y_true'], res['y_proba'], thresholds)
            net_benefits.append(nb)
            net_benefit_areas.append(area)

        sorted_indices = np.argsort(net_benefit_areas)[::-1]
        model_names_sorted = [results[i]['model'] for i in sorted_indices]
        net_benefits_sorted = [net_benefits[i] for i in sorted_indices]

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = [
            '#08306B', '#2171B5', '#6BAED6', '#C6DBEF', '#FED976',
            '#FFFFCC', '#969696', '#D95319', '#EDB120', '#77AC30'
        ]
        for i, (name, nb) in enumerate(zip(model_names_sorted, net_benefits_sorted)):
            color = colors[i % len(colors)]
            ax.plot(thresholds, nb, label=name, color=color, linewidth=2)

        # Treat None
        ax.plot(thresholds, np.zeros_like(thresholds), 'k--', label='Treat None', linewidth=2)
        # Treat All（假设人群患病率 0.5，可根据实际情况调整）
        true_population_prevalence = 0.5
        treat_all = true_population_prevalence - (1 - true_population_prevalence) * (thresholds / (1 - thresholds))
        ax.plot(thresholds, treat_all, 'k-', label=f'Treat All (prev={true_population_prevalence})', linewidth=2)

        ax.set_xlim(0, 1.0)
        ax.set_ylim(-0.1, 0.6)
        ax.grid(True, linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_xlabel('Threshold Probability', fontsize=12)
        ax.set_ylabel('Net Benefit', fontsize=12)
        ax.set_title('Decision Curve Analysis (Radiomics, Sorted by Net Benefit Area ↓)', fontsize=14, pad=30)
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=10, frameon=True, fancybox=True, shadow=True)
        plt.subplots_adjust(top=0.88)
        plt.tight_layout()

        dca_path = r"K:\PCa_2025\8-Radiomics-PCa\output\DCA\dca_radiomics_all_models.tif"
        os.makedirs(os.path.dirname(dca_path), exist_ok=True)
        plt.savefig(dca_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✅ 决策曲线分析图已保存至:\n{dca_path}")
    except Exception as e:
        print(f"\n❌ 决策曲线分析图生成失败: {e}")
        import traceback
        traceback.print_exc()

    # ==============================
    # 8. 雷达图（Top 所有模型）
    # ==============================
    try:
        print("\n📊 正在生成雷达图...")
        sorted_results = sorted(results, key=lambda x: x['AUC'], reverse=True)
        top_models = [res['model'] for res in sorted_results]
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        data = np.array([[res[metric] for metric in metrics] for res in sorted_results])

        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]

        colors = plt.cm.get_cmap('tab20', len(top_models)).colors
        for i, (model, values) in enumerate(zip(top_models, data)):
            values = np.concatenate((values, [values[0]]))
            ax.fill(angles, values, alpha=0.15, color=colors[i])
            ax.plot(angles, values, linewidth=2, label=model, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0.5, 0.9)
        ax.set_title('Radar Chart – Radiomics Models', fontsize=16, pad=40)
        ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1), fontsize=12)
        plt.tight_layout()

        radar_path = r"K:\PCa_2025\8-Radiomics-PCa\output\Radar\radar_radiomics_all_models.tif"
        os.makedirs(os.path.dirname(radar_path), exist_ok=True)
        plt.savefig(radar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✅ 雷达图已保存至:\n{radar_path}")
    except Exception as e:
        print(f"\n❌ 雷达图生成失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ DWI Radiomics 模型分析完成！")


if __name__ == "__main__":
    main()