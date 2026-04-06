# -*- coding: utf-8 -*-
"""
使用已训练的临床模型预测新数据
输入：新临床数据（需包含与训练时相同的特征）
输出：预测概率、预测类别、OOF数据及可解释性分析
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['axes.unicode_minus'] = False


def extract_numeric_id(s):
    """标准化ID提取函数（与训练时保持一致）"""
    return ''.join(c for c in str(s) if c.isdigit())


def preprocess_new_data(df_new, feature_names, categorical_cols):
    """
    预处理新数据，确保与训练时相同的处理流程
    """
    df_processed = df_new.copy()

    # 1. 检查缺失的必要列
    missing_cols = [col for col in feature_names if col not in df_processed.columns]
    if missing_cols:
        raise ValueError(f"新数据中缺少以下必要列: {missing_cols}")

    # 2. 只保留需要的列
    df_processed = df_processed[feature_names].copy()

    # 3. 对分类变量进行编码（使用与训练时相同的编码方式）
    for col in categorical_cols:
        if col in df_processed.columns:
            # 处理缺失值
            df_processed[col] = df_processed[col].fillna('Missing').astype(str)
            # 创建新编码器（实际应用中应加载训练时的编码器）
            le = LabelEncoder()
            # 获取唯一值并拟合
            unique_vals = df_processed[col].unique()
            le.fit(unique_vals)
            df_processed[col] = le.transform(df_processed[col])
            print(f"  - {col}: 使用新LabelEncoder编码，共{len(unique_vals)}个类别")

    # 4. 转换为数值类型
    df_processed = df_processed.apply(pd.to_numeric, errors='coerce')

    return df_processed


def predict_with_cv_models(models_per_fold, X_new):
    """
    使用交叉验证的所有折模型进行预测（集成预测）
    返回：平均预测概率、标准差、每折详细预测结果
    """
    all_probas = []
    fold_predictions = []  # 存储每折的详细信息

    for fold_idx, fold_data in enumerate(models_per_fold):
        print(f"  使用折 {fold_idx + 1} 的模型进行预测...")

        # 获取该折的预处理工具
        imputer = fold_data['imputer']
        var_selector = fold_data['var_selector']
        scaler = fold_data['scaler']
        model = fold_data['model']
        fold_metrics = fold_data.get('metrics', {})

        # 应用相同的预处理流程
        X_imp = imputer.transform(X_new)
        X_var = var_selector.transform(X_imp)
        X_sc = scaler.transform(X_var)

        # 预测
        if hasattr(model, "predict_proba"):
            probas = model.predict_proba(X_sc)[:, 1]
            decision_scores = None
        else:
            decision_scores = model.decision_function(X_sc)
            probas = 1 / (1 + np.exp(-np.clip(decision_scores, -100, 100)))

        all_probas.append(probas)

        # 保存该折的详细信息
        fold_info = {
            'fold_idx': fold_idx,
            'fold_metrics': fold_metrics,
            'probas': probas.tolist() if isinstance(probas, np.ndarray) else probas,
            'decision_scores': decision_scores.tolist() if decision_scores is not None else None,
            'model_type': type(model).__name__,
            'model_params': model.get_params() if hasattr(model, 'get_params') else {},
            'has_predict_proba': hasattr(model, "predict_proba"),
            'n_features': X_var.shape[1],
            'feature_mask': var_selector.get_support().tolist() if hasattr(var_selector, 'get_support') else None
        }
        fold_predictions.append(fold_info)

    # 计算统计量
    all_probas_array = np.array(all_probas)
    avg_probas = np.mean(all_probas_array, axis=0)
    proba_std = np.std(all_probas_array, axis=0) if len(all_probas) > 1 else np.zeros_like(avg_probas)
    proba_range = np.ptp(all_probas_array, axis=0) if len(all_probas) > 1 else np.zeros_like(avg_probas)

    return avg_probas, proba_std, proba_range, fold_predictions


def save_oof_data(oof_data, output_dir, model_name):
    """
    保存OOF数据到文件
    """
    print(f"\n💾 正在保存OOF数据...")

    avg_probas = np.array(oof_data['avg_probas'])
    proba_std = np.array(oof_data['proba_std'])

    # 创建OOF目录
    oof_dir = os.path.join(output_dir, "oof_data")
    Path(oof_dir).mkdir(parents=True, exist_ok=True)

    # 1. 保存完整的OOF数据为pickle格式
    oof_pickle_path = os.path.join(oof_dir, f"{model_name}_oof_predictions.pkl")
    joblib.dump(oof_data, oof_pickle_path)
    print(f"✅ OOF数据(Pickle)已保存到: {oof_pickle_path}")

    # 2. 保存为JSON格式（部分数据，便于查看）
    try:
        # 创建简化版本的OOF数据用于JSON
        # 在 JSON 部分使用 avg_probas（array）
        json_oof_data = {
            'model_name': model_name,
            'timestamp': oof_data['timestamp'],
            'n_samples': oof_data['n_samples'],
            'n_folds': len(oof_data['fold_predictions']),
            'prediction_summary': {
                'mean_probability': float(np.mean(avg_probas)),
                'std_probability': float(np.std(avg_probas)),
                'positive_rate': float(np.mean(avg_probas >= 0.5)),  # ✅
                'positive_count': int(np.sum(avg_probas >= 0.5)),
                'negative_count': int(np.sum(avg_probas < 0.5))
            },
            'fold_metrics_summary': []
        }

        # 添加每折的统计信息
        for fold_info in oof_data['fold_predictions']:
            fold_probas = fold_info['probas']
            fold_summary = {
                'fold_idx': fold_info['fold_idx'],
                'model_type': fold_info['model_type'],
                'n_samples': len(fold_probas),
                'mean_probability': float(np.mean(fold_probas)),
                'std_probability': float(np.std(fold_probas)),
                'positive_rate': float(np.mean(np.array(fold_probas) >= 0.5))
            }
            json_oof_data['fold_metrics_summary'].append(fold_summary)

        oof_json_path = os.path.join(oof_dir, f"{model_name}_oof_summary.json")
        with open(oof_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_oof_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ OOF摘要(JSON)已保存到: {oof_json_path}")

    except Exception as e:
        print(f"⚠️ 保存JSON摘要失败: {e}")

    # 3. 保存详细的CSV格式数据
    try:
        n_samples = oof_data['n_samples']
        n_folds = len(oof_data['fold_predictions'])

        # 创建基础DataFrame
        detailed_data = {
            'patient_id': oof_data['patient_ids'],
            'avg_pred_prob': oof_data['avg_probas'],
            'prob_std': oof_data['proba_std'],
            'prob_range': oof_data['proba_range'],
            'pred_class': (np.array(oof_data['avg_probas']) >= 0.5).astype(int).tolist(),
            'pred_label': ['Non-csPCa' if p < 0.5 else 'csPCa' for p in oof_data['avg_probas']]
        }

        # 添加每折的预测概率
        for fold_idx, fold_info in enumerate(oof_data['fold_predictions']):
            detailed_data[f'fold_{fold_idx + 1}_prob'] = fold_info['probas']

        # 创建DataFrame并保存
        detailed_df = pd.DataFrame(detailed_data)
        detailed_csv_path = os.path.join(oof_dir, f"{model_name}_detailed_oof_predictions.csv")
        detailed_df.to_csv(detailed_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 详细OOF数据(CSV)已保存到: {detailed_csv_path}")

        # 4. 保存每折的统计信息
        fold_stats = []
        for fold_info in oof_data['fold_predictions']:
            fold_probas = np.array(fold_info['probas'])
            fold_stats.append({
                'fold_idx': fold_info['fold_idx'] + 1,
                'model_type': fold_info['model_type'],
                'n_samples': len(fold_probas),
                'mean_probability': float(np.mean(fold_probas)),
                'std_probability': float(np.std(fold_probas)),
                'min_probability': float(np.min(fold_probas)),
                'max_probability': float(np.max(fold_probas)),
                'positive_rate': float(np.mean(fold_probas >= 0.5)),
                'positive_count': int(np.sum(fold_probas >= 0.5)),
                'negative_count': int(np.sum(fold_probas < 0.5)),
                'has_predict_proba': fold_info['has_predict_proba'],
                'n_features_used': fold_info['n_features']
            })

        fold_stats_df = pd.DataFrame(fold_stats)
        fold_stats_path = os.path.join(oof_dir, f"{model_name}_fold_statistics.csv")
        fold_stats_df.to_csv(fold_stats_path, index=False, encoding='utf-8-sig')
        print(f"✅ 折统计信息已保存到: {fold_stats_path}")

        # 5. 保存特征使用情况（如果可用）
        try:
            if 'feature_mask' in oof_data['fold_predictions'][0] and oof_data['fold_predictions'][0][
                'feature_mask'] is not None:
                feature_masks = []
                feature_names = oof_data['feature_names']

                for fold_idx, fold_info in enumerate(oof_data['fold_predictions']):
                    mask = fold_info['feature_mask']
                    if mask and len(mask) == len(feature_names):
                        for feat_idx, (feat_name, is_selected) in enumerate(zip(feature_names, mask)):
                            if is_selected:
                                feature_masks.append({
                                    'fold_idx': fold_idx + 1,
                                    'feature_name': feat_name,
                                    'feature_index': feat_idx,
                                    'is_selected': is_selected
                                })

                if feature_masks:
                    feature_mask_df = pd.DataFrame(feature_masks)
                    feature_mask_path = os.path.join(oof_dir, f"{model_name}_feature_selection.csv")
                    feature_mask_df.to_csv(feature_mask_path, index=False, encoding='utf-8-sig')
                    print(f"✅ 特征选择情况已保存到: {feature_mask_path}")

        except Exception as e:
            print(f"⚠️ 保存特征选择信息失败: {e}")

    except Exception as e:
        print(f"⚠️ 保存详细CSV数据失败: {e}")

    return oof_dir


def generate_oof_report(oof_data, output_dir, model_name):
    """生成OOF数据报告"""
    print("\n📋 生成OOF数据报告...")

    report_content = f"""
    ===========================================
    OOF (Out-of-Fold) 预测数据报告
    ===========================================
    模型名称: {model_name}
    生成时间: {oof_data['timestamp']}
    样本数量: {oof_data['n_samples']}
    交叉验证折数: {len(oof_data['fold_predictions'])}

    --- 预测统计摘要 ---
    平均预测概率: {np.mean(oof_data['avg_probas']):.4f}
    预测概率标准差: {np.std(oof_data['avg_probas']):.4f}
    阳性预测数 (csPCa): {np.sum(np.array(oof_data['avg_probas']) >= 0.5)}
    阴性预测数 (Non-csPCa): {np.sum(np.array(oof_data['avg_probas']) < 0.5)}
    阳性率: {np.mean(np.array(oof_data['avg_probas']) >= 0.5) * 100:.1f}%

    --- 预测概率分布 ---
    最小值: {np.min(oof_data['avg_probas']):.4f}
    最大值: {np.max(oof_data['avg_probas']):.4f}
    25%分位数: {np.percentile(oof_data['avg_probas'], 25):.4f}
    中位数: {np.median(oof_data['avg_probas']):.4f}
    75%分位数: {np.percentile(oof_data['avg_probas'], 75):.4f}

    --- 模型一致性分析 ---
    预测概率标准差平均值: {np.mean(oof_data['proba_std']):.4f}
    预测概率范围平均值: {np.mean(oof_data['proba_range']):.4f}

    --- 每折预测统计 ---
    """

    for fold_info in oof_data['fold_predictions']:
        fold_probas = np.array(fold_info['probas'])
        report_content += f"""
    折 {fold_info['fold_idx'] + 1}:
      模型类型: {fold_info['model_type']}
      样本数: {len(fold_probas)}
      平均概率: {np.mean(fold_probas):.4f}
      标准差: {np.std(fold_probas):.4f}
      阳性率: {np.mean(fold_probas >= 0.5) * 100:.1f}%
      使用特征数: {fold_info['n_features']}
        """

    report_content += f"""

    --- 数据文件说明 ---
    1. {model_name}_oof_predictions.pkl: 完整的OOF数据（Pickle格式）
    2. {model_name}_oof_summary.json: OOF数据摘要（JSON格式）
    3. {model_name}_detailed_oof_predictions.csv: 详细的OOF预测结果
    4. {model_name}_fold_statistics.csv: 每折的统计信息

    --- 使用说明 ---
    这些OOF数据可用于：
    1. SHAP分析：解释模型预测
    2. 模型校准：检查预测概率的可靠性
    3. 不确定性分析：分析模型预测的一致性
    4. 后续研究：如元分析或模型集成
    """

    # 保存报告
    report_path = os.path.join(output_dir, f"{model_name}_oof_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"✅ OOF报告已保存到: {report_path}")

    return report_path


def visualize_oof_data(oof_data, output_dir, model_name):
    print("\n📈 正在生成OOF数据可视化...")

    try:
        viz_dir = os.path.join(output_dir, "visualization")
        Path(viz_dir).mkdir(parents=True, exist_ok=True)

        # 🔧 关键修复：确保是 numpy array
        avg_probas = np.array(oof_data['avg_probas'])
        proba_std = np.array(oof_data['proba_std'])
        n_folds = len(oof_data['fold_predictions'])

        # 1. 预测概率分布图（带不确定性）
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.hist(avg_probas, bins=20, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Decision threshold (0.5)')

        # 添加核密度估计
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(avg_probas)
        x_range = np.linspace(min(avg_probas), max(avg_probas), 1000)
        plt.plot(x_range, kde(x_range), 'r-', linewidth=2, alpha=0.7, label='Density')

        plt.xlabel('Predicted probability')
        plt.ylabel('Density')
        plt.title(f'{model_name} - Predicted Probability Distribution')
        plt.legend()
        plt.grid(alpha=0.3)

        # 2. 模型一致性图
        plt.subplot(1, 2, 2)

        # 按概率排序
        sorted_idx = np.argsort(avg_probas)
        sorted_probas = np.array(avg_probas)[sorted_idx]
        sorted_std = np.array(proba_std)[sorted_idx]

        plt.plot(sorted_probas, 'b-', linewidth=2, alpha=0.7, label='Mean probability')
        plt.fill_between(range(len(sorted_probas)),
                         sorted_probas - sorted_std,
                         sorted_probas + sorted_std,
                         alpha=0.3, color='blue', label='±1 std')
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Decision threshold')

        plt.xlabel('Samples (sorted by probability)')
        plt.ylabel('Predicted probability')
        plt.title(f'{model_name} - Model Consistency Across Folds')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{model_name}_oof_distribution_consistency.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        # 3. 每折预测对比图（如果折数合适）
        if n_folds <= 5:  # 只绘制最多5折的对比
            plt.figure(figsize=(10, 8))

            # 准备每折的数据
            fold_data = []
            for fold_info in oof_data['fold_predictions']:
                fold_probas = np.array(fold_info['probas'])
                fold_data.append(fold_probas)

            # 创建箱线图
            positions = np.arange(1, n_folds + 1)
            bp = plt.boxplot(fold_data, positions=positions, patch_artist=True)

            # 设置颜色
            colors = plt.cm.Set3(np.linspace(0, 1, n_folds))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.7)
            plt.xlabel('Fold Number')
            plt.ylabel('Predicted Probability')
            plt.title(f'{model_name} - Probability Distribution by Fold')
            plt.xticks(positions, [f'Fold {i + 1}' for i in range(n_folds)])
            plt.grid(alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"{model_name}_oof_by_fold.png"),
                        dpi=300, bbox_inches='tight')
            plt.close()

        # 4. 不确定性分析图
        plt.figure(figsize=(10, 6))

        # 计算每个样本的不确定性（标准差）
        uncertainty = proba_std

        # 按不确定性排序
        uncertainty_sorted_idx = np.argsort(uncertainty)
        uncertainty_sorted = uncertainty[uncertainty_sorted_idx]
        probas_sorted = np.array(avg_probas)[uncertainty_sorted_idx]

        plt.scatter(range(len(uncertainty_sorted)), uncertainty_sorted,
                    c=probas_sorted, cmap='coolwarm', alpha=0.6, s=20)
        plt.colorbar(label='Predicted Probability')

        # 标记高不确定性区域
        high_uncertainty_threshold = np.percentile(uncertainty, 90)
        high_uncertainty_idx = np.where(uncertainty_sorted > high_uncertainty_threshold)[0]
        if len(high_uncertainty_idx) > 0:
            plt.scatter(high_uncertainty_idx, uncertainty_sorted[high_uncertainty_idx],
                        c='red', alpha=0.8, s=30, marker='x', label=f'Top 10% uncertain')

        plt.axhline(y=high_uncertainty_threshold, color='red', linestyle=':',
                    alpha=0.5, label='90th percentile')
        plt.xlabel('Samples (sorted by uncertainty)')
        plt.ylabel('Uncertainty (Standard Deviation)')
        plt.title(f'{model_name} - Prediction Uncertainty Analysis')
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{model_name}_oof_uncertainty_analysis.png"),
                    dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✅ OOF可视化图表已保存到: {viz_dir}")

    except Exception as e:
        print(f"⚠️ OOF可视化失败: {e}")


def main():
    # ==============================
    # 1. 配置路径
    # ==============================
    MODEL_DIR = r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\4rd\Clinical_only"

    # 模型文件路径
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model_SVM_(RBF).pkl")
    REP_MODEL_PATH = os.path.join(MODEL_DIR, "representative_model_SVM_RBF.pkl")

    # 新数据路径（根据实际情况修改）
    NEW_DATA_PATH = r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\aligned_results\clinical_aligned.csv"

    # 输出目录
    OUTPUT_DIR = os.path.join(MODEL_DIR, r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\predictions_new\Clinical_only")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print(f"📁 模型目录: {MODEL_DIR}")
    print(f"📁 新数据路径: {NEW_DATA_PATH}")
    print(f"📁 输出目录: {OUTPUT_DIR}")

    # ==============================
    # 2. 加载模型和数据
    # ==============================
    print("\n📂 正在加载模型...")

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(f"未找到模型文件: {BEST_MODEL_PATH}")

    # 加载最佳模型数据
    best_model_data = joblib.load(BEST_MODEL_PATH)
    model_name = best_model_data['model_name']
    feature_names = best_model_data['feature_names']
    categorical_cols = best_model_data['categorical_cols']
    all_candidate_cols = best_model_data['clinical_cols']

    print(f"✅ 加载模型: {model_name}")
    print(f"✅ 特征数量: {len(feature_names)}")
    print(f"✅ 分类变量: {categorical_cols}")

    # 加载新数据
    df_new = pd.read_csv(NEW_DATA_PATH)

    # ==============================
    # 关键修改1：提取GT列
    # ==============================
    # 定义可能的GT列名
    gt_candidates = ['PI_RADS', 'ground_truth', 'label', 'y_true', 'target', 'gt']

    gt_col = None
    y_true = None

    for col in gt_candidates:
        if col in df_new.columns:
            gt_col = col
            y_true = df_new[col].values
            print(f"✅ 发现真实标签列: {gt_col}")
            print(f"  标签分布: {pd.Series(y_true).value_counts().to_dict()}")
            break

    if y_true is None:
        print("⚠️ 警告：未找到真实标签列！融合测试将无法计算性能指标")
        print("⚠️ 如果这是测试数据，请确保CSV包含GT列")
        print("⚠️ 如果这是无标签数据，融合测试只能做预测，不能评估")

    # ... 预处理数据时要排除GT列 ...
    if gt_col and gt_col in feature_names:
        feature_names = [f for f in feature_names if f != gt_col]
        print(f"  已从特征中排除GT列: {gt_col}")

    # ==============================
    # 3. 预处理新数据
    # ==============================
    print("\n🔄 正在预处理新数据...")

    # 检查是否包含所有候选列（用于参考）
    missing_candidate_cols = [col for col in all_candidate_cols if col not in df_new.columns]
    if missing_candidate_cols:
        print(f"⚠️ 注意：新数据缺少以下候选列（可能影响后续分析）: {missing_candidate_cols}")

    # 预处理数据（只使用实际用于训练的特征）
    X_new_processed = preprocess_new_data(df_new, feature_names, categorical_cols)

    print(f"✅ 预处理后数据形状: {X_new_processed.shape}")

    # ==============================
    # 4. 进行预测（获取OOF数据）
    # ==============================
    print("\n🔮 正在进行预测并生成OOF数据...")

    # 使用交叉验证模型进行预测，获取详细的OOF数据
    models_per_fold = best_model_data['all_folds_models']
    avg_probas, proba_std, proba_range, fold_predictions = predict_with_cv_models(
        models_per_fold, X_new_processed.values)

    # 转换为预测类别（使用0.5作为阈值）
    preds = (avg_probas >= 0.5).astype(int)

    # ==============================
    # 5. 准备OOF数据
    # ==============================
    print("\n📊 准备OOF数据结构...")

    # 获取患者ID
    patient_ids = df_new['patient_id'].values if 'patient_id' in df_new.columns else [f"PAT_{i + 1:03d}" for i in
                                                                                      range(len(df_new))]

    # 构建OOF数据结构（修改后包含GT）
    oof_data = {
        'model_name': model_name,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'feature_names': feature_names,
        'n_samples': len(avg_probas),
        'patient_ids': patient_ids.tolist() if isinstance(patient_ids, np.ndarray) else patient_ids,
        'avg_probas': avg_probas.tolist() if isinstance(avg_probas, np.ndarray) else avg_probas,
        'proba_std': proba_std.tolist() if isinstance(proba_std, np.ndarray) else proba_std,
        'proba_range': proba_range.tolist() if isinstance(proba_range, np.ndarray) else proba_range,
        'pred_classes': preds.tolist() if isinstance(preds, np.ndarray) else preds,
        'fold_predictions': fold_predictions,
        'model_info': {
            'model_type': 'SVM_RBF',
            'n_folds': len(models_per_fold),
            'feature_count': len(feature_names),
            'categorical_features': categorical_cols,
            'prediction_threshold': 0.5
        },
        # 添加GT数据（如果有）
        'y_true': y_true.tolist() if y_true is not None else None,
        'ground_truth': y_true.tolist() if y_true is not None else None,
        'ground_truth_column': gt_col,
        'has_ground_truth': y_true is not None,
    }

    # ==============================
    # 关键修改3：如果有多折数据，验证GT是否一致
    # ==============================
    if y_true is not None:
        # 计算准确率等指标
        accuracy = np.mean(preds == y_true)
        # 计算 AUC（推荐方式）
        try:
            roc_auc = roc_auc_score(y_true, avg_probas)
            ap = average_precision_score(y_true, avg_probas)
        except Exception as e:
            print(f"⚠️ 计算AUC失败: {e}")
            roc_auc = None
            ap = None

        # 计算ROC
        fpr, tpr, _ = roc_curve(y_true, avg_probas)

        # 更详细的评估指标

        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值

        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # 绘制并保存 ROC 曲线
        roc_path = os.path.join(OUTPUT_DIR, f"{model_name}_roc_curve.png")
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})' if roc_auc is not None else 'ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)

        # 保存图像
        plt.savefig(roc_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # 👈 非常重要！避免内存泄漏或图形叠加

        print(f"✅ ROC曲线已保存至: {roc_path}")
        print(f"\n📊 与GT对齐结果:")
        print("=" * 60)
        print(f"  样本总数: {len(y_true)}")
        print(f"  真实阳性/阴性: {y_true.sum()}/{len(y_true) - y_true.sum()}")
        print(f"  预测阳性/阴性: {preds.sum()}/{len(preds) - preds.sum()}")
        print(f"\n  混淆矩阵:")
        print(f"             预测")
        print(f"           0     1")
        print(f"      0   {tn:4d}  {fp:4d}")
        print(f" 真实 1   {fn:4d}  {tp:4d}")
        print(f"\n  性能指标:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  灵敏度(召回率): {sensitivity:.4f}")
        print(f"  特异度: {specificity:.4f}")
        print(f"  阳性预测值: {ppv:.4f}")
        print(f"  阴性预测值: {npv:.4f}")
        print("=" * 60)

        # 如果有概率，还可以计算AUC
        if len(avg_probas) == len(y_true):
            try:
                auc = roc_auc_score(y_true, avg_probas)
                ap = average_precision_score(y_true, avg_probas)
                print(f"  AUC: {auc:.4f}")
                print(f"  Average Precision: {ap:.4f}")

                # 添加到oof_data中
                oof_data['performance_metrics'] = {
                    'accuracy': float(accuracy),
                    'sensitivity': float(sensitivity),
                    'specificity': float(specificity),
                    'ppv': float(ppv),
                    'npv': float(npv),
                    'auc': float(auc),
                    'average_precision': float(ap)
                }
            except Exception as e:
                print(f"⚠️ 计算AUC失败: {e}")

    # ==============================
    # 6. 保存OOF数据
    # ==============================
    oof_dir = save_oof_data(oof_data, OUTPUT_DIR, model_name)

    # 生成OOF报告
    oof_report_path = generate_oof_report(oof_data, oof_dir, model_name)

    # 可视化OOF数据
    visualize_oof_data(oof_data, oof_dir, model_name)

    # ==============================
    # 7. 保存标准预测结果
    # ==============================
    print("\n💾 正在保存标准预测结果...")

    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'pred_prob_ensemble': avg_probas,
        'pred_prob_std': proba_std,
        'pred_prob_range': proba_range,
        'pred_class': preds,
        'pred_label': ['Non-csPCa' if p == 0 else 'csPCa' for p in preds]
    })

    # 添加置信度评估
    results_df['confidence_level'] = pd.cut(
        results_df['pred_prob_ensemble'],
        bins=[0, 0.3, 0.7, 1.0],
        labels=['低置信度', '中等置信度', '高置信度']
    )

    # 添加不确定性标记
    results_df['uncertainty_level'] = pd.cut(
        results_df['pred_prob_std'],
        bins=[0, 0.05, 0.15, 1.0],
        labels=['低不确定性', '中等不确定性', '高不确定性']
    )

    # 保存结果
    results_path = os.path.join(OUTPUT_DIR, f"{model_name}_predictions.csv")
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"✅ 标准预测结果已保存到: {results_path}")

    # ==============================
    # 8. 生成高风险样本列表
    # ==============================
    print("\n⚠️ 高风险样本分析...")

    # 定义高风险样本：预测为csPCa且置信度高的样本
    high_risk_samples = results_df[
        (results_df['pred_class'] == 1) &
        (results_df['confidence_level'] == '高置信度') &
        (results_df['uncertainty_level'] == '低不确定性')
        ].copy()

    if len(high_risk_samples) > 0:
        high_risk_samples = high_risk_samples.sort_values('pred_prob_ensemble', ascending=False)
        print(f"发现 {len(high_risk_samples)} 个高风险样本:")
        print(high_risk_samples[
                  ['patient_id', 'pred_prob_ensemble', 'confidence_level', 'uncertainty_level']].to_string())

        # 保存高风险样本列表
        high_risk_path = os.path.join(OUTPUT_DIR, "high_risk_samples.csv")
        high_risk_samples.to_csv(high_risk_path, index=False, encoding='utf-8-sig')
        print(f"✅ 高风险样本列表已保存到: {high_risk_path}")
    else:
        print("未发现高风险样本")

    # ==============================
    # 9. 生成不确定性样本列表
    # ==============================
    print("\n⚠️ 不确定性样本分析...")

    uncertain_samples = results_df[
        (results_df['uncertainty_level'] == '高不确定性') |
        ((results_df['pred_prob_ensemble'] >= 0.4) & (results_df['pred_prob_ensemble'] <= 0.6))
        ].copy()

    if len(uncertain_samples) > 0:
        uncertain_samples = uncertain_samples.sort_values('pred_prob_std', ascending=False)
        print(f"发现 {len(uncertain_samples)} 个不确定样本:")
        print(uncertain_samples[['patient_id', 'pred_prob_ensemble', 'pred_prob_std', 'uncertainty_level']].to_string())

        # 保存不确定样本列表
        uncertain_path = os.path.join(OUTPUT_DIR, "uncertain_samples.csv")
        uncertain_samples.to_csv(uncertain_path, index=False, encoding='utf-8-sig')
        print(f"✅ 不确定样本列表已保存到: {uncertain_path}")
    else:
        print("未发现高度不确定样本")

    # ==============================
    # 10. 生成汇总统计
    # ==============================
    print("\n📋 预测结果汇总:")
    print("=" * 60)
    print(f"总样本数: {len(results_df)}")
    print(f"Non-csPCa 预测数: {(results_df['pred_class'] == 0).sum()}")
    print(f"csPCa 预测数: {(results_df['pred_class'] == 1).sum()}")
    print(f"平均预测概率: {results_df['pred_prob_ensemble'].mean():.3f}")
    print(f"平均预测标准差: {results_df['pred_prob_std'].mean():.3f}")
    print(f"高置信度样本比例: {(results_df['confidence_level'] == '高置信度').mean() * 100:.1f}%")
    print(f"低不确定性样本比例: {(results_df['uncertainty_level'] == '低不确定性').mean() * 100:.1f}%")
    print("=" * 60)

    # ==============================
    # 11. 创建标准化的OOF文件供其他脚本使用（新增）
    # ==============================
    print("\n💾 创建标准化Clinical OOF文件...")

    try:
        # 创建标准化的OOF数据结构，便于其他脚本读取
        standardized_oof_data = {
            'modality': 'Clinical',
            'model_name': model_name,
            'timestamp': oof_data['timestamp'],
            'patient_ids': oof_data['patient_ids'],
            'pred_probabilities': oof_data['avg_probas'],
            'y_proba': oof_data['avg_probas'],  # 兼容不同脚本的命名
            'probas': oof_data['avg_probas'],  # 兼容不同脚本的命名
            'pred_classes': oof_data['pred_classes'],
            'pred_labels': ['Non-csPCa' if p == 0 else 'csPCa' for p in oof_data['pred_classes']],
            'fold_details': [
                {
                    'fold_idx': fold_info['fold_idx'],
                    'probas': fold_info['probas'],
                    'model_type': fold_info['model_type']
                }
                for fold_info in oof_data['fold_predictions']
            ],
            'metadata': {
                'n_samples': oof_data['n_samples'],
                'n_folds': len(oof_data['fold_predictions']),
                'mean_probability': float(np.mean(oof_data['avg_probas'])),
                'positive_rate': float(np.mean(np.array(oof_data['pred_classes']) == 1)),
                'positive_count': int(np.sum(np.array(oof_data['pred_classes']) == 1)),
                'negative_count': int(np.sum(np.array(oof_data['pred_classes']) == 0))
            }
        }

        # 保存为标准化名称的OOF文件
        clinical_oof_path = os.path.join(OUTPUT_DIR, "clinical_oof_predictions.pkl")
        joblib.dump(standardized_oof_data, clinical_oof_path)
        print(f"✅ 标准化Clinical OOF文件已保存到: {clinical_oof_path}")

        # 同时保存一份到Test-Ex目录下，便于融合测试脚本使用
        test_ex_dir = r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex"
        Path(test_ex_dir).mkdir(parents=True, exist_ok=True)  # 确保目录存在
        clinical_test_ex_path = os.path.join(test_ex_dir, "clinical_oof_predictions.pkl")
        joblib.dump(standardized_oof_data, clinical_test_ex_path)
        print(f"✅ 复制到Test-Ex目录: {clinical_test_ex_path}")

        # 创建简化的CSV版本
        clinical_csv_data = pd.DataFrame({
            'patient_id': standardized_oof_data['patient_ids'],
            'clinical_proba': standardized_oof_data['pred_probabilities'],
            'clinical_pred': standardized_oof_data['pred_classes'],
            'clinical_label': standardized_oof_data['pred_labels']
        })

        clinical_csv_path = os.path.join(OUTPUT_DIR, "clinical_oof_predictions.csv")
        clinical_csv_data.to_csv(clinical_csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ Clinical OOF CSV文件已保存到: {clinical_csv_path}")

        # 同时保存一份到Test-Ex目录
        clinical_csv_test_ex_path = os.path.join(test_ex_dir, "clinical_oof_predictions.csv")
        clinical_csv_data.to_csv(clinical_csv_test_ex_path, index=False, encoding='utf-8-sig')
        print(f"✅ Clinical OOF CSV复制到Test-Ex目录: {clinical_csv_test_ex_path}")

        # 为融合测试创建专用的数据文件
        fusion_input_data = {
            'patient_id': standardized_oof_data['patient_ids'],
            'y_proba': standardized_oof_data['pred_probabilities'],
            'y_true': None,  # 如果没有真实标签
            'modality': 'Clinical',
            'model_info': {
                'model_name': model_name,
                'model_type': best_model_data.get('model_type', 'SVM_RBF'),
                'feature_names': feature_names
            }
        }

        fusion_input_path = os.path.join(OUTPUT_DIR, "clinical_for_fusion.pkl")
        joblib.dump(fusion_input_data, fusion_input_path)
        print(f"✅ 融合测试专用Clinical数据已保存到: {fusion_input_path}")

        # 生成标准化说明文档
        oof_info_doc = f"""
        Clinical OOF数据文件说明
        =========================

        文件: clinical_oof_predictions.pkl

        数据结构:
        ---------
        {{
            'modality': 'Clinical',
            'model_name': '模型名称',
            'timestamp': '生成时间',
            'patient_ids': ['患者ID列表'],
            'pred_probabilities': [预测概率列表],
            'y_proba': [预测概率列表 - 兼容名称],
            'probas': [预测概率列表 - 兼容名称],
            'pred_classes': [预测类别列表 (0/1)],
            'pred_labels': ['Non-csPCa'/'csPCa'列表],
            'fold_details': [每折详细信息],
            'metadata': {{
                'n_samples': 样本数,
                'n_folds': 交叉验证折数,
                'mean_probability': 平均预测概率,
                'positive_rate': 阳性率,
                'positive_count': 阳性样本数,
                'negative_count': 阴性样本数
            }}
        }}

        使用方式:
        --------
        1. 加载文件:
           import joblib
           data = joblib.load('clinical_oof_predictions.pkl')

        2. 获取预测概率:
           probas = data['pred_probabilities']  # 或 data['y_proba'], data['probas']

        3. 获取患者ID:
           patient_ids = data['patient_ids']

        4. 获取预测类别:
           pred_classes = data['pred_classes']

        5. 获取元数据:
           metadata = data['metadata']

        适用于:
        - 多模态融合测试
        - 模型性能对比
        - 后续分析研究

        生成时间: {standardized_oof_data['timestamp']}
        样本数量: {standardized_oof_data['metadata']['n_samples']}
        """

        oof_info_path = os.path.join(OUTPUT_DIR, "clinical_oof_file_info.txt")
        with open(oof_info_path, 'w', encoding='utf-8') as f:
            f.write(oof_info_doc)
        print(f"✅ OOF文件说明文档已保存到: {oof_info_path}")

    except Exception as e:
        print(f"⚠️ 创建标准化OOF文件失败: {e}")

    print(f"\n✅ 新数据预测完成！")
    print(f"📁 所有结果保存在: {OUTPUT_DIR}")
    print(f"📄 主要文件:")
    print(f"  - 标准预测结果: {results_path}")
    print(f"  - OOF数据目录: {oof_dir}/")
    print(f"    • 完整OOF数据: {model_name}_oof_predictions.pkl")
    print(f"    • OOF摘要: {model_name}_oof_summary.json")
    print(f"    • 详细OOF结果: {model_name}_detailed_oof_predictions.csv")
    print(f"    • OOF报告: {model_name}_oof_report.txt")
    print(f"  - 标准化Clinical OOF文件: clinical_oof_predictions.pkl")
    print(f"  - Clinical OOF CSV文件: clinical_oof_predictions.csv")
    if 'high_risk_samples' in locals() and len(high_risk_samples) > 0:
        print(f"  - 高风险样本: {high_risk_path}")
    if 'uncertain_samples' in locals() and len(uncertain_samples) > 0:
        print(f"  - 不确定样本: {uncertain_path}")


if __name__ == "__main__":
    main()