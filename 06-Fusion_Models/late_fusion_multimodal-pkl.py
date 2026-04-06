# -*- coding: utf-8 -*-
"""
改进版多模态后融合（Late Fusion）脚本 - CSV版本
直接从CSV文件读取预测结果，支持多模态后融合分析
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, train_test_split, cross_val_predict
)
from matplotlib.lines import Line2D
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, cohen_kappa_score,
    classification_report, roc_curve, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import re
import json
from glob import glob
import seaborn as sns

warnings.filterwarnings('ignore')

# 设置路径
OUTPUT_DIR = r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\ML\clinical4-t23"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# 全局颜色方案
# ==============================
colors = {
    'Clinical_only': '#F27873',  # 红色
    'T2_only': '#FFD373',  # 黄色
    'dce_only': '#008A45',  # 绿色
    'DWI_only': '#80C5A2',  # 浅绿
    'AUC_weighted': '#468BCA',  # 蓝色
    'Stacking_LR': '#7DD2F6',  # 浅蓝
    'Stacking_RF': '#B384BA',  # 紫色
    'Heuristic': '#D9C2DD',  # 浅紫
    'Conditional_T2_dceDWI': '#9467BD',  # 紫色系 - 条件融合1
    'Conditional_T2_LR': '#DEAA82',  # 橙色系 - 条件融合2
    'Conditional_Clinical_T2_dceDWI': '#8A2BE2',  # 蓝紫色 - 新增
    'Conditional_Clinical_T2_LR': '#FF8C00'  # 深橙色 - 新增
}


# ==============================
# 辅助函数定义
# ==============================

def _get_method_description(method):
    """获取方法的详细描述"""
    descriptions = {
        'AUC_weighted': 'AUC-based weighted fusion: weights are proportional to each modality\'s AUC performance',
        'Stacking_LR': 'Stacking ensemble with Logistic Regression as meta-learner',
        'Stacking_RF': 'Stacking ensemble with Random Forest as meta-learner',
        'Heuristic': 'Fixed heuristic weights based on clinical importance (Clinical:0.3, T2:0.5, DCE:0.1, DWI:0.1)',
        'Conditional_Clinical_T2_dceDWI': 'KDC (Knowledge-Driven Conditional) fusion: When both Clinical (p1∈[0.3,0.7]) and T2 (p2∈[0.4,0.6]) are uncertain, uses Stacking LR; otherwise simple average of all 4 modalities',
        'Conditional_Clinical_T2_LR': 'Conditional fusion with LR fine-tuning: When both Clinical and T2 are uncertain, uses Logistic Regression fine-tuning'
    }
    return descriptions.get(method, 'Fusion method')


def _get_method_parameters(method, simplified_modality_names):
    """获取方法参数"""
    if method == 'AUC_weighted':
        return {'weight_calculation': 'normalized_AUC', 'normalization': 'sum_to_1'}
    elif method == 'Stacking_LR':
        return {
            'meta_learner': 'LogisticRegression',
            'parameters': {'C': 'optimized_by_inner_CV', 'class_weight': 'balanced', 'solver': 'liblinear'}
        }
    elif method == 'Stacking_RF':
        return {
            'meta_learner': 'RandomForestClassifier',
            'parameters': {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 5}
        }
    elif method == 'Heuristic':
        return {
            'weights': dict(zip(simplified_modality_names, [0.3, 0.5, 0.1, 0.1])),
            'rationale': 'Clinical importance and diagnostic value'
        }
    elif method == 'Conditional_T2_dceDWI':
        return {
            'thresholds': {'low': 0.4, 'high': 0.6},
            'uncertain_weights': {'T2': 0.6, 'DCE': 0.2, 'DWI': 0.2},
            'certain_strategy': 'use_T2_only'
        }
    elif method == 'Conditional_T2_LR':
        return {
            'thresholds': {'low': 0.4, 'high': 0.6},
            'fine_tuning': 'LogisticRegression_on_uncertain_samples',
            'parameter_tuning': 'inner_CV_on_uncertain_samples'
        }
    elif method == 'Conditional_Clinical_T2_dceDWI':
        return {
            'thresholds': {
                'Clinical': {'low': 0.3, 'high': 0.7},
                'T2': {'low': 0.4, 'high': 0.6}
            },
            'uncertain_weights': {'Clinical': 0.4, 'T2': 0.3, 'DCE': 0.15, 'DWI': 0.15},
            'condition': 'both_Clinical_and_T2_uncertain'
        }
    elif method == 'Conditional_Clinical_T2_LR':
        return {
            'thresholds': {
                'Clinical': {'low': 0.3, 'high': 0.7},
                'T2': {'low': 0.4, 'high': 0.6}
            },
            'fine_tuning': 'LogisticRegression_on_Clinical_T2_uncertain_samples',
            'parameter_tuning': 'inner_CV_on_uncertain_samples',
            'condition': 'both_Clinical_and_T2_uncertain'
        }
    return {}


# ==============================
# 1. CSV数据加载和对齐函数
# ==============================

def load_csv_predictions(csv_files_pattern):
    """
    从CSV文件加载预测结果
    csv_files_pattern: 可以是文件路径列表或通配符模式
    """
    print("🔍 从CSV文件加载预测结果...")

    if isinstance(csv_files_pattern, str):
        # 如果是通配符模式
        csv_files = glob(csv_files_pattern)
    elif isinstance(csv_files_pattern, list):
        # 如果是文件路径列表
        csv_files = csv_files_pattern
    else:
        raise ValueError("csv_files_pattern必须是字符串(通配符)或列表")

    print(f"找到 {len(csv_files)} 个CSV文件:")
    for f in csv_files:
        print(f"  - {os.path.basename(f)}")

    # 加载所有CSV文件
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # 根据文件名确定模态名称
            filename = os.path.basename(csv_file).lower()
            if 'clinical' in filename:
                modality = 'Clinical'
            elif 't2' in filename:
                modality = 'T2'
            elif 'dce' in filename:
                modality = 'DCE'
            elif 'dwi' in filename:
                modality = 'DWI'
            else:
                # 尝试从列名推断
                col_names_lower = [col.lower() for col in df.columns]
                if 'clinical' in ' '.join(col_names_lower):
                    modality = 'Clinical'
                elif 't2' in ' '.join(col_names_lower):
                    modality = 'T2'
                elif 'dce' in ' '.join(col_names_lower):
                    modality = 'DCE'
                elif 'dwi' in ' '.join(col_names_lower):
                    modality = 'DWI'
                else:
                    modality = os.path.splitext(os.path.basename(csv_file))[0]

            print(f"✅ 加载 {modality}: {len(df)} 行, {df.shape[1]} 列")
            dfs.append((modality, df))
        except Exception as e:
            print(f"❌ 加载 {csv_file} 失败: {e}")
            continue

    if len(dfs) < 2:
        raise ValueError(f"至少需要2个模态数据，只找到了{len(dfs)}个")

    return dfs


def align_csv_predictions(dfs):
    """
    对齐多个CSV文件的预测结果
    返回：对齐后的概率矩阵和标签
    """
    print("\n🔍 对齐CSV数据...")

    # 首先提取关键信息
    modality_data = {}

    for modality, df in dfs:
        # 查找包含预测概率的列
        proba_cols = [col for col in df.columns if 'prob' in col.lower() or 'proba' in col.lower()]
        true_cols = [col for col in df.columns if
                     'true' in col.lower() or 'label' in col.lower() or 'y_true' in col.lower()]

        if len(proba_cols) == 0:
            print(f"⚠️ {modality}: 未找到概率列，使用第一列作为预测概率")
            proba_col = df.columns[0]
        else:
            proba_col = proba_cols[0]

        if len(true_cols) == 0:
            print(f"⚠️ {modality}: 未找到真实标签列，使用第二列作为真实标签")
            true_col = df.columns[1] if len(df.columns) > 1 else None
        else:
            true_col = true_cols[0]

        # 提取数据
        y_proba = df[proba_col].values.astype(float)

        if true_col:
            y_true = df[true_col].values.astype(int)
        else:
            print(f"⚠️ {modality}: 未找到真实标签，尝试从其他列推断")
            # 查找可能包含标签的列
            for col in df.columns:
                if col != proba_col and df[col].nunique() <= 3:  # 类别数少可能是标签
                    y_true = df[col].values.astype(int)
                    print(f"    使用列 '{col}' 作为标签")
                    break
            else:
                # 如果没有找到，创建占位符标签（假设都是正类）
                print(f"⚠️ {modality}: 未找到标签列，创建占位符标签")
                y_true = np.ones(len(y_proba), dtype=int)

        # 检查数据范围
        if y_proba.min() < 0 or y_proba.max() > 1:
            print(f"⚠️ {modality}: 概率值不在[0,1]范围内，进行归一化")
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())

        modality_data[modality] = {
            'y_proba': y_proba,
            'y_true': y_true,
            'original_length': len(y_proba)
        }

        print(f"✅ {modality}: 提取 {len(y_proba)} 个样本，概率范围 [{y_proba.min():.3f}, {y_proba.max():.3f}]")

    # 检查样本数是否一致
    lengths = [data['original_length'] for data in modality_data.values()]
    min_length = min(lengths)
    max_length = max(lengths)

    if min_length != max_length:
        print(f"⚠️ 警告: 各模态样本数不一致 (最小={min_length}, 最大={max_length})")
        print("  使用最小样本数进行截断对齐")

        aligned_probas = []
        aligned_labels = []
        simplified_modality_names = []

        for name, data in modality_data.items():
            aligned_probas.append(data['y_proba'][:min_length])
            aligned_labels.append(data['y_true'][:min_length])
            simplified_modality_names.append(name)

        aligned_matrix = np.column_stack(aligned_probas)
        aligned_labels = np.array(aligned_labels)

        # 取第一个模态的标签作为基准
        y_true_aligned = aligned_labels[0]

        # 验证标签一致性
        for i, labels in enumerate(aligned_labels[1:], 1):
            mismatch = np.sum(y_true_aligned != labels)
            if mismatch > 0:
                print(f"⚠️ {simplified_modality_names[i]}: {mismatch}/{min_length} 个标签不一致")

        # 使用多数投票确定最终标签
        if mismatch > 0:
            print("  使用投票法确定最终标签...")
            y_true_voted = []
            for sample_idx in range(aligned_labels.shape[1]):
                labels = aligned_labels[:, sample_idx]
                count_0 = np.sum(labels == 0)
                count_1 = np.sum(labels == 1)
                y_true_voted.append(0 if count_0 >= count_1 else 1)

            y_true_aligned = np.array(y_true_voted)

        # 创建患者ID
        patient_ids = [f"sample_{i}" for i in range(min_length)]

        return aligned_matrix, y_true_aligned, simplified_modality_names, patient_ids
    else:
        # 样本数一致，直接对齐
        aligned_probas = []
        aligned_labels = []
        simplified_modality_names = []

        for name, data in modality_data.items():
            aligned_probas.append(data['y_proba'])
            aligned_labels.append(data['y_true'])
            simplified_modality_names.append(name)

        aligned_matrix = np.column_stack(aligned_probas)
        aligned_labels = np.array(aligned_labels)

        # 取第一个模态的标签作为基准
        y_true_aligned = aligned_labels[0]

        # 验证标签一致性
        label_mismatch_count = 0
        for i in range(1, aligned_labels.shape[0]):
            mismatch = np.sum(aligned_labels[i] != y_true_aligned)
            if mismatch > 0:
                label_mismatch_count += mismatch

        if label_mismatch_count > 0:
            print(f"⚠️ 总共有 {label_mismatch_count} 个标签不一致")
            print("  使用投票法确定最终标签...")

            y_true_voted = []
            for sample_idx in range(aligned_labels.shape[1]):
                labels = aligned_labels[:, sample_idx]
                count_0 = np.sum(labels == 0)
                count_1 = np.sum(labels == 1)
                y_true_voted.append(0 if count_0 >= count_1 else 1)

            y_true_aligned = np.array(y_true_voted)

        # 创建患者ID
        patient_ids = [f"sample_{i}" for i in range(len(y_true_aligned))]

        return aligned_matrix, y_true_aligned, simplified_modality_names, patient_ids


# ==============================
# 2. 嵌套交叉验证评估函数（完整定义）- 修改版本
# ==============================

def nested_cv_evaluation(X_meta, y_true, simplified_modality_names, n_outer_folds=5, n_inner_folds=5):
    """
    嵌套交叉验证评估所有融合方法（包含条件融合）
    """
    print(f"\n🔄 开始 {n_outer_folds}折嵌套交叉验证...")

    skf_outer = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=42)

    # 存储每折的预测结果
    n_samples = len(y_true)
    results = {
        'Clinical_only': np.zeros(n_samples),
        'T2_only': np.zeros(n_samples),
        'dce_only': np.zeros(n_samples),
        'DWI_only': np.zeros(n_samples),
        'AUC_weighted': np.zeros(n_samples),
        'Stacking_LR': np.zeros(n_samples),
        'Stacking_RF': np.zeros(n_samples),
        'Heuristic': np.zeros(n_samples),
        'Conditional_T2_dceDWI': np.zeros(n_samples),
        'Conditional_T2_LR': np.zeros(n_samples),
        'Conditional_Clinical_T2_dceDWI': np.zeros(n_samples),
        'Conditional_Clinical_T2_LR': np.zeros(n_samples)
    }

    fold_metrics = []
    fold_details = []

    # 确定模态数量和类型
    n_modalities = X_meta.shape[1]
    print(f"  检测到 {n_modalities} 个模态: {simplified_modality_names}")

    # 创建模态类型映射
    modality_type = {}
    for i, name in enumerate(simplified_modality_names):
        name_lower = name.lower()
        if 'clinical' in name_lower:
            modality_type[i] = 'Clinical'
        elif 't2' in name_lower:
            modality_type[i] = 'T2'
        elif 'dce' in name_lower:
            modality_type[i] = 'DCE'
        elif 'dwi' in name_lower:
            modality_type[i] = 'DWI'
        else:
            # 尝试从其他模式推断
            if i == 0:
                modality_type[i] = 'Clinical'
            elif i == 1:
                modality_type[i] = 'T2'
            elif i == 2:
                modality_type[i] = 'DCE'
            else:
                modality_type[i] = f'Modality_{i + 1}'

    print(f"  模态类型映射: {modality_type}")

    # 动态生成启发式权重
    heuristic_weights = np.ones(n_modalities) / n_modalities  # 默认等权重

    # 如果有明确的模态类型，使用预设权重
    if 'Clinical' in modality_type.values() and 'T2' in modality_type.values():
        # 根据实际存在的模态调整权重
        weights_dict = {'Clinical': 0.3, 'T2': 0.5, 'DCE': 0.1, 'DWI': 0.1}
        heuristic_weights = np.zeros(n_modalities)

        for i, mod_type in modality_type.items():
            if mod_type in weights_dict:
                heuristic_weights[i] = weights_dict[mod_type]
            else:
                heuristic_weights[i] = 0.1  # 默认权重

        # 归一化权重
        heuristic_weights = heuristic_weights / heuristic_weights.sum()

    print(f"  启发式权重: {heuristic_weights}")

    for fold, (train_idx, test_idx) in enumerate(skf_outer.split(X_meta, y_true), 1):
        print(f"\n📊 外层折 {fold}/{n_outer_folds}")
        print(f"  训练集: {len(train_idx)} 样本")
        print(f"  测试集: {len(test_idx)} 样本")

        # 分割数据
        X_train = X_meta[train_idx]
        y_train = y_true[train_idx]
        X_test = X_meta[test_idx]
        y_test = y_true[test_idx]

        # 确定模态索引
        clinical_idx = None
        t2_idx = None
        dce_idx = None
        dwi_idx = None

        for i, mod_type in modality_type.items():
            if mod_type == 'Clinical':
                clinical_idx = i
            elif mod_type == 'T2':
                t2_idx = i
            elif mod_type == 'DCE':
                dce_idx = i
            elif mod_type == 'DWI':
                dwi_idx = i

        # --- 首先计算单模态性能（用于基准）---
        for i, name in enumerate(simplified_modality_names):
            # 简化模态名称，移除括号和特殊字符
            simple_name = name.replace('predictions_rank01_', '').replace('(', '').replace(')', '').replace(' ', '_')
            modality_key = f'{simple_name}_only'

            # 确保键存在
            if modality_key not in results:
                results[modality_key] = np.zeros(n_samples)

            results[modality_key][test_idx] = X_test[:, i]

        # --- 方法1: AUC加权融合 ---
        # 在内层折上计算最优权重
        skf_inner = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
        best_weights = None
        best_inner_auc = 0

        for inner_train_idx, inner_val_idx in skf_inner.split(X_train, y_train):
            X_inner_train = X_train[inner_train_idx]
            y_inner_train = y_train[inner_train_idx]

            X_inner_val = X_train[inner_val_idx]
            y_inner_val = y_train[inner_val_idx]

            # 计算各模态在内层训练集上的AUC
            modality_aucs = []
            for i in range(X_inner_train.shape[1]):
                auc_val = roc_auc_score(y_inner_train, X_inner_train[:, i])
                modality_aucs.append(auc_val)

            # 计算权重
            total_auc = sum(modality_aucs)
            if total_auc > 0:
                weights = [auc / total_auc for auc in modality_aucs]
            else:
                weights = [1 / X_inner_train.shape[1]] * X_inner_train.shape[1]

            # 在内层验证集评估
            y_val_weighted = np.average(X_inner_val, axis=1, weights=weights)
            val_auc = roc_auc_score(y_inner_val, y_val_weighted)

            if val_auc > best_inner_auc:
                best_inner_auc = val_auc
                best_weights = weights

        # 在测试集应用最佳权重
        if best_weights is None:
            best_weights = [1 / X_train.shape[1]] * X_train.shape[1]

        y_test_weighted = np.average(X_test, axis=1, weights=best_weights)
        results['AUC_weighted'][test_idx] = y_test_weighted

        # --- 方法2: Stacking with Logistic Regression ---
        # 内层CV调参
        best_lr = None
        best_lr_auc = 0

        for C_param in [0.001, 0.01, 0.1, 1.0, 10.0]:
            lr = LogisticRegression(
                C=C_param,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            )

            # 内层CV评估
            inner_aucs = []
            for inner_train_idx, inner_val_idx in skf_inner.split(X_train, y_train):
                X_inner_train = X_train[inner_train_idx]
                y_inner_train = y_train[inner_train_idx]

                X_inner_val = X_train[inner_val_idx]
                y_inner_val = y_train[inner_val_idx]

                lr.fit(X_inner_train, y_inner_train)
                y_val_pred = lr.predict_proba(X_inner_val)[:, 1]
                inner_auc = roc_auc_score(y_inner_val, y_val_pred)
                inner_aucs.append(inner_auc)

            mean_inner_auc = np.mean(inner_aucs)
            if mean_inner_auc > best_lr_auc:
                best_lr_auc = mean_inner_auc
                best_lr = LogisticRegression(
                    C=C_param,
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42,
                    solver='liblinear'
                )

        # 用最佳参数在整个训练集训练
        if best_lr is None:
            best_lr = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            )

        best_lr.fit(X_train, y_train)
        y_test_stacking_lr = best_lr.predict_proba(X_test)[:, 1]
        results['Stacking_LR'][test_idx] = y_test_stacking_lr

        # --- 方法3: Stacking with Random Forest ---
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_test_stacking_rf = rf.predict_proba(X_test)[:, 1]
        results['Stacking_RF'][test_idx] = y_test_stacking_rf

        # --- 方法4: 启发式加权 ---
        y_test_heuristic = np.average(X_test, axis=1, weights=heuristic_weights)
        results['Heuristic'][test_idx] = y_test_heuristic

        # --- 条件融合方法（根据实际模态情况调整）---
        print("  🔍 计算条件融合方法...")

        # 初始化条件融合结果
        y_test_conditional1 = np.zeros(len(y_test))
        y_test_conditional2 = np.zeros(len(y_test))

        # 条件融合1: T2 + DCE/DWI
        if t2_idx is not None:
            t2_proba_test = X_test[:, t2_idx]
            y_test_conditional1 = t2_proba_test.copy()  # 默认为T2结果

            # 如果还有其他模态
            if dce_idx is not None or dwi_idx is not None:
                # 定义不确定区域
                t2_uncertain_mask = (t2_proba_test >= 0.4) & (t2_proba_test <= 0.6)

                if t2_uncertain_mask.sum() > 0:
                    # 收集可用的模态
                    available_modalities = []
                    available_weights = []

                    if t2_idx is not None:
                        available_modalities.append(t2_proba_test)
                        available_weights.append(0.6)

                    if dce_idx is not None:
                        available_modalities.append(X_test[:, dce_idx])
                        available_weights.append(0.2)

                    if dwi_idx is not None:
                        available_modalities.append(X_test[:, dwi_idx])
                        available_weights.append(0.2)

                    # 归一化权重
                    total_weight = sum(available_weights)
                    normalized_weights = [w / total_weight for w in available_weights]

                    # 计算加权平均值
                    for i in range(len(available_modalities)):
                        y_test_conditional1[t2_uncertain_mask] += normalized_weights[i] * available_modalities[i][
                            t2_uncertain_mask]

        results['Conditional_T2_dceDWI'][test_idx] = y_test_conditional1

        # 条件融合2: T2 + LR微调
        if t2_idx is not None:
            t2_proba_train = X_train[:, t2_idx]
            t2_proba_test = X_test[:, t2_idx]

            y_test_conditional2 = t2_proba_test.copy()  # 默认为T2结果

            # 在训练集上定义不确定区域
            t2_uncertain_mask_train = (t2_proba_train >= 0.4) & (t2_proba_train <= 0.6)

            # 如果训练集中有不确定样本，训练微调LR
            if t2_uncertain_mask_train.sum() > 5:
                X_uncertain_train = X_train[t2_uncertain_mask_train]
                y_uncertain_train = y_train[t2_uncertain_mask_train]

                # 内层CV优化微调LR的参数
                best_lr_tune = None
                best_tune_auc = 0

                for C_param in [0.01, 0.1, 1.0, 10.0]:
                    lr_tune = LogisticRegression(
                        C=C_param,
                        class_weight='balanced',
                        max_iter=1000,
                        random_state=42,
                        solver='liblinear'
                    )

                    # 简单验证
                    if len(y_uncertain_train) >= 10:
                        try:
                            skf_tune = StratifiedKFold(n_splits=min(3, len(y_uncertain_train) // 3),
                                                       shuffle=True, random_state=42)
                            tune_aucs = []

                            for tune_train_idx, tune_val_idx in skf_tune.split(X_uncertain_train, y_uncertain_train):
                                X_tune_train = X_uncertain_train[tune_train_idx]
                                y_tune_train = y_uncertain_train[tune_train_idx]
                                X_tune_val = X_uncertain_train[tune_val_idx]
                                y_tune_val = y_uncertain_train[tune_val_idx]

                                lr_tune.fit(X_tune_train, y_tune_train)
                                y_tune_pred = lr_tune.predict_proba(X_tune_val)[:, 1]
                                auc_val = roc_auc_score(y_tune_val, y_tune_pred)
                                tune_aucs.append(auc_val)

                            mean_tune_auc = np.mean(tune_aucs)
                            if mean_tune_auc > best_tune_auc:
                                best_tune_auc = mean_tune_auc
                                best_lr_tune = lr_tune
                        except:
                            continue

                if best_lr_tune is not None:
                    # 找出测试集中的不确定样本
                    t2_uncertain_mask_test = (t2_proba_test >= 0.4) & (t2_proba_test <= 0.6)

                    if t2_uncertain_mask_test.sum() > 0:
                        X_uncertain_test = X_test[t2_uncertain_mask_test]
                        refined_proba = best_lr_tune.predict_proba(X_uncertain_test)[:, 1]
                        y_test_conditional2[t2_uncertain_mask_test] = refined_proba

            results['Conditional_T2_LR'][test_idx] = y_test_conditional2

        # Clinical+T2条件融合（根据实际模态情况）
        if clinical_idx is not None and t2_idx is not None:
            clinical_proba_test = X_test[:, clinical_idx]
            t2_proba_test = X_test[:, t2_idx]

            # KDC (Knowledge-Driven Conditional) 融合
            # 输入: 4个单模态概率值 (p1=Clinical, p2=T2, p3=DCE, p4=DWI)
            # 条件: p1∈[0.3,0.7] AND p2∈[0.4,0.6]
            # - 不满足条件(certain): 简单平均 (p1+p2+p3+p4)/4
            # - 满足条件(uncertain): Stacking LR

            # 默认: 简单平均所有4个模态
            y_test_conditional3 = (clinical_proba_test + t2_proba_test + dce_proba_test + dwi_proba_test) / 4

            # 定义不确定区域
            clinical_uncertain_mask = (clinical_proba_test >= 0.3) & (clinical_proba_test <= 0.7)
            t2_uncertain_mask = (t2_proba_test >= 0.4) & (t2_proba_test <= 0.6)
            uncertain_mask = clinical_uncertain_mask & t2_uncertain_mask

            # 在不确定区域内使用Stacking LR
            if uncertain_mask.sum() > 0:
                # 在训练集的不确定区域上训练LR模型
                clinical_proba_train = X_train[:, clinical_idx]
                t2_proba_train = X_train[:, t2_idx]
                dce_proba_train = X_train[:, dce_idx]
                dwi_proba_train = X_train[:, dwi_idx]

                clinical_uncertain_mask_train = (clinical_proba_train >= 0.3) & (clinical_proba_train <= 0.7)
                t2_uncertain_mask_train = (t2_proba_train >= 0.4) & (t2_proba_train <= 0.6)
                uncertain_mask_train = clinical_uncertain_mask_train & t2_uncertain_mask_train

                if uncertain_mask_train.sum() > 5:
                    # 使用4个模态的概率值作为特征
                    X_uncertain_train = np.column_stack([
                        clinical_proba_train[uncertain_mask_train],
                        t2_proba_train[uncertain_mask_train],
                        dce_proba_train[uncertain_mask_train],
                        dwi_proba_train[uncertain_mask_train]
                    ])
                    y_uncertain_train = y_train[uncertain_mask_train]

                    # 训练Stacking LR
                    lr_stack = LogisticRegression(
                        C=1.0,
                        class_weight='balanced',
                        max_iter=1000,
                        random_state=42,
                        solver='liblinear'
                    )

                    try:
                        lr_stack.fit(X_uncertain_train, y_uncertain_train)

                        # 在测试集的不确定区域上预测
                        X_uncertain_test = np.column_stack([
                            clinical_proba_test[uncertain_mask],
                            t2_proba_test[uncertain_mask],
                            dce_proba_test[uncertain_mask],
                            dwi_proba_test[uncertain_mask]
                        ])
                        refined_proba = lr_stack.predict_proba(X_uncertain_test)[:, 1]
                        y_test_conditional3[uncertain_mask] = refined_proba
                    except:
                        pass

            results['Conditional_Clinical_T2_dceDWI'][test_idx] = y_test_conditional3

            # 条件融合4: Clinical+T2+LR
            y_test_conditional4 = (clinical_proba_test + t2_proba_test) / 2

            if uncertain_mask.sum() > 0:
                # 在训练集上训练微调模型
                clinical_proba_train = X_train[:, clinical_idx]
                t2_proba_train = X_train[:, t2_idx]

                clinical_uncertain_mask_train = (clinical_proba_train >= 0.3) & (clinical_proba_train <= 0.7)
                t2_uncertain_mask_train = (t2_proba_train >= 0.4) & (t2_proba_train <= 0.6)
                uncertain_mask_train = clinical_uncertain_mask_train & t2_uncertain_mask_train

                if uncertain_mask_train.sum() > 5:
                    X_uncertain_train = X_train[uncertain_mask_train]
                    y_uncertain_train = y_train[uncertain_mask_train]

                    # 简单LR微调
                    lr_tune = LogisticRegression(
                        C=1.0,
                        class_weight='balanced',
                        max_iter=1000,
                        random_state=42,
                        solver='liblinear'
                    )

                    try:
                        lr_tune.fit(X_uncertain_train, y_uncertain_train)
                        X_uncertain_test = X_test[uncertain_mask]
                        refined_proba = lr_tune.predict_proba(X_uncertain_test)[:, 1]
                        y_test_conditional4[uncertain_mask] = refined_proba
                    except:
                        pass

            results['Conditional_Clinical_T2_LR'][test_idx] = y_test_conditional4

        # --- 计算本折的性能指标 ---
        fold_metric = {'fold': fold}

        # 计算各方法的AUC
        for method_name, probas in results.items():
            if len(probas[test_idx]) > 0 and not np.all(probas[test_idx] == 0):
                try:
                    auc_val = roc_auc_score(y_test, probas[test_idx])
                    fold_metric[method_name] = auc_val
                except:
                    fold_metric[method_name] = 0

        fold_metrics.append(fold_metric)

        # 记录本折详细信息
        fold_detail = {
            'fold': fold,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_class_0': np.sum(y_train == 0),
            'train_class_1': np.sum(y_train == 1),
            'test_class_0': np.sum(y_test == 0),
            'test_class_1': np.sum(y_test == 1),
            'best_weights': best_weights,
            'n_modalities': n_modalities,
            'clinical_idx': clinical_idx,
            't2_idx': t2_idx,
            'dce_idx': dce_idx,
            'dwi_idx': dwi_idx
        }
        fold_details.append(fold_detail)

    # 计算最终性能指标（在整个数据集上）
    print("\n📊 计算最终性能指标...")
    final_metrics = {}

    for method_name, probas in results.items():
        if len(probas) > 0 and not np.all(probas == 0):
            try:
                auc_val = roc_auc_score(y_true, probas)
                y_pred = (probas > 0.5).astype(int)
                bal_acc = balanced_accuracy_score(y_true, y_pred)
                kappa = cohen_kappa_score(y_true, y_pred)

                final_metrics[method_name] = {
                    'AUC': auc_val,
                    'Balanced_Accuracy': bal_acc,
                    'Kappa': kappa
                }
                print(f"  {method_name:25s}: AUC={auc_val:.3f}, BalAcc={bal_acc:.3f}, Kappa={kappa:.3f}")
            except Exception as e:
                print(f"  ⚠️ 计算{method_name}指标时出错: {e}")
                final_metrics[method_name] = {
                    'AUC': 0,
                    'Balanced_Accuracy': 0,
                    'Kappa': 0
                }

    return results, final_metrics, fold_metrics, fold_details


# ==============================
# 3. 绘制ROC曲线函数（分别绘制每种融合方法）
# ==============================

def plot_roc_curves_separately(X_meta, y_true, results, final_metrics, simplified_modality_names,
                               best_method, output_dir):
    """
    为每种融合方法分别绘制与单模态对比的ROC曲线（包含条件融合）
    """
    print("\n📈 开始绘制每种融合方法的单独ROC曲线...")

    # 线型方案
    line_styles = {
        'single_modality': '--',  # 单模态用虚线
        'fusion_method': '-',  # 融合方法用实线
        'conditional_method': '-.',  # 条件融合用点划线
        'random': ':'  # 随机线用点线
    }

    # 定义要绘制的融合方法 - 根据实际存在的方法名称
    fusion_methods = []

    # 只绘制实际存在的方法
    for method in ['AUC_weighted', 'Stacking_LR', 'Stacking_RF', 'Heuristic',
                   'Conditional_T2_dceDWI', 'Conditional_T2_LR',
                   'Conditional_Clinical_T2_dceDWI', 'Conditional_Clinical_T2_LR']:
        if method in results and method in final_metrics:
            fusion_methods.append(method)

    print(f"  发现 {len(fusion_methods)} 个有效融合方法: {fusion_methods}")

    # 方法描述映射
    title_map = {
        'AUC_weighted': 'AUC-weighted Fusion vs Single Modalities',
        'Stacking_LR': 'Stacking (Logistic Regression) vs Single Modalities',
        'Stacking_RF': 'Stacking (Random Forest) vs Single Modalities',
        'Heuristic': 'Heuristic-weighted Fusion vs Single Modalities',
        'Conditional_T2_dceDWI': 'Conditional Fusion (T2 + DCE/DWI) vs Single Modalities',
        'Conditional_T2_LR': 'Conditional Fusion (T2 + LR Tuning) vs Single Modalities',
        'Conditional_Clinical_T2_dceDWI': 'Conditional Fusion (Clinical+T2+DCE/DWI) vs Single Modalities',
        'Conditional_Clinical_T2_LR': 'Conditional Fusion (Clinical+T2+LR) vs Single Modalities'
    }

    # 为每种融合方法单独绘图
    for fusion_method in fusion_methods:
        print(f"  正在绘制 {fusion_method} 的ROC曲线...")

        # 创建新的图形
        fig, ax = plt.subplots(figsize=(10, 8))

        # 1. 绘制单模态基准线
        for i, name in enumerate(simplified_modality_names):
            modality_key = f'{name}_only'
            if modality_key in results and modality_key in final_metrics:
                fpr, tpr, _ = roc_curve(y_true, results[modality_key])
                auc_val = final_metrics[modality_key]['AUC']
                ax.plot(fpr, tpr, color=colors.get(modality_key, '#888888'),
                        lw=2, linestyle=line_styles['single_modality'], alpha=0.8,
                        label=f'{name} only (AUC={auc_val:.3f})')

        # 2. 绘制当前融合方法
        if fusion_method in results and fusion_method in final_metrics:
            fpr_fusion, tpr_fusion, _ = roc_curve(y_true, results[fusion_method])
            auc_fusion = final_metrics[fusion_method]['AUC']

            # 根据方法类型选择线型
            if 'Conditional' in fusion_method:
                linestyle = line_styles['conditional_method']
            else:
                linestyle = line_styles['fusion_method']

            ax.plot(fpr_fusion, tpr_fusion, color=colors.get(fusion_method, '#9467BD'),
                    lw=3, linestyle=linestyle,
                    label=f'{fusion_method} (AUC={auc_fusion:.3f})')

        # 3. 绘制随机线
        ax.plot([0, 1], [0, 1], 'k-', lw=1.5, alpha=0.5,
                linestyle=line_styles['random'], label='Random (AUC=0.500)')

        # 设置图形属性
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)

        # 设置标题
        title = title_map.get(fusion_method, f'{fusion_method} vs Single Modalities')
        ax.set_title(title, fontsize=14, pad=15)

        # 添加图例
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        # 添加数据集信息文本框
        stats_text = f"""Dataset Summary:
────────────────
Total: {len(y_true)}
Class 0: {np.sum(y_true == 0)}
Class 1: {np.sum(y_true == 1)}

Fusion Method: {fusion_method}
AUC: {auc_fusion:.3f}

Validation: 5×5 Nested CV"""

        # 对于条件融合，添加额外信息
        if 'Conditional' in fusion_method:
            # 获取T2概率（假设T2是第二个模态）
            t2_idx = 1  # Clinical=0, T2=1, DCE=2, DWI=3
            t2_proba = X_meta[:, t2_idx]

            # 不同的条件融合使用不同的阈值
            if fusion_method == 'Conditional_T2_dceDWI' or fusion_method == 'Conditional_T2_LR':
                # T2条件融合：只看T2不确定区间
                uncertain_samples = ((t2_proba >= 0.4) & (t2_proba <= 0.6)).sum()
                uncertain_ratio = uncertain_samples / len(y_true) * 100
                threshold_info = "Threshold: [0.4, 0.6] (T2 only)"
            else:
                # Clinical+T2条件融合：看两者都不确定
                clinical_idx = 0
                clinical_proba = X_meta[:, clinical_idx]
                clinical_uncertain = (clinical_proba >= 0.3) & (clinical_proba <= 0.7)
                t2_uncertain = (t2_proba >= 0.4) & (t2_proba <= 0.6)
                uncertain_samples = (clinical_uncertain & t2_uncertain).sum()
                uncertain_ratio = uncertain_samples / len(y_true) * 100
                threshold_info = "Thresholds: Clinical[0.3, 0.7], T2[0.4, 0.6]"

            stats_text += f"""

Conditional Fusion Info:
Uncertain Samples: {uncertain_samples} ({uncertain_ratio:.1f}%)
{threshold_info}"""

        # 添加文本框到左上角
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='left',
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="white",
                          edgecolor="#4682B4",
                          linewidth=1.2,
                          alpha=0.9),
                zorder=10)

        # 调整布局
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存图形
        roc_path = os.path.join(output_dir, f"ROC_{fusion_method}_vs_single_modalities.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✅ 保存至: {roc_path}")

    # 绘制所有方法在一起的综合图（可选）
    if fusion_methods:
        print("  正在绘制所有方法的综合ROC曲线...")
        fig_all, ax_all = plt.subplots(figsize=(12, 10))

        # 绘制所有单模态
        for i, name in enumerate(simplified_modality_names):
            modality_key = f'{name}_only'
            if modality_key in results and modality_key in final_metrics:
                fpr, tpr, _ = roc_curve(y_true, results[modality_key])
                auc_val = final_metrics[modality_key]['AUC']
                ax_all.plot(fpr, tpr, color=colors.get(modality_key, '#888888'),
                            lw=2, linestyle='--', alpha=0.7,
                            label=f'{name} only (AUC={auc_val:.3f})')

        # 绘制所有融合方法
        for fusion_method in fusion_methods:
            if fusion_method in results and fusion_method in final_metrics:
                fpr_fusion, tpr_fusion, _ = roc_curve(y_true, results[fusion_method])
                auc_fusion = final_metrics[fusion_method]['AUC']

                # 根据方法类型选择线型
                if 'Conditional' in fusion_method:
                    linestyle = '-.'
                else:
                    linestyle = '-'

                ax_all.plot(fpr_fusion, tpr_fusion, color=colors.get(fusion_method, '#9467BD'),
                            lw=2.5, linestyle=linestyle,
                            label=f'{fusion_method} (AUC={auc_fusion:.3f})')

        # 绘制随机线
        ax_all.plot([0, 1], [0, 1], 'k-', lw=1.5, alpha=0.5, label='Random (AUC=0.500)')

        ax_all.set_xlim([0.0, 1.0])
        ax_all.set_ylim([0.0, 1.05])
        ax_all.set_xlabel('False Positive Rate', fontsize=14)
        ax_all.set_ylabel('True Positive Rate', fontsize=14)
        ax_all.set_title('ROC Curves: All Fusion Methods vs Single Modalities', fontsize=16, pad=20)

        # 简化图例
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, linestyle='--', label='Single Modality'),
            Line2D([0], [0], color='gray', lw=2.5, linestyle='-', label='Fusion Method'),
            Line2D([0], [0], color='gray', lw=2.5, linestyle='-.', label='Conditional Fusion'),
            Line2D([0], [0], color='black', lw=1.5, linestyle=':', alpha=0.5, label='Random')
        ]

        # 添加图例
        ax_all.legend(handles=legend_elements, loc='lower right', fontsize=10)

        # 添加详细的图例说明
        legend_text = "Single Modalities:\n"
        for i, name in enumerate(simplified_modality_names):
            modality_key = f'{name}_only'
            if modality_key in final_metrics:
                legend_text += f"  • {name}: AUC={final_metrics[modality_key]['AUC']:.3f}\n"

        legend_text += "\nFusion Methods:\n"
        for fusion_method in fusion_methods:
            if fusion_method in final_metrics:
                marker = '★' if fusion_method == best_method[0] else '•'
                legend_text += f"  {marker} {fusion_method}: AUC={final_metrics[fusion_method]['AUC']:.3f}\n"

        # 添加文本框
        ax_all.text(0.02, 0.98, legend_text, transform=ax_all.transAxes,
                    fontsize=8.5,
                    verticalalignment='top',
                    horizontalalignment='left',
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor="white",
                              edgecolor="#4682B4",
                              linewidth=1.2,
                              alpha=0.9))

        ax_all.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # 保存综合图
        roc_all_path = os.path.join(output_dir, "ROC_All_Methods_Comparison.png")
        plt.savefig(roc_all_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✅ 综合ROC图保存至: {roc_all_path}")

    return len(fusion_methods) + 1  # 返回绘制的图形数量（单独图 + 综合图）

# ==============================
# 4. 绘制混淆矩阵热力图函数
# ==============================
def plot_confusion_matrices(X_meta, y_true, results, final_metrics, simplified_modality_names, output_dir):
    """
    为所有方法（单模态 + 融合方法）绘制混淆矩阵热力图
    """
    print("\n📊 开始绘制混淆矩阵热力图...")

    # 获取所有要绘制的方法
    all_methods = []
    # 单模态
    for name in simplified_modality_names:
        modality_key = f'{name}_only'
        if modality_key in results:
            all_methods.append(modality_key)
    # 融合方法
    fusion_methods = [m for m in results.keys() if not m.endswith('_only')]
    all_methods.extend(fusion_methods)

    n_methods = len(all_methods)
    if n_methods == 0:
        print("⚠️ 没有找到可绘制混淆矩阵的方法")
        return 0

    # 动态设置子图网格（最多每行4个）
    cols = min(4, n_methods)
    rows = (n_methods + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    if n_methods == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for idx, method in enumerate(all_methods):
        ax = axes[idx]
        y_pred = (results[method] > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'], ax=ax)
        auc_val = final_metrics.get(method, {}).get('AUC', 0)
        ax.set_title(f"{method}\n(AUC={auc_val:.3f})", fontsize=12)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    # 隐藏多余的子图
    for idx in range(n_methods, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout(pad=3.0)
    cm_path = os.path.join(output_dir, "Confusion_Matrices_All_Methods.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵热力图已保存至: {cm_path}")
    return 1


# ==============================
# 5. 主程序
# ==============================

def main():
    print("=" * 60)
    print("🎯 改进版多模态后融合分析 - CSV版本")
    print("=" * 60)

    # ==================== 修改这里：指定CSV文件路径 ====================
    # 方法1: 使用文件路径列表
    csv_files = [
        r"K:\PCa_2026\Article\放射组学\图表\roc\内部测试\单模态-ROC\csv\clinical.csv",  # 临床特征预测结果
        r"K:\PCa_2026\Article\放射组学\图表\roc\内部测试\单模态-ROC\csv\t2.csv",  # T2预测结果
        r"K:\PCa_2026\Article\放射组学\图表\roc\内部测试\单模态-ROC\csv\DCE.csv",  # dce预测结果
        r"K:\PCa_2026\Article\放射组学\图表\roc\内部测试\单模态-ROC\csv\dwi.csv"  # DWI预测结果
    ]

    # 或者使用方法2: 使用通配符模式
    # csv_files_pattern = r"path\to\*\predictions.csv"

    # 注意: CSV文件应该至少包含以下列:
    # 1. 预测概率列 (列名包含"prob"或"proba")
    # 2. 真实标签列 (列名包含"true"或"label"或"y_true")

    # 1. 加载CSV文件
    dfs = load_csv_predictions(csv_files)

    # 2. 对齐数据
    X_meta, y_true, simplified_modality_names, patient_ids = align_csv_predictions(dfs)

    print(f"\n✅ 数据对齐完成!")
    print(f"   总样本数: {len(y_true)}")
    print(f"   模态数: {len(simplified_modality_names)}")
    print(f"   标签分布: 0={np.sum(y_true == 0)}, 1={np.sum(y_true == 1)}")
    print(f"   模态顺序: {simplified_modality_names}")

    # 3. 嵌套交叉验证评估
    results, final_metrics, fold_metrics, fold_details = nested_cv_evaluation(
        X_meta, y_true, simplified_modality_names, n_outer_folds=5, n_inner_folds=5
    )

    # 4. 显示结果
    print("\n" + "=" * 60)
    print("📊 嵌套交叉验证结果汇总")
    print("=" * 60)

    # 各折AUC平均值
    df_fold = pd.DataFrame(fold_metrics)

    # 获取所有方法名（包括单模态）
    all_methods = list(results.keys())
    single_modalities = [m for m in all_methods if m.endswith('_only')]
    fusion_methods = [m for m in all_methods if not m.endswith('_only')]

    print("\n📋 单模态性能:")
    for method in single_modalities:
        if method in df_fold.columns:
            mean_auc = df_fold[method].mean()
            std_auc = df_fold[method].std()
            print(f"  {method:15s}: {mean_auc:.3f} ± {std_auc:.3f}")

    print("\n📋 融合方法性能:")
    for method in fusion_methods:
        if method in df_fold.columns:
            mean_auc = df_fold[method].mean()
            std_auc = df_fold[method].std()
            print(f"  {method:15s}: {mean_auc:.3f} ± {std_auc:.3f}")

    # 选择最佳方法
    print("\n🔍 选择最佳融合方法...")
    fusion_final_metrics = {k: v for k, v in final_metrics.items() if not k.endswith('_only')}
    best_method = max(fusion_final_metrics.items(), key=lambda x: x[1]['AUC'])
    print(f"✅ 最佳融合方法: {best_method[0]} (AUC={best_method[1]['AUC']:.3f})")

    print("\n🔍 选择最佳单模态方法...")
    single_final_metrics = {k: v for k, v in final_metrics.items() if k.endswith('_only')}
    best_single = max(single_final_metrics.items(), key=lambda x: x[1]['AUC'])
    print(f"✅ 最佳单模态: {best_single[0]} (AUC={best_single[1]['AUC']:.3f})")

    # 5. 为每个融合方法创建单独的文件夹并保存相关文件
    print("\n📁 为每个融合方法创建单独的文件夹...")

    # 获取T2概率用于条件融合分析（假设T2是第二个模态）
    t2_proba = X_meta[:, 1]  # Clinical=0, T2=1, DCE=2, DWI=3
    uncertain_mask = (t2_proba >= 0.4) & (t2_proba <= 0.6)

    # 预计算各模态AUC（避免重复计算）
    modality_aucs_full = []
    for i in range(X_meta.shape[1]):
        auc_val = roc_auc_score(y_true, X_meta[:, i])
        modality_aucs_full.append(auc_val)
    total_auc_full = sum(modality_aucs_full)

    # 定义方法特定的处理函数
    def save_model_info(method_dir, method, model_info):
        """保存模型信息文件"""
        model_info_path = os.path.join(method_dir, f"{method}_model_info.json")
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=4, ensure_ascii=False)
        print(f"    保存模型信息: {model_info_path}")

    # 主循环 - 处理每个融合方法
    for method in fusion_methods:
        if method not in results:
            continue

        print(f"\n📁 处理融合方法: {method}")

        # 创建方法专属文件夹
        method_dir = os.path.join(OUTPUT_DIR, method)
        os.makedirs(method_dir, exist_ok=True)
        print(f"  ✅ 创建文件夹: {method_dir}")

        # ==================== 1. 保存预测结果 ====================
        # 基础预测结果
        # 保存预测结果时也使用简化名称
        method_results_df = pd.DataFrame({
            'patient_id': patient_ids,
            'y_true': y_true,
            f'{method}_proba': results[method],
            f'{method}_pred': (results[method] > 0.5).astype(int)
        })

        # 添加单模态预测结果作为参考
        # 使用原始模态名称，不要重新赋值
        for i, name in enumerate(simplified_modality_names):
            # 简化模态名称
            simple_name = name.replace('predictions_rank01_', '').replace('(', '').replace(')', '').replace(' ', '_')
            modality_key = f'{simple_name}_only'

            # 如果简化键存在，使用它
            if modality_key in results:
                method_results_df[f'{simple_name}_proba'] = X_meta[:, i]
            else:
                # 否则使用原始名称
                original_key = f'{name}_only'
                if original_key in results:
                    method_results_df[f'{name}_proba'] = X_meta[:, i]
                else:
                    # 如果都不存在，直接添加
                    method_results_df[f'{name}_proba'] = X_meta[:, i]

        # 保存预测结果
        results_path = os.path.join(method_dir, f"{method}_predictions.csv")
        method_results_df.to_csv(results_path, index=False)
        print(f"    💾 保存预测结果: {results_path}")

        # ==================== 2. 方法特定的额外信息 ====================
        # 条件融合的额外信息
        if 'Conditional' in method:
            # 不确定样本分析
            uncertain_info_df = method_results_df.copy()
            uncertain_info_df['T2_is_uncertain'] = uncertain_mask.astype(int)
            uncertain_info_df['T2_proba_interval'] = pd.cut(t2_proba,
                                                            bins=[0, 0.4, 0.6, 1.0],
                                                            labels=['Low', 'Uncertain', 'High'])

            # 按不确定性分组统计
            uncertain_stats = uncertain_info_df.groupby('T2_proba_interval').agg({
                'y_true': ['count', 'mean'],
                f'{method}_proba': ['mean', 'std'],
                f'{method}_pred': 'mean'
            }).round(3)

            # 保存不确定样本详细信息
            uncertain_path = os.path.join(method_dir, f"{method}_uncertain_samples.csv")
            uncertain_info_df.to_csv(uncertain_path, index=False)

            # 保存统计摘要
            stats_path = os.path.join(method_dir, f"{method}_uncertain_stats.csv")
            uncertain_stats.to_csv(stats_path)

            print(f"    📈 不确定样本分析:")
            print(f"      不确定区间样本: {uncertain_mask.sum()} ({uncertain_mask.sum() / len(y_true) * 100:.1f}%)")
            print(f"      保存详细数据: {uncertain_path}")
            print(f"      保存统计摘要: {stats_path}")

            # Conditional_T2_LR 的微调模型保存
            if method == 'Conditional_T2_LR' and uncertain_mask.sum() > 10:
                try:
                    X_uncertain = X_meta[uncertain_mask]
                    y_uncertain = y_true[uncertain_mask]

                    # 使用内层CV优化参数
                    best_lr_tune = None
                    best_auc = 0

                    for C_param in [0.01, 0.1, 1.0, 10.0]:
                        lr_tune = LogisticRegression(
                            C=C_param,
                            class_weight='balanced',
                            max_iter=1000,
                            random_state=42,
                            solver='liblinear'
                        )

                        # 简单交叉验证
                        cv_scores = []
                        skf = StratifiedKFold(n_splits=min(5, len(y_uncertain) // 2), shuffle=True, random_state=42)
                        for train_idx, val_idx in skf.split(X_uncertain, y_uncertain):
                            X_train_tune = X_uncertain[train_idx]
                            y_train_tune = y_uncertain[train_idx]
                            X_val_tune = X_uncertain[val_idx]
                            y_val_tune = y_uncertain[val_idx]

                            lr_tune.fit(X_train_tune, y_train_tune)
                            y_pred_tune = lr_tune.predict_proba(X_val_tune)[:, 1]
                            auc_val = roc_auc_score(y_val_tune, y_pred_tune)
                            cv_scores.append(auc_val)

                        mean_auc = np.mean(cv_scores)
                        if mean_auc > best_auc:
                            best_auc = mean_auc
                            best_lr_tune = lr_tune

                    if best_lr_tune:
                        # 在整个不确定数据集上训练最终模型
                        best_lr_tune.fit(X_uncertain, y_uncertain)
                        tune_model_path = os.path.join(method_dir, f"{method}_tuning_model.pkl")
                        joblib.dump(best_lr_tune, tune_model_path)

                        # 保存微调模型权重
                        if hasattr(best_lr_tune, 'coef_'):
                            tune_weights = pd.DataFrame({
                                'Feature': ['Intercept'] + simplified_modality_names,
                                'Coefficient': [best_lr_tune.intercept_[0]] + best_lr_tune.coef_[0].tolist()
                            })
                            tune_weights_path = os.path.join(method_dir, f"{method}_tuning_weights.csv")
                            tune_weights.to_csv(tune_weights_path, index=False)
                            print(f"    💾 保存微调模型权重: {tune_weights_path}")

                        print(f"    💾 保存微调模型: {tune_model_path}")

                except Exception as e:
                    print(f"    ⚠️ 微调模型保存失败: {e}")

        # ==================== 3. 模型保存（只针对有模型的方法） ====================
        if method in ['Stacking_LR', 'Stacking_RF']:
            # 使用最佳参数训练最终模型
            if method == 'Stacking_LR':
                # 从折叠详情中获取最佳C参数
                best_C = 0.1  # 默认值
                for detail in fold_details:
                    if detail.get('best_C'):
                        best_C = detail['best_C']
                        break

                model = LogisticRegression(
                    C=best_C,
                    class_weight='balanced',
                    max_iter=1000,
                    random_state=42,
                    solver='liblinear'
                )
            elif method == 'Stacking_RF':
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=5,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1
                )

            # 训练并保存模型
            model.fit(X_meta, y_true)
            model_path = os.path.join(method_dir, f"{method}_model.pkl")
            joblib.dump(model, model_path)
            print(f"    💾 保存模型: {model_path}")

            # 保存模型参数/权重
            if hasattr(model, 'coef_'):  # Logistic Regression
                weights_df = pd.DataFrame({
                    'Modality': simplified_modality_names,
                    'Weight': model.coef_[0],
                    'Absolute_Weight': np.abs(model.coef_[0])
                }).sort_values('Absolute_Weight', ascending=False)

                weights_path = os.path.join(method_dir, f"{method}_weights.csv")
                weights_df.to_csv(weights_path, index=False)
                print(f"    📋 保存模型权重: {weights_path}")

            elif hasattr(model, 'feature_importances_'):  # Random Forest
                importances_df = pd.DataFrame({
                    'Modality': simplified_modality_names,
                    'Importance': model.feature_importances_,
                    'Rank': range(1, len(simplified_modality_names) + 1)
                }).sort_values('Importance', ascending=False)

                importance_path = os.path.join(method_dir, f"{method}_feature_importance.csv")
                importances_df.to_csv(importance_path, index=False)
                print(f"    📋 保存特征重要性: {importance_path}")

        # ==================== 4. 权重信息保存 ====================
        elif method == 'AUC_weighted':
            # 使用预计算的AUC值
            # 确保模态名称和AUC值数量一致
            if len(simplified_modality_names) == len(modality_aucs_full):
                final_weights = [auc / total_auc_full for auc in modality_aucs_full]

                weights_df = pd.DataFrame({
                    'Modality': simplified_modality_names,
                    'AUC': modality_aucs_full,
                    'Weight': final_weights,
                    'Weighted_Contribution': [auc * weight for auc, weight in zip(modality_aucs_full, final_weights)]
                }).sort_values('Weight', ascending=False)

                weights_path = os.path.join(method_dir, "fusion_weights.csv")
                weights_df.to_csv(weights_path, index=False)
                print(f"    ⚖️ 保存融合权重: {weights_path}")
            else:
                print(
                    f"    ⚠️ 模态名称和AUC值数量不匹配: {len(simplified_modality_names)} vs {len(modality_aucs_full)}")
                # 创建简单的权重文件
                weights_df = pd.DataFrame({
                    'Modality': simplified_modality_names,
                    'Weight': [1 / len(simplified_modality_names)] * len(simplified_modality_names)  # 等权重
                })

                weights_path = os.path.join(method_dir, "fusion_weights.csv")
                weights_df.to_csv(weights_path, index=False)
                print(f"    ⚖️ 保存等权重: {weights_path}")

        # ==================== 5. 条件融合的权重说明 ====================
        elif method == 'Conditional_T2_dceDWI':
            # 保存条件融合的权重策略
            # 根据实际模态情况调整
            strategy_data = []

            # 判断哪些模态存在
            has_t2 = 't2' in ' '.join([name.lower() for name in simplified_modality_names])
            has_dce = 'dce' in ' '.join([name.lower() for name in simplified_modality_names])
            has_dwi = 'dwi' in ' '.join([name.lower() for name in simplified_modality_names])

            if has_t2 and (has_dce or has_dwi):
                strategy_data.append({
                    'Condition': 'T2_proba < 0.4',
                    'Description': 'Use T2 only (Low suspicion)',
                    'Weight_T2': 1.0,
                    'Weight_dce': 0.0 if has_dce else 'N/A',
                    'Weight_DWI': 0.0 if has_dwi else 'N/A'
                })

                # 根据实际存在的模态调整权重
                if has_dce and has_dwi:
                    weights_desc = 'Weighted: T2(60%) + DCE(20%) + DWI(20%)'
                elif has_dce:
                    weights_desc = 'Weighted: T2(75%) + DCE(25%)'
                elif has_dwi:
                    weights_desc = 'Weighted: T2(75%) + DWI(25%)'
                else:
                    weights_desc = 'Use T2 only'

                strategy_data.append({
                    'Condition': '0.4 ≤ T2_proba ≤ 0.6',
                    'Description': weights_desc,
                    'Weight_T2': 0.6 if has_dce or has_dwi else 1.0,
                    'Weight_dce': 0.2 if has_dce else 'N/A',
                    'Weight_DWI': 0.2 if has_dwi else 'N/A'
                })

                strategy_data.append({
                    'Condition': 'T2_proba > 0.6',
                    'Description': 'Use T2 only (High suspicion)',
                    'Weight_T2': 1.0,
                    'Weight_dce': 0.0 if has_dce else 'N/A',
                    'Weight_DWI': 0.0 if has_dwi else 'N/A'
                })
            else:
                strategy_data.append({
                    'Condition': 'Not applicable',
                    'Description': 'T2 modality not found or no DCE/DWI for enhancement',
                    'Weight_T2': 'N/A',
                    'Weight_dce': 'N/A',
                    'Weight_DWI': 'N/A'
                })

            strategy_df = pd.DataFrame(strategy_data)
            strategy_path = os.path.join(method_dir, "fusion_strategy.csv")
            strategy_df.to_csv(strategy_path, index=False)
            print(f"    📝 保存融合策略: {strategy_path}")

        # ==================== 6. 性能指标保存 ====================
        if method in final_metrics:
            metrics_df = pd.DataFrame([{
                'Method': method,
                'AUC': final_metrics[method]['AUC'],
                'AUC_Mean_CV': df_fold[method].mean() if method in df_fold.columns else final_metrics[method]['AUC'],
                'AUC_Std_CV': df_fold[method].std() if method in df_fold.columns else 0,
                'Balanced_Accuracy': final_metrics[method]['Balanced_Accuracy'],
                'Kappa': final_metrics[method]['Kappa'],
                'Precision': classification_report(y_true, (results[method] > 0.5).astype(int),
                                                   output_dict=True)['1']['precision'],
                'Recall': classification_report(y_true, (results[method] > 0.5).astype(int),
                                                output_dict=True)['1']['recall'],
                'F1_Score': classification_report(y_true, (results[method] > 0.5).astype(int),
                                                  output_dict=True)['1']['f1-score']
            }])

            metrics_path = os.path.join(method_dir, f"{method}_performance_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
            print(f"    📈 保存性能指标: {metrics_path}")

        # ==================== 7. 模型信息文件 ====================
        model_info = {
            'method_name': method,
            'description': _get_method_description(method),
            'creation_time': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            'modalities': simplified_modality_names,
            'n_samples': len(y_true),
            'class_distribution': {
                'class_0': int(np.sum(y_true == 0)),
                'class_1': int(np.sum(y_true == 1)),
                'ratio': f"{np.sum(y_true == 1) / np.sum(y_true == 0):.2f}:1"
            },
            'validation_strategy': {
                'outer_folds': 5,
                'inner_folds': 5,
                'shuffle': True,
                'random_state': 42
            },
            'performance': final_metrics.get(method, {}),
            'parameters': _get_method_parameters(method, simplified_modality_names)
        }

        # 添加不确定样本信息（针对条件融合）
        if 'Conditional' in method:
            model_info['conditional_fusion_info'] = {
                'threshold_low': 0.4,
                'threshold_high': 0.6,
                'uncertain_samples': int(uncertain_mask.sum()),
                'uncertain_percentage': float(uncertain_mask.sum() / len(y_true) * 100)
            }

        save_model_info(method_dir, method, model_info)

    print("\n✅ 所有融合方法文件保存完成！")

    # 6. 用全部数据训练最终模型（只训练最佳融合方法）
    print(f"\n🔄 用全部数据训练最终 {best_method[0]} 模型...")

    # 最佳方法的文件夹
    best_method_dir = os.path.join(OUTPUT_DIR, best_method[0])

    if best_method[0] in ['Stacking_LR', 'Stacking_RF']:
        if best_method[0] == 'Stacking_LR':
            final_model = LogisticRegression(
                C=0.1,
                class_weight='balanced',
                max_iter=1000,
                random_state=42,
                solver='liblinear'
            )
        elif best_method[0] == 'Stacking_RF':
            final_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=5,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )

        final_model.fit(X_meta, y_true)
        model_path = os.path.join(best_method_dir, f"best_fusion_model_{best_method[0]}.pkl")
        joblib.dump(final_model, model_path)
        print(f"💾 最终模型保存至: {model_path}")
    else:
        print(f"⚠️ {best_method[0]} 方法无需训练模型")

    # 7. 绘制ROC曲线（分别绘制每种融合方法）
    num_plots = plot_roc_curves_separately(X_meta, y_true, results, final_metrics,
                                           simplified_modality_names, best_method, OUTPUT_DIR)
    print(f"\n📈 共绘制了 {num_plots} 张ROC曲线图")

    # 8. 绘制混淆矩阵
    num_cm_plots = plot_confusion_matrices(X_meta, y_true, results, final_metrics, simplified_modality_names,
                                           OUTPUT_DIR)
    print(f"📊 共生成 {num_cm_plots} 张混淆矩阵图")

    # 8. 保存详细结果（全局结果）
    print("\n💾 保存详细结果...")

    # 保存所有方法的预测结果
    results_df = pd.DataFrame()
    results_df['patient_id'] = patient_ids
    results_df['y_true'] = y_true

    for method_name, probas in results.items():
        results_df[f'{method_name}_proba'] = probas
        results_df[f'{method_name}_pred'] = (probas > 0.5).astype(int)

    results_path = os.path.join(OUTPUT_DIR, "late_fusion_nestedcv_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"  保存全局预测结果: {results_path}")

    # 保存性能指标
    metrics_data = []
    for method, metrics in final_metrics.items():
        if method in df_fold.columns:
            metrics_data.append({
                'Method': method,
                'Type': 'Single Modality' if method.endswith('_only') else 'Fusion',
                'AUC_mean': df_fold[method].mean(),
                'AUC_std': df_fold[method].std(),
                'AUC_overall': metrics['AUC'],
                'BalAcc_overall': metrics['Balanced_Accuracy'],
                'Kappa_overall': metrics['Kappa']
            })

    metrics_df = pd.DataFrame(metrics_data)
    metrics_path = os.path.join(OUTPUT_DIR, "late_fusion_performance_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  保存全局性能指标: {metrics_path}")

    # 保存折叠详细信息
    fold_details_df = pd.DataFrame(fold_details)
    fold_details_path = os.path.join(OUTPUT_DIR, "fold_details.csv")
    fold_details_df.to_csv(fold_details_path, index=False)
    print(f"  保存折叠详情: {fold_details_path}")

    # 9. 生成性能对比表
    print("\n📊 生成性能对比表...")

    # 创建对比表格
    comparison_data = []

    # 添加单模态
    for method in single_modalities:
        if method in final_metrics:
            comparison_data.append({
                'Method': method.replace('_only', ''),
                'Type': 'Single Modality',
                'AUC': final_metrics[method]['AUC'],
                'Balanced_Accuracy': final_metrics[method]['Balanced_Accuracy'],
                'Kappa': final_metrics[method]['Kappa'],
                'Is_Best_Single': method == best_single[0]
            })

    # 添加融合方法
    for method in fusion_methods:
        if method in final_metrics:
            comparison_data.append({
                'Method': method,
                'Type': 'Fusion',
                'AUC': final_metrics[method]['AUC'],
                'Balanced_Accuracy': final_metrics[method]['Balanced_Accuracy'],
                'Kappa': final_metrics[method]['Kappa'],
                'Is_Best_Fusion': method == best_method[0]
            })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_path = os.path.join(OUTPUT_DIR, "performance_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    print(f"  保存性能对比表: {comparison_path}")

    # 10. 生成Markdown格式的性能报告
    print("\n📄 生成Markdown格式性能报告...")

    # 计算不确定样本信息
    t2_proba = X_meta[:, 1]  # 假设T2是第二个模态
    uncertain_mask = (t2_proba >= 0.4) & (t2_proba <= 0.6)
    uncertain_samples = int(uncertain_mask.sum())
    uncertain_ratio = uncertain_samples / len(y_true) * 100

    md_report = f"""# 多模态后融合性能报告

## 数据集信息
- **总样本数**: {len(y_true)}
- **类别分布**: Class 0 (PI-RADS ≤3) = {np.sum(y_true == 0)}, Class 1 (PI-RADS ≥4) = {np.sum(y_true == 1)}
- **模态数量**: {len(simplified_modality_names)}
- **验证策略**: 5×5 嵌套交叉验证
- **T2不确定样本**: {uncertain_samples} 个 ({uncertain_ratio:.1f}%)

## 单模态性能 (5折CV平均AUC ± 标准差)

| 模态 | AUC | Balanced Accuracy | Kappa |
|------|-----|-------------------|-------|"""

    # 添加单模态性能
    for method in sorted(single_modalities):
        if method in final_metrics:
            method_name = method.replace('_only', '')
            metrics = final_metrics[method]
            mean_auc = df_fold[method].mean() if method in df_fold.columns else metrics['AUC']
            std_auc = df_fold[method].std() if method in df_fold.columns else 0
            is_best = "⭐" if method == best_single[0] else ""

            md_report += f"\n| {method_name}{is_best} | {mean_auc:.3f} ± {std_auc:.3f} | {metrics['Balanced_Accuracy']:.3f} | {metrics['Kappa']:.3f} |"

    md_report += f"""

## 融合方法性能 (5折CV平均AUC ± 标准差)

| 方法 | AUC | Balanced Accuracy | Kappa |
|------|-----|-------------------|-------|"""

    # 添加融合方法性能
    for method in sorted(fusion_methods):
        if method in final_metrics:
            metrics = final_metrics[method]
            mean_auc = df_fold[method].mean() if method in df_fold.columns else metrics['AUC']
            std_auc = df_fold[method].std() if method in df_fold.columns else 0
            is_best = "🏆" if method == best_method[0] else ""

            md_report += f"\n| {method}{is_best} | {mean_auc:.3f} ± {std_auc:.3f} | {metrics['Balanced_Accuracy']:.3f} | {metrics['Kappa']:.3f} |"

    md_report += f"""

## 关键发现
1. **最佳单模态**: {best_single[0].replace('_only', '')} (AUC = {best_single[1]['AUC']:.3f})
2. **最佳融合方法**: {best_method[0]} (AUC = {best_method[1]['AUC']:.3f})
3. **性能提升**: {best_method[1]['AUC'] - best_single[1]['AUC']:.3f} AUC units ({((best_method[1]['AUC'] / best_single[1]['AUC'] - 1) * 100):.1f}%)

## 融合方法说明
- **AUC_weighted**: 基于各模态AUC值动态分配权重（AUC越高，权重越大）
- **Stacking_LR**: 使用逻辑回归作为元分类器，学习最优权重组合
- **Stacking_RF**: 使用随机森林作为元分类器，捕捉非线性关系
- **Heuristic**: 固定权重分配 (Clinical:0.3, T2:0.5, DCE:0.1, DWI:0.1)
- **Conditional_T2_dceDWI**: 当T2预测概率在[0.4, 0.6]和clinical[0.3-0.7]同时在不确定区间时，使用Clinical(30%)+T2(60%)+DCE(5%)+DWI(5%)加权升级
- **Conditional_T2_LR**: 当T2预测概率在[0.4, 0.6]和clinical[0.3-0.7]同时在不确定区间时，使用逻辑回归微调器（四个模态）重新预测

## 不确定样本分析
- **T2不确定样本**: {uncertain_samples} 个 ({uncertain_ratio:.1f}%)
- **阈值区间**: [0.4, 0.6] (模拟PI-RADS 3的临床不确定性)
- **临床意义**: 在{uncertain_ratio:.1f}%的病例中，T2单模态预测存在不确定性，需要多模态融合辅助

## 生成的文件
1. **ROC曲线图**:
   - 每种融合方法分别与单模态对比图
   - 所有方法综合对比图
2. **预测结果**:
   - 全局预测结果CSV
   - 每种方法的单独预测结果
3. **性能指标**:
   - 详细性能指标CSV
   - 性能对比表
4. **模型文件**:
   - 最佳融合模型
   - 模型权重/重要性
5. **分析报告**:
   - Markdown格式性能报告
   - 折叠详细信息

*生成时间: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}*
*验证方式: 5×5 嵌套交叉验证 (内层调参，外层评估)*"""

    # 保存Markdown报告
    md_path = os.path.join(OUTPUT_DIR, "performance_report.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"✅ Markdown报告已保存: {md_path}")

    # 11. 绘制性能对比柱状图
    print("\n📊 绘制性能对比柱状图...")

    fig_bar, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 准备数据
    method_names = []
    auc_values = []
    colors_bar = []

    # 单模态（放在左边）
    for method in single_modalities:
        if method in final_metrics:
            method_names.append(method.replace('_only', ''))
            auc_values.append(final_metrics[method]['AUC'])
            colors_bar.append(colors.get(method, '#888888'))

    # 融合方法（放在右边）
    for method in fusion_methods:
        if method in final_metrics:
            method_names.append(method)
            auc_values.append(final_metrics[method]['AUC'])
            colors_bar.append(colors.get(method, '#888888'))

    # 绘制柱状图
    x_pos = np.arange(len(method_names))
    bars = ax1.bar(x_pos, auc_values, color=colors_bar, alpha=0.8, edgecolor='black')

    # 标记最佳方法
    best_single_idx = method_names.index(best_single[0].replace('_only', ''))
    best_fusion_idx = method_names.index(best_method[0])

    bars[best_single_idx].set_edgecolor('red')
    bars[best_single_idx].set_linewidth(2)
    bars[best_fusion_idx].set_edgecolor('blue')
    bars[best_fusion_idx].set_linewidth(2)

    # 添加数值标签
    for i, (bar, auc_val) in enumerate(zip(bars, auc_values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{auc_val:.3f}', ha='center', va='bottom', fontsize=9)

    ax1.set_xlabel('Methods', fontsize=12)
    ax1.set_ylabel('AUC', fontsize=12)
    ax1.set_title('AUC Comparison: Single Modalities vs Fusion Methods', fontsize=14, pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(method_names, rotation=45, ha='right')
    ax1.set_ylim([0.5, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')

    # 添加图例说明
    ax1.text(0.02, 0.98, f"Best Single: {best_single[0].replace('_only', '')}\nBest Fusion: {best_method[0]}",
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

    # 第二张图：性能提升对比
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # 计算相对于最佳单模态的提升
    baseline_auc = best_single[1]['AUC']
    improvements = []
    improvement_names = []
    improvement_colors = []

    for method in fusion_methods:
        if method in final_metrics:
            improvement = final_metrics[method]['AUC'] - baseline_auc
            improvements.append(improvement)
            improvement_names.append(method)
            improvement_colors.append(colors.get(method, '#888888'))

    x_pos2 = np.arange(len(improvements))
    bars2 = ax2.bar(x_pos2, improvements, color=improvement_colors, alpha=0.8, edgecolor='black')

    # 标记最佳融合方法
    best_improvement_idx = improvement_names.index(best_method[0])
    bars2[best_improvement_idx].set_edgecolor('green')
    bars2[best_improvement_idx].set_linewidth(2)

    # 添加数值标签
    for i, (bar, imp) in enumerate(zip(bars2, improvements)):
        height = bar.get_height()
        va = 'bottom' if imp >= 0 else 'top'
        y_offset = 0.005 if imp >= 0 else -0.005
        ax2.text(bar.get_x() + bar.get_width() / 2., height + y_offset,
                 f'{imp:+.3f}', ha='center', va=va, fontsize=9,
                 fontweight='bold' if imp > 0 else 'normal')

    ax2.set_xlabel('Fusion Methods', fontsize=12)
    ax2.set_ylabel('AUC Improvement', fontsize=12)
    ax2.set_title(f'AUC Improvement vs Best Single Modality ({best_single[0].replace("_only", "")})',
                  fontsize=14, pad=15)
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(improvement_names, rotation=45, ha='right')

    # 设置合适的y轴范围
    max_abs_improvement = max(abs(min(improvements)), abs(max(improvements)))
    ax2.set_ylim([-max_abs_improvement * 1.2, max_abs_improvement * 1.5])
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 保存柱状图
    bar_path = os.path.join(OUTPUT_DIR, "AUC_Comparison_BarChart.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    ✅ 性能对比柱状图保存至: {bar_path}")

    # 打印最终结果
    print("\n" + "=" * 60)
    print("🎉 改进版多模态后融合分析完成！")
    print("=" * 60)
    print(f"📊 最佳单模态: {best_single[0].replace('_only', '')} (AUC={best_single[1]['AUC']:.3f})")
    print(f"📊 最佳融合方法: {best_method[0]} (AUC={best_method[1]['AUC']:.3f})")
    print(f"📊 性能提升: +{best_method[1]['AUC'] - best_single[1]['AUC']:.3f} AUC")
    print(f"📊 验证策略: 5×5 嵌套交叉验证")
    print(f"📈 生成图形: {num_plots} 张ROC曲线图 + 1 张柱状图")
    print(f"💾 生成文件: {len(os.listdir(OUTPUT_DIR))} 个文件")
    print("=" * 60)

    # 显示重要文件列表
    print("\n📁 生成的重要文件:")
    important_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.csv', '.pkl', '.md', '.txt'))]
    for file in sorted(important_files):
        file_path = os.path.join(OUTPUT_DIR, file)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"  • {file} ({file_size:.1f} KB)")


if __name__ == "__main__":
    main()