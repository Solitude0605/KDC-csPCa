# -*- coding: utf-8 -*-
"""
融合模型测试脚本 - 使用训练好的融合模型对新数据进行预测
修复版：直接使用外部GT文件，不依赖OOF文件中的GT，添加ROC曲线
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
import matplotlib.pyplot as plt
import time
from functools import wraps
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, confusion_matrix, classification_report
import json
from pathlib import Path
import traceback
from matplotlib import rcParams

warnings.filterwarnings('ignore')
# 设置中文字体和图表样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9


def timeit(func):
    """计算函数执行时间的装饰器"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"⏱️  {func.__name__} 执行时间: {end_time - start_time:.2f} 秒")
        return result

    return wrapper


class FusionModelTester:
    """融合模型测试器 - 加载训练好的融合模型进行预测"""

    def __init__(self, model_path=None, model_dir=None):
        """
        初始化测试器
        Args:
            model_path: 直接指定模型文件路径
            model_dir: 或指定模型文件夹路径
        """
        self.model = None
        self.model_info = None
        self.modality_names = None
        self.fusion_method = None
        self.model_type = None

        if model_path and os.path.exists(model_path):
            self.load_model_from_file(model_path)
        elif model_dir:
            self.model_dir = Path(model_dir)
            self.load_model_from_dir()

    def load_model_from_file(self, model_path):
        """直接从文件加载模型"""
        try:
            self.model = joblib.load(model_path)
            print(f"✅ 从文件加载模型: {os.path.basename(model_path)}")
            self.model_type = type(self.model).__name__

            # 尝试加载对应的模型信息文件
            model_dir = os.path.dirname(model_path)
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            info_path = os.path.join(model_dir, f"{model_name}_info.json")

            if os.path.exists(info_path):
                with open(info_path, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
                print(f"✅ 加载模型信息: {os.path.basename(info_path)}")

                self.modality_names = self.model_info.get('modalities', [])
                self.fusion_method = self.model_info.get('method_name', model_name)
            else:
                # 使用默认值
                self.modality_names = ['Clinical', 'T2', 'DCE', 'DWI']
                self.fusion_method = model_name

            return True

        except Exception as e:
            print(f"❌ 从文件加载模型失败: {e}")
            return False

    def load_model_from_dir(self):
        """从目录加载模型"""
        try:
            # 查找模型文件
            model_files = list(self.model_dir.glob("*model*.pkl"))
            if not model_files:
                print(f"❌ 在 {self.model_dir} 中未找到模型文件")
                return False

            # 加载第一个模型文件
            model_path = model_files[0]
            self.model = joblib.load(model_path)
            print(f"✅ 从目录加载模型: {model_path.name}")
            self.model_type = type(self.model).__name__

            # 查找模型信息文件
            info_files = list(self.model_dir.glob("*info*.json"))
            if info_files:
                with open(info_files[0], 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)
                print(f"✅ 加载模型信息: {info_files[0].name}")
                self.modality_names = self.model_info.get('modalities', [])
                self.fusion_method = self.model_info.get('method_name', self.model_dir.name)
            else:
                self.modality_names = ['Clinical', 'T2', 'DCE', 'DWI']
                self.fusion_method = self.model_dir.name

            return True

        except Exception as e:
            print(f"❌ 从目录加载模型失败: {e}")
            return False

    @timeit
    def load_test_data_with_gt(self, data_paths, gt_file_path, data_format='pkl'):
        """
        加载测试数据并使用外部GT文件
        Args:
            data_paths: 字典，键为模态名，值为OOF文件路径
            gt_file_path: 外部GT文件路径
            data_format: 数据格式，'pkl' 或 'csv'
        Returns:
            X_meta: 元特征矩阵 (n_samples, n_modalities)
            y_true: 真实标签
            patient_ids: 患者ID列表
        """
        print("\n📊 加载测试数据...")
        print(f"📄 GT文件: {os.path.basename(gt_file_path)}")

        # 加载外部GT文件
        try:
            gt_data = pd.read_csv(gt_file_path)
            print(f"✅ 加载GT文件成功，样本数: {len(gt_data)}")
            print(f"   GT文件列名: {list(gt_data.columns)}")

            # 查找合适的标签列 - 优先使用5分制的列
            label_col = None
            # 优先使用 TZ_5_score
            for col in ['TZ_5_score', 'PZ_5_score', 'GT_TZ', 'TZ_3_score', 'PZ_3_score']:
                if col in gt_data.columns:
                    label_col = col
                    break

            # 如果没有找到指定列，再找其他标签列
            if not label_col:
                for col in gt_data.columns:
                    if any(keyword in col.lower() for keyword in ['score', 'label', 'target', 'gt']):
                        label_col = col
                        break

            if not label_col and len(gt_data.columns) > 0:
                label_col = gt_data.columns[-1]

            if label_col:
                print(f"✅ 使用标签列: {label_col}")

                # 提取GT标签
                y_true_raw = gt_data[label_col].values

                # 处理缺失值
                y_true_raw = np.nan_to_num(y_true_raw, nan=0)
                y_true_raw = y_true_raw.astype(int)

                # 打印原始标签分布
                unique_before, counts_before = np.unique(y_true_raw, return_counts=True)
                print(f"📊 原始标签分布 ({label_col}):")
                for val, count in zip(unique_before, counts_before):
                    print(f"   {int(val)}分: {count}个 ({count / len(y_true_raw) * 100:.1f}%)")

                # 检查是否是5分制
                max_score = np.max(y_true_raw)
                print(f"🔍 最高分数: {max_score}")

                # 转换为二分类
                if max_score == 5:
                    print(f"🔧 转换5分制标签为二分类")
                    print(f"   转换规则: >=4 为阳性(1), <4 为阴性(0)")

                    # 详细统计每个分数
                    for score in range(1, 6):
                        count = np.sum(y_true_raw == score)
                        if count > 0:
                            print(f"     {score}分: {count}个 ({count / len(y_true_raw) * 100:.1f}%)")

                    # 执行转换
                    y_true = (y_true_raw >= 4).astype(int)

                elif max_score == 3:
                    print(f"🔧 转换3分制标签为二分类")
                    print(f"   转换规则: >=2 为阳性(1), <2 为阴性(0)")

                    # 详细统计每个分数
                    for score in range(1, 4):
                        count = np.sum(y_true_raw == score)
                        if count > 0:
                            print(f"     {score}分: {count}个 ({count / len(y_true_raw) * 100:.1f}%)")

                    # 执行转换
                    y_true = (y_true_raw >= 2).astype(int)

                elif max_score == 2 or max_score == 1:
                    print(f"✅ 已经是二分类标签")
                    y_true = y_true_raw
                else:
                    print(f"⚠️ 未知评分制 (最高分={max_score})，尝试自动转换")
                    # 尝试找到合适的阈值
                    if max_score >= 4:
                        threshold = 4
                    elif max_score >= 2:
                        threshold = 2
                    else:
                        threshold = 1

                    print(f"   使用阈值 {threshold} 进行二分类转换")
                    y_true = (y_true_raw >= threshold).astype(int)

                # 打印转换后分布
                unique_after, counts_after = np.unique(y_true, return_counts=True)
                print(f"📊 转换后标签分布:")
                for val, count in zip(unique_after, counts_after):
                    label_name = "阴性(0)" if val == 0 else "阳性(1)"
                    print(f"   {label_name}: {count}个 ({count / len(y_true) * 100:.1f}%)")

                # 检查是否有阳性样本
                if np.sum(y_true == 1) == 0:
                    print(f"⚠️ 警告：转换后没有阳性样本！")
                    print(f"   无法计算AUC等性能指标")
                    print(f"   请检查GT文件中的标签定义")
                elif np.sum(y_true == 0) == 0:
                    print(f"⚠️ 警告：转换后没有阴性样本！")
                    print(f"   无法计算AUC等性能指标")

            else:
                print("❌ 未找到合适的标签列")
                y_true = None

            # 提取患者ID
            patient_id_col = None
            for col in gt_data.columns:
                if any(keyword in col.lower() for keyword in ['id', 'patient', 'name']):
                    patient_id_col = col
                    break

            if patient_id_col:
                gt_patient_ids = gt_data[patient_id_col].values.tolist()
                print(f"✅ 从GT文件获取患者ID: {patient_id_col}, 数量: {len(gt_patient_ids)}")
                print(f"   示例: {gt_patient_ids[:3] if len(gt_patient_ids) > 3 else gt_patient_ids}")
            else:
                print("⚠️ GT文件中未找到患者ID列")
                gt_patient_ids = [f"GT_{i + 1:03d}" for i in range(len(gt_data))]

        except Exception as e:
            print(f"❌ 加载GT文件失败: {e}")
            traceback.print_exc()
            return None, None, None

        # 加载各模态的预测概率
        modality_probas = []
        modality_patient_ids = []

        for modality in self.modality_names:
            if modality in data_paths:
                data_path = data_paths[modality]
                if not os.path.exists(data_path):
                    print(f"⚠️ {modality}: 文件不存在 {data_path}")
                    modality_probas.append(np.array([]))
                    continue

                try:
                    if data_format == 'pkl':
                        data = joblib.load(data_path)

                        # 查找预测概率
                        proba_key = None
                        proba_keys = ['pred_probabilities', 'probas', 'probabilities',
                                      'avg_probas', 'predictions']
                        for key in proba_keys:
                            if key in data:
                                proba_key = key
                                break

                        if proba_key:
                            probas = data[proba_key]
                            if isinstance(probas, list):
                                probas = np.array(probas)
                            modality_probas.append(probas)
                            print(f"✅ {modality}: 加载 {len(probas)} 个预测概率")
                        else:
                            print(f"❌ {modality}: 未找到预测概率")
                            modality_probas.append(np.array([]))
                            continue

                        # 查找患者ID
                        pid_key = None
                        pid_keys = ['patient_ids', 'patient_id', 'sample_id', 'ids']
                        for key in pid_keys:
                            if key in data:
                                pid_key = key
                                break

                        if pid_key:
                            pids = data[pid_key]
                            if isinstance(pids, list):
                                modality_patient_ids.append(pids)
                            else:
                                modality_patient_ids.append(pids.tolist())
                            print(f"✅ {modality}: 找到患者ID，数量: {len(pids)}")
                        else:
                            print(f"⚠️ {modality}: 未找到患者ID")

                    elif data_format == 'csv':
                        df = pd.read_csv(data_path)

                        # 查找预测概率列
                        proba_cols = [col for col in df.columns if 'prob' in col.lower()]
                        if proba_cols:
                            probas = df[proba_cols[0]].values
                            modality_probas.append(probas)
                            print(f"✅ {modality}: 加载 {len(probas)} 个预测概率")
                        else:
                            print(f"❌ {modality}: 未找到预测概率列")
                            modality_probas.append(np.array([]))
                            continue

                except Exception as e:
                    print(f"❌ {modality}: 加载失败 - {e}")
                    modality_probas.append(np.array([]))
            else:
                print(f"⚠️ {modality}: 未提供数据路径")
                modality_probas.append(np.array([]))

        # 检查数据长度一致性
        valid_probas = [p for p in modality_probas if len(p) > 0]
        if not valid_probas:
            print("❌ 没有成功加载任何预测概率数据")
            return None, None, None

        # 找到最小样本数
        min_length = min(len(p) for p in valid_probas)
        print(f"\n📏 所有模态的最小样本数: {min_length}")

        # 确保GT标签长度匹配
        if len(y_true) != min_length:
            print(f"⚠️ GT标签长度({len(y_true)}) ≠ 预测数据长度({min_length})")
            print(f"   截取到最小长度: {min_length}")
            y_true = y_true[:min_length]
            gt_patient_ids = gt_patient_ids[:min_length]

        # 截断所有模态数据到最小长度
        truncated_probas = []
        for i in range(len(modality_probas)):
            if len(modality_probas[i]) >= min_length:
                truncated_probas.append(modality_probas[i][:min_length])
            else:
                # 如果长度不足，用0填充
                print(f"⚠️ {self.modality_names[i]}: 长度不足，用0填充")
                padding = np.zeros(min_length)
                if len(modality_probas[i]) > 0:
                    padding[:len(modality_probas[i])] = modality_probas[i]
                truncated_probas.append(padding)

        # 创建元特征矩阵
        X_meta = np.column_stack(truncated_probas)

        print(f"\n✅ 数据加载完成总结:")
        print(f"   元特征矩阵形状: {X_meta.shape}")
        print(f"   GT标签形状: {y_true.shape}")
        print(f"   患者ID数量: {len(gt_patient_ids)}")
        print(f"   阳性样本数: {np.sum(y_true == 1)} ({np.sum(y_true == 1) / len(y_true) * 100:.1f}%)")
        print(f"   阴性样本数: {np.sum(y_true == 0)} ({np.sum(y_true == 0) / len(y_true) * 100:.1f}%)")

        return X_meta, y_true, gt_patient_ids

    @timeit
    def predict(self, X_meta):
        """
        使用融合模型进行预测
        """
        print("\n🔮 进行融合预测...")

        if self.model is None:
            print("❌ 模型未加载")
            return None, None

        try:
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X_meta)[:, 1]
                print("✅ 使用 predict_proba 方法")
            elif hasattr(self.model, 'decision_function'):
                decision = self.model.decision_function(X_meta)
                probas = 1 / (1 + np.exp(-np.clip(decision, -100, 100)))
                print("✅ 使用 decision_function 方法")
            else:
                print("⚠️ 模型不支持概率预测，使用 predict 方法")
                preds = self.model.predict(X_meta)
                probas = preds.astype(float)
                return probas, preds

            preds = (probas >= 0.5).astype(int)

            print(f"✅ 预测完成")
            print(f"   预测概率范围: [{probas.min():.3f}, {probas.max():.3f}]")
            print(f"   阳性预测数: {np.sum(preds == 1)} ({np.sum(preds == 1) / len(preds) * 100:.1f}%)")
            print(f"   阴性预测数: {np.sum(preds == 0)} ({np.sum(preds == 0) / len(preds) * 100:.1f}%)")

            return probas, preds

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            traceback.print_exc()
            return None, None

    @timeit
    def evaluate(self, y_true, probas, preds):
        """
        评估预测结果，计算AUC等指标
        """
        print("\n📊 评估预测结果...")

        if y_true is None:
            print("⚠️ 未提供真实标签，跳过评估")
            return None

        if len(y_true) != len(probas):
            print(f"❌ 标签数量不匹配: 标签={len(y_true)}, 预测={len(probas)}")
            print(f"   将对齐到最小长度")
            min_len = min(len(y_true), len(probas))
            y_true = y_true[:min_len]
            probas = probas[:min_len]
            preds = preds[:min_len]
            print(f"   对齐后长度: {min_len}")

        try:
            # 检查标签的唯一值
            unique_labels = np.unique(y_true)
            print(f"🔍 真实标签唯一值: {unique_labels}")

            # 确保标签是二分类
            if len(unique_labels) < 2:
                print(f"❌ 标签只有一种类别 ({unique_labels})，无法计算AUC")
                print(f"   标签分布:")
                for label in unique_labels:
                    count = np.sum(y_true == label)
                    print(f"     {label}: {count}个 ({count / len(y_true) * 100:.1f}%)")
                return None

            # 计算AUC
            try:
                auc = roc_auc_score(y_true, probas)
                print(f"✅ AUC计算成功: {auc:.4f}")
            except Exception as e:
                print(f"❌ AUC计算失败: {e}")
                # 尝试使用ovr策略
                try:
                    auc = roc_auc_score(y_true, probas, multi_class='ovr')
                    print(f"✅ 使用ovr策略计算AUC成功: {auc:.4f}")
                except:
                    print(f"❌ 所有AUC计算方法都失败")
                    auc = 0.5  # 默认值

            # 计算其他指标
            bal_acc = balanced_accuracy_score(y_true, preds)

            # 计算混淆矩阵
            cm = confusion_matrix(y_true, preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()

                # 计算指标
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                f1_score = 2 * precision * sensitivity / (precision + sensitivity) if (
                                                                                              precision + sensitivity) > 0 else 0
                accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            else:
                tn = fp = fn = tp = 0
                sensitivity = specificity = precision = f1_score = accuracy = 0

            # 打印结果
            print("=" * 60)
            print("🎯 性能评估结果")
            print("=" * 60)
            print(f"  样本总数: {len(y_true)}")
            print(f"  真实阳性: {np.sum(y_true == 1)} ({np.sum(y_true == 1) / len(y_true) * 100:.1f}%)")
            print(f"  真实阴性: {np.sum(y_true == 0)} ({np.sum(y_true == 0) / len(y_true) * 100:.1f}%)")
            print(f"  预测阳性: {np.sum(preds == 1)} ({np.sum(preds == 1) / len(preds) * 100:.1f}%)")
            print(f"  预测阴性: {np.sum(preds == 0)} ({np.sum(preds == 0) / len(preds) * 100:.1f}%)")
            print(f"  🔥 AUC: {auc:.4f} 🔥")
            print(f"  准确率: {accuracy:.4f}")
            print(f"  平衡准确率: {bal_acc:.4f}")
            print(f"  敏感性(召回率): {sensitivity:.4f}")
            print(f"  特异性: {specificity:.4f}")
            print(f"  精确率: {precision:.4f}")
            print(f"  F1分数: {f1_score:.4f}")

            # 打印混淆矩阵
            print(f"\n📋 混淆矩阵:")
            print(f"              Predicted")
            print(f"             0       1")
            print(f"Actual 0  {tn:4d}  {fp:4d}")
            print(f"       1  {fn:4d}  {tp:4d}")

            # 打印分类报告
            print(f"\n📋 分类报告:")
            print(classification_report(y_true, preds, target_names=['阴性(Non-csPCa)', '阳性(csPCa)']))

            # 返回评估结果
            eval_results = {
                'AUC': auc,
                'Accuracy': accuracy,
                'Balanced_Accuracy': bal_acc,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'Precision': precision,
                'F1_Score': f1_score,
                'Confusion_Matrix': cm.tolist(),
                'True_Negative': int(tn),
                'False_Positive': int(fp),
                'False_Negative': int(fn),
                'True_Positive': int(tp),
                'N_Samples': len(y_true),
                'Positive_Rate_True': float(np.sum(y_true == 1) / len(y_true)),
                'Positive_Rate_Pred': float(np.sum(preds == 1) / len(preds))
            }

            return eval_results

        except Exception as e:
            print(f"❌ 评估失败: {e}")
            traceback.print_exc()
            return None

    def plot_roc_curve(self, y_true, probas, auc_score, output_dir):
        """
        绘制并保存ROC曲线
        Args:
            y_true: 真实标签
            probas: 预测概率
            auc_score: AUC分数
            output_dir: 输出目录
        """
        print("\n📈 绘制ROC曲线...")

        try:
            # 计算ROC曲线
            fpr, tpr, thresholds = roc_curve(y_true, probas)

            # 创建图形
            fig, ax = plt.subplots(figsize=(8, 6))

            # 绘制ROC曲线
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'ROC curve (AUC = {auc_score:.3f})')

            # 绘制对角线（随机猜测线）
            ax.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--',
                    label='Random (AUC = 0.500)')

            # 设置图形属性
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
            ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
            ax.set_title(f'ROC Curve - {self.fusion_method}', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(True, alpha=0.3)

            # 添加网格线
            ax.grid(True, which='both', alpha=0.3, linestyle='--')

            # 添加AUC分数文本
            ax.text(0.6, 0.2, f'AUC = {auc_score:.3f}',
                    fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

            # 计算Youden's J统计量找到最佳阈值
            youden_j = tpr - fpr
            best_idx = np.argmax(youden_j)
            best_threshold = thresholds[best_idx]
            best_fpr = fpr[best_idx]
            best_tpr = tpr[best_idx]

            # 在曲线上标记最佳阈值点
            ax.plot(best_fpr, best_tpr, 'ro', markersize=10,
                    label=f'Optimal threshold = {best_threshold:.3f}')

            # 添加最佳阈值信息
            ax.text(best_fpr + 0.02, best_tpr - 0.05,
                    f'Threshold = {best_threshold:.3f}\nSens = {best_tpr:.3f}, Spec = {1 - best_fpr:.3f}',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

            # 设置图形布局
            plt.tight_layout()

            # 保存图形
            roc_path = os.path.join(output_dir, f"{self.fusion_method}_roc_curve.png")
            plt.savefig(roc_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ ROC曲线保存到: {roc_path}")

            # 保存ROC曲线数据
            roc_data = pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            })
            roc_data_path = os.path.join(output_dir, f"{self.fusion_method}_roc_data.csv")
            roc_data.to_csv(roc_data_path, index=False)
            print(f"✅ ROC曲线数据保存到: {roc_data_path}")

            # 打印最佳阈值信息
            print(f"📊 最佳阈值分析:")
            print(f"   最佳阈值: {best_threshold:.3f}")
            print(f"   敏感性: {best_tpr:.3f}")
            print(f"   特异性: {1 - best_fpr:.3f}")
            print(f"   Youden's J统计量: {youden_j[best_idx]:.3f}")

            return roc_path, best_threshold

        except Exception as e:
            print(f"⚠️ 绘制ROC曲线失败: {e}")
            traceback.print_exc()
            return None, None

    def save_results(self, patient_ids, y_true, probas, preds, eval_results, output_dir):
        """
        保存预测结果和评估结果
        """
        print("\n💾 保存结果...")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'sample_index': range(len(patient_ids)),
            'patient_id': patient_ids,
            f'{self.fusion_method}_proba': probas,
            f'{self.fusion_method}_pred': preds,
            f'{self.fusion_method}_label': ['Non-csPCa' if p == 0 else 'csPCa' for p in preds],
            'confidence_score': 1 - 2 * np.abs(probas - 0.5),
            'risk_level': pd.cut(probas, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                                 labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        })

        # 添加真实标签（如果有）
        if y_true is not None:
            results_df['y_true'] = y_true
            results_df['y_label'] = ['Non-csPCa' if y == 0 else 'csPCa' for y in y_true]
            results_df['is_correct'] = (preds == y_true).astype(int)

            # 计算准确率
            correct_predictions = (preds == y_true).sum()
            accuracy = correct_predictions / len(y_true)
            print(f"   预测准确率: {accuracy:.4f} ({correct_predictions}/{len(y_true)})")

            # 添加更多统计信息
            results_df['prediction_error'] = np.abs(probas - y_true)
            results_df['is_fp'] = ((preds == 1) & (y_true == 0)).astype(int)  # 假阳性
            results_df['is_fn'] = ((preds == 0) & (y_true == 1)).astype(int)  # 假阴性
            results_df['is_tp'] = ((preds == 1) & (y_true == 1)).astype(int)  # 真阳性
            results_df['is_tn'] = ((preds == 0) & (y_true == 0)).astype(int)  # 真阴性

        # 保存预测结果
        results_path = output_dir / f"{self.fusion_method}_test_predictions.csv"
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        print(f"✅ 预测结果保存到: {results_path}")

        # 保存评估结果（如果有）
        if eval_results:
            eval_df = pd.DataFrame([eval_results])
            eval_path = output_dir / f"{self.fusion_method}_test_evaluation.csv"
            eval_df.to_csv(eval_path, index=False, encoding='utf-8-sig')
            print(f"✅ 评估结果保存到: {eval_path}")

            # 保存详细评估指标
            detailed_eval = {
                'Metric': ['AUC', 'Accuracy', 'Balanced Accuracy', 'Sensitivity',
                           'Specificity', 'Precision', 'F1 Score',
                           'True Positive', 'False Positive',
                           'True Negative', 'False Negative',
                           'Total Samples', 'Positive Samples (True)',
                           'Negative Samples (True)', 'Positive Rate (Pred)',
                           'Negative Rate (Pred)'],
                'Value': [eval_results['AUC'], eval_results['Accuracy'],
                          eval_results['Balanced_Accuracy'], eval_results['Sensitivity'],
                          eval_results['Specificity'], eval_results['Precision'],
                          eval_results['F1_Score'], eval_results['True_Positive'],
                          eval_results['False_Positive'], eval_results['True_Negative'],
                          eval_results['False_Negative'], eval_results['N_Samples'],
                          int(np.sum(y_true == 1)), int(np.sum(y_true == 0)),
                          eval_results['Positive_Rate_Pred'],
                          1 - eval_results['Positive_Rate_Pred']]
            }

            detailed_eval_df = pd.DataFrame(detailed_eval)
            detailed_eval_path = output_dir / f"{self.fusion_method}_detailed_evaluation.csv"
            detailed_eval_df.to_csv(detailed_eval_path, index=False, encoding='utf-8-sig')
            print(f"✅ 详细评估结果保存到: {detailed_eval_path}")

            # 绘制ROC曲线
            if 'AUC' in eval_results:
                roc_path, best_threshold = self.plot_roc_curve(y_true, probas, eval_results['AUC'], output_dir)
                if roc_path:
                    print(f"📈 ROC曲线已保存")
                    # 保存最佳阈值
                    if best_threshold is not None:
                        threshold_info = {
                            'best_threshold': best_threshold,
                            'current_threshold': 0.5,
                            'recommendation': 'Consider adjusting threshold to improve performance'
                        }
                        threshold_df = pd.DataFrame([threshold_info])
                        threshold_path = output_dir / f"{self.fusion_method}_optimal_threshold.csv"
                        threshold_df.to_csv(threshold_path, index=False)
                        print(f"✅ 最佳阈值信息保存到: {threshold_path}")

        return results_path

    def calculate_prediction_quality_metrics(self, probas, preds):
        """
        计算预测质量指标
        """
        print("\n📊 计算预测质量指标...")

        try:
            metrics = {}

            # 概率分布指标
            metrics['probability_distribution'] = {
                'mean': float(np.mean(probas)),
                'std': float(np.std(probas)),
                'min': float(np.min(probas)),
                'max': float(np.max(probas)),
                'median': float(np.median(probas)),
                'q25': float(np.percentile(probas, 25)),
                'q75': float(np.percentile(probas, 75)),
                'skewness': float(pd.Series(probas).skew()),
                'kurtosis': float(pd.Series(probas).kurtosis())
            }

            # 类别分布指标
            n_total = len(preds)
            n_positive = np.sum(preds == 1)
            n_negative = np.sum(preds == 0)

            metrics['class_distribution'] = {
                'total_samples': int(n_total),
                'positive_samples': int(n_positive),
                'negative_samples': int(n_negative),
                'positive_rate': float(n_positive / n_total),
                'negative_rate': float(n_negative / n_total)
            }

            # 风险等级分布
            risk_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            risk_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
            risk_counts = pd.cut(probas, bins=risk_bins, labels=risk_labels).value_counts()

            metrics['risk_distribution'] = {}
            for label in risk_labels:
                if label in risk_counts:
                    metrics['risk_distribution'][label] = int(risk_counts[label])
                else:
                    metrics['risk_distribution'][label] = 0

            # 打印结果
            print("=" * 50)
            print("📈 预测质量评估结果")
            print("=" * 50)
            print(f"  平均预测概率: {metrics['probability_distribution']['mean']:.4f}")
            print(f"  预测概率标准差: {metrics['probability_distribution']['std']:.4f}")
            print(f"  预测概率范围: [{metrics['probability_distribution']['min']:.4f}, "
                  f"{metrics['probability_distribution']['max']:.4f}]")
            print(f"  预测概率中位数: {metrics['probability_distribution']['median']:.4f}")
            print(f"  阳性预测率: {metrics['class_distribution']['positive_rate']:.4f}")
            print(f"  风险等级分布:")
            for label in risk_labels:
                count = metrics['risk_distribution'][label]
                percentage = count / n_total * 100
                print(f"    {label}: {count}个 ({percentage:.1f}%)")

            return metrics

        except Exception as e:
            print(f"⚠️ 计算预测质量指标失败: {e}")
            return None

    def generate_test_report(self, y_true, probas, preds, eval_results, prediction_metrics,
                             patient_ids, data_paths, gt_file_path, output_dir):
        """
        生成测试报告
        """
        print("\n📋 生成测试报告...")

        # 创建报告内容
        report_content = f"""
融合方法测试报告
===============================
测试时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
融合方法: {self.fusion_method}
模型类型: {self.model_type}
测试样本数: {len(patient_ids)}
GT文件: {os.path.basename(gt_file_path)}

数据来源:
"""
        for modality, data_path in data_paths.items():
            report_content += f"  - {modality}: {os.path.basename(data_path)}\n"

        report_content += f"""
预测结果统计:
  - 平均预测概率: {np.mean(probas):.3f}
  - 预测概率范围: [{np.min(probas):.3f}, {np.max(probas):.3f}]
  - 预测概率标准差: {np.std(probas):.3f}
  - 阳性预测数: {np.sum(preds == 1)} ({np.sum(preds == 1) / len(preds) * 100:.1f}%)
  - 阴性预测数: {np.sum(preds == 0)} ({np.sum(preds == 0) / len(preds) * 100:.1f}%)
"""

        if y_true is not None:
            report_content += f"""
真实标签分布:
  - 真实阳性: {np.sum(y_true == 1)} ({np.sum(y_true == 1) / len(y_true) * 100:.1f}%)
  - 真实阴性: {np.sum(y_true == 0)} ({np.sum(y_true == 0) / len(y_true) * 100:.1f}%)
"""

        if eval_results:
            report_content += f"""
性能评估结果:
  - 🔥 AUC: {eval_results.get('AUC', 0):.4f} 🔥
  - 准确率: {eval_results.get('Accuracy', 0):.4f}
  - 平衡准确率: {eval_results.get('Balanced_Accuracy', 0):.4f}
  - 敏感性: {eval_results.get('Sensitivity', 0):.4f}
  - 特异性: {eval_results.get('Specificity', 0):.4f}
  - 精确率: {eval_results.get('Precision', 0):.4f}
  - F1分数: {eval_results.get('F1_Score', 0):.4f}
  - 真阳性: {eval_results.get('True_Positive', 0)}
  - 假阳性: {eval_results.get('False_Positive', 0)}
  - 真阴性: {eval_results.get('True_Negative', 0)}
  - 假阴性: {eval_results.get('False_Negative', 0)}
"""

        if prediction_metrics:
            report_content += f"""
预测质量分析:
  - 预测概率偏度: {prediction_metrics.get('probability_distribution', {}).get('skewness', 0):.3f}
  - 预测概率峰度: {prediction_metrics.get('probability_distribution', {}).get('kurtosis', 0):.3f}
  - 风险等级分布:
"""
            risk_dist = prediction_metrics.get('risk_distribution', {})
            for risk_level in ['Very Low', 'Low', 'Medium', 'High', 'Very High']:
                count = risk_dist.get(risk_level, 0)
                percentage = count / len(probas) * 100 if len(probas) > 0 else 0
                report_content += f"    {risk_level}: {count}个 ({percentage:.1f}%)\n"

        report_content += f"""
输出文件目录: {output_dir}

文件说明:
1. {self.fusion_method}_test_predictions.csv - 详细的预测结果
2. {self.fusion_method}_test_evaluation.csv - 性能评估指标
3. {self.fusion_method}_detailed_evaluation.csv - 详细评估指标
4. {self.fusion_method}_roc_curve.png - ROC曲线图
5. {self.fusion_method}_roc_data.csv - ROC曲线数据
6. {self.fusion_method}_optimal_threshold.csv - 最佳阈值建议
7. {self.fusion_method}_test_summary.txt - 本报告文件

临床建议:
"""

        # 根据AUC给出建议
        if eval_results and 'AUC' in eval_results:
            auc = eval_results['AUC']
            if auc >= 0.9:
                report_content += "  - 🔥 模型性能优秀，AUC ≥ 0.9，具有很高的临床价值\n"
            elif auc >= 0.8:
                report_content += "  - ✅ 模型性能良好，AUC ≥ 0.8，可用于临床决策支持\n"
            elif auc >= 0.7:
                report_content += "  - ⚠️ 模型性能一般，AUC ≥ 0.7，建议结合临床判断使用\n"
            elif auc >= 0.6:
                report_content += "  - ⚠️ 模型性能有限，AUC ≥ 0.6，建议谨慎使用\n"
            else:
                report_content += "  - ❌ 模型性能较差，AUC < 0.6，不建议用于临床决策\n"

        # 根据阳性预测率给出建议
        if prediction_metrics and 'class_distribution' in prediction_metrics:
            positive_rate = prediction_metrics['class_distribution'].get('positive_rate', 0)
            if positive_rate > 0.7:
                report_content += f"  - ⚠️ 阳性预测率较高 ({positive_rate:.1%})，可能存在过度诊断风险\n"
            elif positive_rate < 0.3:
                report_content += f"  - ⚠️ 阳性预测率较低 ({positive_rate:.1%})，可能存在漏诊风险\n"
            else:
                report_content += f"  - ✅ 阳性预测率适中 ({positive_rate:.1%})，预测结果较为平衡\n"

        # 高风险样本提示
        if prediction_metrics and 'risk_distribution' in prediction_metrics:
            high_risk = prediction_metrics['risk_distribution'].get('High', 0)
            very_high_risk = prediction_metrics['risk_distribution'].get('Very High', 0)
            total_high_risk = high_risk + very_high_risk

            if total_high_risk > 0:
                report_content += f"  - 🔍 发现 {total_high_risk} 个高风险样本，建议重点关注\n"

        # 阈值调整建议
        if eval_results and eval_results.get('AUC', 0) > 0.7:
            report_content += f"""
阈值调整建议:
  - 当前使用阈值: 0.5
  - 建议尝试调整阈值以平衡敏感性和特异性
  - 对于筛查用途，可降低阈值以提高敏感性
  - 对于确诊用途，可提高阈值以提高特异性
  - 详细阈值分析见: {self.fusion_method}_optimal_threshold.csv
"""

        report_content += f"""
注意事项:
1. 本测试使用外部GT文件进行性能评估
2. 预测结果应与临床实际情况结合使用
3. 对于不确定病例（预测概率在0.4-0.6之间），建议进行专家复核
4. 所有预测结果仅供参考，最终诊断应由临床医生决定
"""

        # 保存报告
        report_path = os.path.join(output_dir, f"{self.fusion_method}_test_summary.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✅ 测试报告保存到: {report_path}")
        return report_path


@timeit
def test_single_fusion_method():
    """测试单个融合方法"""
    print("=" * 60)
    print("🎯 单个融合方法测试")
    print("=" * 60)

    # ==============================
    # 配置参数
    # ==============================
    MODEL_PATH = r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\4rd\last_fusion\stacking_lr_model.pkl"

    TEST_DATA_PATHS = {
        'Clinical': r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\clinical_oof_predictions.pkl",
        'T2': r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\T2_oof_predictions.pkl",
        'DCE': r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\dce_oof_predictions.pkl",
        'DWI': r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\DWI_oof_predictions.pkl"
    }

    # 外部GT文件路径
    GT_FILE_PATH = r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\GT_ExternalTest.csv"

    OUTPUT_DIR = r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\predict-value"
    DATA_FORMAT = 'pkl'

    # ==============================
    # 初始化测试器
    # ==============================
    print(f"📂 加载融合模型: {MODEL_PATH}")

    try:
        tester = FusionModelTester(model_path=MODEL_PATH)
        if tester.model is None:
            print("❌ 无法加载融合模型")
            return
    except Exception as e:
        print(f"❌ 初始化测试器失败: {e}")
        traceback.print_exc()
        return

    # ==============================
    # 加载测试数据（使用外部GT）
    # ==============================
    print(f"\n📊 加载测试数据和GT文件...")

    # 检查GT文件是否存在
    if not os.path.exists(GT_FILE_PATH):
        print(f"❌ GT文件不存在: {GT_FILE_PATH}")
        print(f"   请检查路径是否正确")
        return

    X_meta, y_true, patient_ids = tester.load_test_data_with_gt(
        TEST_DATA_PATHS,
        GT_FILE_PATH,
        data_format=DATA_FORMAT
    )

    if X_meta is None or y_true is None:
        print("❌ 无法加载测试数据或GT标签")
        return

    print(f"✅ 测试数据加载完成")
    print(f"   样本数: {X_meta.shape[0]}")
    print(f"   特征数: {X_meta.shape[1]}")
    print(f"   真实标签分布: 0={np.sum(y_true == 0)}, 1={np.sum(y_true == 1)}")

    # ==============================
    # 进行预测
    # ==============================
    probas, preds = tester.predict(X_meta)

    if probas is None:
        print("❌ 预测失败")
        return

    # ==============================
    # 评估结果
    # ==============================
    eval_results = tester.evaluate(y_true, probas, preds)

    # ==============================
    # 计算预测质量指标
    # ==============================
    prediction_metrics = tester.calculate_prediction_quality_metrics(probas, preds)

    # ==============================
    # 保存结果
    # ==============================
    method_output_dir = os.path.join(OUTPUT_DIR, tester.fusion_method)
    Path(method_output_dir).mkdir(parents=True, exist_ok=True)

    results_path = tester.save_results(patient_ids, y_true, probas, preds, eval_results,
                                       method_output_dir)

    # ==============================
    # 生成测试报告
    # ==============================
    report_path = tester.generate_test_report(y_true, probas, preds, eval_results,
                                              prediction_metrics, patient_ids,
                                              TEST_DATA_PATHS, GT_FILE_PATH,
                                              method_output_dir)

    print(f"\n🎉 {tester.fusion_method} 测试完成!")
    print(f"📁 结果保存在: {method_output_dir}")

    # 打印重要指标
    if eval_results:
        print("\n📊 重要指标汇总:")
        print(f"  🔥 AUC: {eval_results.get('AUC', 0):.4f}")
        print(f"  准确率: {eval_results.get('Accuracy', 0):.4f}")
        print(f"  敏感性: {eval_results.get('Sensitivity', 0):.4f}")
        print(f"  特异性: {eval_results.get('Specificity', 0):.4f}")

        # 打印ROC曲线信息
        print(f"\n📈 ROC曲线信息:")
        print(f"  曲线已保存为PNG格式")
        print(f"  详细数据已保存为CSV格式")

    # 打印文件列表
    print(f"\n📄 生成的文件:")
    print(f"  ✓ {tester.fusion_method}_test_predictions.csv")
    print(f"  ✓ {tester.fusion_method}_test_evaluation.csv")
    print(f"  ✓ {tester.fusion_method}_detailed_evaluation.csv")
    print(f"  ✓ {tester.fusion_method}_roc_curve.png")
    print(f"  ✓ {tester.fusion_method}_roc_data.csv")
    print(f"  ✓ {tester.fusion_method}_optimal_threshold.csv")
    print(f"  ✓ {tester.fusion_method}_test_summary.txt")

    return True


@timeit
def batch_test_all_fusion_methods():
    """批量测试所有融合方法"""
    print("=" * 60)
    print("🎯 批量测试所有融合方法")
    print("=" * 60)

    # ==============================
    # 配置参数
    # ==============================
    MODELS_DIR = r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\4rd\last_fusion"

    TEST_DATA_PATHS = {
        'Clinical': r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\clinical_oof_predictions.pkl",
        'T2': r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\T2_oof_predictions.pkl",
        'DCE': r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\dce_oof_predictions.pkl",
        'DWI': r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\DWI_oof_predictions.pkl"
    }

    # 外部GT文件路径
    GT_FILE_PATH = r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\GT_ExternalTest.csv"

    OUTPUT_DIR = r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\predict-value"
    DATA_FORMAT = 'pkl'

    # ==============================
    # 获取所有融合方法
    # ==============================
    print(f"🔍 在目录中查找模型文件: {MODELS_DIR}")

    fusion_methods = []
    # 查找.pkl模型文件
    model_files = list(Path(MODELS_DIR).glob("*.pkl"))

    for model_file in model_files:
        model_name = model_file.stem
        # 检查是否是模型文件
        if 'model' in model_name.lower() and not model_name.startswith('.'):
            fusion_methods.append(model_name)
            print(f"  ✅ 找到模型: {model_name}")

    # 如果没有找到.pkl文件，查找包含模型文件的目录
    if not fusion_methods:
        for item in os.listdir(MODELS_DIR):
            item_path = os.path.join(MODELS_DIR, item)
            if os.path.isdir(item_path):
                model_files_in_dir = list(Path(item_path).glob("*.pkl"))
                if model_files_in_dir:
                    fusion_methods.append(item)
                    print(f"  ✅ 找到模型目录: {item}")

    print(f"🔍 共发现 {len(fusion_methods)} 个融合方法")

    if not fusion_methods:
        print("❌ 未找到融合方法")
        return

    # ==============================
    # 批量测试
    # ==============================
    all_results = []

    for fusion_method in fusion_methods:
        print(f"\n{'=' * 40}")
        print(f"测试融合方法: {fusion_method}")
        print(f"{'=' * 40}")

        try:
            # 构建模型路径
            model_path = None

            # 尝试直接查找.pkl文件
            possible_model_path = os.path.join(MODELS_DIR, f"{fusion_method}.pkl")
            if os.path.exists(possible_model_path):
                model_path = possible_model_path
            else:
                # 尝试在子目录中查找
                model_dir = os.path.join(MODELS_DIR, fusion_method)
                if os.path.isdir(model_dir):
                    model_files_in_dir = list(Path(model_dir).glob("*.pkl"))
                    if model_files_in_dir:
                        model_path = str(model_files_in_dir[0])

            if not model_path or not os.path.exists(model_path):
                print(f"⚠️ 无法找到模型文件，跳过 {fusion_method}")
                continue

            # 初始化测试器
            tester = FusionModelTester(model_path=model_path)
            if tester.model is None:
                print(f"⚠️ 无法加载模型，跳过")
                continue

            # 加载测试数据（使用外部GT）
            X_meta, y_true, patient_ids = tester.load_test_data_with_gt(
                TEST_DATA_PATHS,
                GT_FILE_PATH,
                data_format=DATA_FORMAT
            )

            if X_meta is None:
                print(f"⚠️ 无法加载测试数据，跳过")
                continue

            # 进行预测
            probas, preds = tester.predict(X_meta)
            if probas is None:
                print(f"⚠️ 预测失败，跳过")
                continue

            # 评估结果
            eval_results = tester.evaluate(y_true, probas, preds)

            # 计算预测质量指标
            prediction_metrics = tester.calculate_prediction_quality_metrics(probas, preds)

            # 保存结果
            method_output_dir = os.path.join(OUTPUT_DIR, fusion_method)
            Path(method_output_dir).mkdir(parents=True, exist_ok=True)

            tester.save_results(patient_ids, y_true, probas, preds, eval_results,
                                method_output_dir)

            # 生成测试报告
            tester.generate_test_report(y_true, probas, preds, eval_results,
                                        prediction_metrics, patient_ids,
                                        TEST_DATA_PATHS, GT_FILE_PATH,
                                        method_output_dir)

            # 保存结果到汇总
            method_result = {
                'Fusion_Method': fusion_method,
                'Model_Type': tester.model_type,
                'N_Samples': len(patient_ids),
                'Mean_Probability': float(np.mean(probas)),
                'Positive_Rate': float(np.sum(preds == 1) / len(preds)),
                'AUC': eval_results.get('AUC') if eval_results else None,
                'Accuracy': eval_results.get('Accuracy') if eval_results else None,
                'Sensitivity': eval_results.get('Sensitivity') if eval_results else None,
                'Specificity': eval_results.get('Specificity') if eval_results else None,
                'F1_Score': eval_results.get('F1_Score') if eval_results else None
            }
            all_results.append(method_result)

            print(f"✅ {fusion_method} 测试完成")

        except Exception as e:
            print(f"❌ {fusion_method} 测试失败: {e}")
            traceback.print_exc()
            continue

    # ==============================
    # 生成批量测试报告
    # ==============================
    print(f"\n{'=' * 60}")
    print("📋 生成批量测试报告")
    print(f"{'=' * 60}")

    if all_results:
        summary_df = pd.DataFrame(all_results)

        # 按AUC排序（如果有）
        if 'AUC' in summary_df.columns and not summary_df['AUC'].isna().all():
            summary_df = summary_df.sort_values('AUC', ascending=False)

        # 保存汇总结果
        summary_path = os.path.join(OUTPUT_DIR, "batch_test_summary.csv")
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"✅ 批量测试汇总保存到: {summary_path}")

        # 打印汇总结果
        print("\n📊 批量测试汇总结果:")
        print("-" * 80)
        print(summary_df.to_string())
        print("-" * 80)

        # 生成详细报告
        report_content = f"""# 融合方法批量测试报告

## 测试信息
- **测试时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
- **GT文件**: {os.path.basename(GT_FILE_PATH)}
- **测试方法数**: {len(all_results)}
- **测试样本数**: {summary_df['N_Samples'].iloc[0] if len(summary_df) > 0 else 'Unknown'}

## 性能对比

| 融合方法 | 模型类型 | 样本数 | 平均概率 | 阳性率 | AUC | 准确率 | 敏感性 | 特异性 | F1分数 |
|---------|---------|--------|---------|--------|-----|--------|--------|--------|--------|"""

        for _, row in summary_df.iterrows():
            auc_str = f"{row['AUC']:.4f}" if pd.notna(row['AUC']) else "N/A"
            acc_str = f"{row['Accuracy']:.4f}" if pd.notna(row['Accuracy']) else "N/A"
            sens_str = f"{row['Sensitivity']:.4f}" if pd.notna(row['Sensitivity']) else "N/A"
            spec_str = f"{row['Specificity']:.4f}" if pd.notna(row['Specificity']) else "N/A"
            f1_str = f"{row['F1_Score']:.4f}" if pd.notna(row['F1_Score']) else "N/A"

            report_content += f"\n| {row['Fusion_Method']} | {row['Model_Type']} | {row['N_Samples']} | {row['Mean_Probability']:.3f} | {row['Positive_Rate']:.3f} | {auc_str} | {acc_str} | {sens_str} | {spec_str} | {f1_str} |"

        # 最佳方法
        if 'AUC' in summary_df.columns and not summary_df['AUC'].isna().all():
            best_method = summary_df.iloc[0]
            report_content += f"""

## 最佳方法
- **方法名称**: {best_method['Fusion_Method']}
- **AUC**: {best_method['AUC']:.4f}
- **准确率**: {best_method['Accuracy']:.4f}
- **模型类型**: {best_method['Model_Type']}

## 测试数据说明
测试数据来自外部验证集，包含以下模态的预测概率:
"""
            for modality, path in TEST_DATA_PATHS.items():
                report_content += f"- **{modality}**: {path}\n"

            report_content += f"""
## 输出文件
所有方法的详细结果保存在: {OUTPUT_DIR}

每个方法包含以下文件:
1. 预测结果CSV
2. 评估指标CSV
3. 详细评估CSV
4. ROC曲线PNG
5. ROC数据CSV
6. 最佳阈值CSV
7. 测试报告TXT

## 临床建议
"""
            # 分析阳性率
            avg_positive_rate = summary_df['Positive_Rate'].mean()
            report_content += f"- 平均阳性预测率为 {avg_positive_rate:.1%}\n"

            # 分析AUC
            valid_aucs = summary_df['AUC'].dropna()
            if len(valid_aucs) > 0:
                avg_auc = valid_aucs.mean()
                max_auc = valid_aucs.max()
                min_auc = valid_aucs.min()
                report_content += f"- 平均AUC为 {avg_auc:.4f}，范围: [{min_auc:.4f}, {max_auc:.4f}]\n"

            # 最佳方法建议
            report_content += f"""
## 使用建议
1. 对于临床决策，建议使用AUC最高的 **{best_method['Fusion_Method']}** 方法
2. 查看该方法的ROC曲线图: {best_method['Fusion_Method']}_roc_curve.png
3. 不同融合方法的预测结果可能存在差异，可参考多个方法的结果
4. 对于不确定病例（预测概率在0.4-0.6之间），建议进行专家复核
5. 所有预测结果仅供参考，最终诊断应由临床医生决定
"""

        # 保存报告
        report_path = os.path.join(OUTPUT_DIR, "batch_test_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"✅ 批量测试报告保存到: {report_path}")

    print(f"\n🎉 批量测试完成!")
    print(f"📊 成功测试了 {len(all_results)}/{len(fusion_methods)} 个融合方法")
    print(f"📁 所有结果保存在: {OUTPUT_DIR}")


def main():
    """主函数"""
    print("=" * 60)
    print("🎯 融合模型测试系统 (含ROC曲线)")
    print("=" * 60)
    print("\n请选择测试模式:")
    print("1. 测试单个融合方法")
    print("2. 批量测试所有融合方法")
    print("3. 退出")

    choice = input("\n请输入选择 (1-3): ").strip()

    if choice == '1':
        test_single_fusion_method()
    elif choice == '2':
        batch_test_all_fusion_methods()
    elif choice == '3':
        print("👋 退出程序")
    else:
        print("❌ 无效选择")


if __name__ == "__main__":
    main()