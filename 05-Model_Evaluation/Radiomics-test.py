# -*- coding: utf-8 -*-
"""
最终修复版MRI预测系统 - 每个模态单独保存OOF数据（包含GT评估）
"""

import pandas as pd
import numpy as np
import joblib
import warnings
import os
import matplotlib.pyplot as plt
from pathlib import Path
import re
import json
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'Calibri'
plt.rcParams['axes.unicode_minus'] = False


class AdaptiveMRIPredictor:
    """自适应MRI预测器 - 处理特征数不匹配"""

    def __init__(self, model_dirs):
        self.model_dirs = model_dirs
        self.models = {}
        self.load_models()

    def load_models(self):
        """加载模型"""
        print("📂 正在加载模型...")

        for modality, model_dir in self.model_dirs.items():
            # 查找模型文件
            model_files = []
            for file in Path(model_dir).glob("*.pkl"):
                model_files.append(str(file))

            if not model_files:
                print(f"⚠️ {modality}: 目录中没有.pkl文件")
                continue

            # 尝试加载每个模型文件
            loaded = False
            for model_path in model_files:
                try:
                    model_data = joblib.load(model_path)

                    # 检查是否是完整的模型数据
                    if 'feature_names' in model_data:
                        self.models[modality] = {
                            'data': model_data,
                            'model_name': model_data.get('model_name', f'{modality}_model'),
                            'feature_names': model_data.get('feature_names', []),
                            'top20_features': model_data.get('top20_features', []),
                            'performance': model_data.get('performance_metrics', {}),
                            'path': model_path
                        }

                        loaded = True
                        print(f"✅ {modality}: 加载成功")
                        print(f"   - AUC: {model_data.get('performance_metrics', {}).get('AUC', 'N/A'):.3f}")
                        print(f"   - 特征数: {len(model_data.get('feature_names', []))}")
                        break
                except Exception as e:
                    continue

            if not loaded:
                print(f"❌ {modality}: 无法加载任何模型文件")

        if not self.models:
            raise ValueError("未加载任何模型！")

        print(f"\n✅ 成功加载 {len(self.models)} 个模型")

    def extract_base_feature_name(self, feature_name):
        """提取基础特征名（移除original_前缀，转为小写）"""
        # 移除original_前缀
        if feature_name.startswith('original_'):
            base_name = feature_name.replace('original_', '', 1)
        else:
            base_name = feature_name

        # 转为小写，移除所有非字母数字字符（除了下划线）
        base_name = base_name.lower()
        base_name = re.sub(r'[^a-z0-9_]', '', base_name)

        return base_name

    def find_best_feature_match(self, model_features, available_features):
        """为每个模型特征找到最佳匹配的新数据特征"""
        print(f"\n🔄 Feature matching:")

        # 获取模型特征的基础名
        model_base_features = []
        for feat in model_features:
            base_name = self.extract_base_feature_name(feat)
            model_base_features.append(base_name)

        # 获取新数据特征的基础名
        new_base_features = []
        for feat in available_features:
            base_name = self.extract_base_feature_name(feat)
            new_base_features.append(base_name)

        # 创建映射
        mapping = {}
        used_new_features = set()

        # 策略1: 精确匹配
        for model_feat, model_base in zip(model_features, model_base_features):
            if model_base in new_base_features:
                idx = new_base_features.index(model_base)
                if idx not in used_new_features:
                    mapping[model_feat] = available_features[idx]
                    used_new_features.add(idx)
                    print(f"    ✅ 精确匹配: {model_feat} -> {available_features[idx]}")

        print(f"    精确匹配数: {len(mapping)}")

        # 策略2: 部分匹配（如果精确匹配不够）
        if len(mapping) < len(model_features):
            print(f"    尝试部分匹配...")

            for model_feat, model_base in zip(model_features, model_base_features):
                if model_feat in mapping:
                    continue  # 已经匹配过了

                # 将模型特征名拆分成部分
                model_parts = set(model_base.split('_'))

                # 在新特征中寻找最佳匹配
                best_match = None
                best_score = 0

                for i, new_base in enumerate(new_base_features):
                    if i in used_new_features:
                        continue

                    # 计算匹配分数
                    new_parts = set(new_base.split('_'))
                    common_parts = model_parts.intersection(new_parts)
                    score = len(common_parts) / max(len(model_parts), len(new_parts))

                    if score > best_score and score > 0.3:
                        best_score = score
                        best_match = i

                if best_match is not None:
                    mapping[model_feat] = available_features[best_match]
                    used_new_features.add(best_match)
                    print(f"    🔄 部分匹配: {model_feat} -> {available_features[best_match]} (分数: {best_score:.2f})")

        print(f"    总匹配数: {len(mapping)}/{len(model_features)}")
        return mapping

    def prepare_data_with_mapping(self, df, model_features, mapping):
        """
        根据映射准备数据
        如果某些特征没有匹配，用均值填充
        """
        X_data = []
        missing_features = []
        present_features = []

        for model_feat in model_features:
            if model_feat in mapping:
                # 特征有匹配
                new_feat = mapping[model_feat]
                values = df[new_feat].values
                present_features.append(model_feat)
            else:
                # 特征没有匹配，用0填充
                values = np.zeros(len(df))
                missing_features.append(model_feat)

            X_data.append(values)

        # 转换为numpy数组
        X = np.column_stack(X_data)

        print(f"    准备的数据形状: {X.shape}")
        print(f"    匹配特征: {len(present_features)}")
        print(f"    缺失特征: {len(missing_features)}")
        if missing_features:
            print(f"    缺失特征示例: {missing_features[:3]}")

        return X, present_features, missing_features

    def adaptive_predict(self, X, modality, present_features, missing_features):
        """
        自适应预测：处理缺失特征
        返回：预测概率，预测类别，以及每折的详细预测结果
        """
        if modality not in self.models:
            return None, None, None

        model_data = self.models[modality]['data']

        # 检查是否有交叉验证模型
        if 'all_folds_models' not in model_data:
            print(f"⚠️ {modality}: 没有保存交叉验证模型")
            return None, None, None

        models_per_fold = model_data['all_folds_models']

        if len(models_per_fold) == 0:
            print(f"❌ {modality}: 没有可用的模型")
            return None, None, None

        all_probas = []
        fold_details = []  # 保存每折的详细信息

        for fold_idx, fold_data in enumerate(models_per_fold):
            try:
                # 获取预处理工具
                imputer = fold_data['imputer']
                var_selector = fold_data['var_selector']
                scaler = fold_data['scaler']
                model = fold_data['model']

                # 应用预处理流程
                X_imp = imputer.transform(X)
                X_var = var_selector.transform(X_imp)
                X_sc = scaler.transform(X_var)

                # 预测
                if hasattr(model, "predict_proba"):
                    probas = model.predict_proba(X_sc)[:, 1]
                else:
                    decision = model.decision_function(X_sc)
                    probas = 1 / (1 + np.exp(-np.clip(decision, -100, 100)))

                all_probas.append(probas)

                # 保存这折的详细信息
                fold_info = {
                    'fold_idx': fold_idx,
                    'probas': probas,
                    'model_type': type(model).__name__,
                    'has_predict_proba': hasattr(model, "predict_proba")
                }
                fold_details.append(fold_info)

                print(f"    ✅ 第{fold_idx + 1}折预测成功")

            except Exception as e:
                print(f"    ⚠️ 第{fold_idx + 1}折失败: {e}")
                # 尝试使用备用预测方法
                backup_probas = self.backup_predict(X, fold_data, missing_features)
                if backup_probas is not None:
                    all_probas.append(backup_probas)

                    # 保存备用方法的详细信息
                    fold_info = {
                        'fold_idx': fold_idx,
                        'probas': backup_probas,
                        'model_type': 'Backup_Prediction',
                        'has_predict_proba': False,
                        'note': 'Used backup prediction method'
                    }
                    fold_details.append(fold_info)

                    print(f"    🔄 使用备用方法第{fold_idx + 1}折")

        if not all_probas:
            print(f"❌ {modality}: 所有折都失败")
            return None, None, None

        # 计算平均概率
        avg_probas = np.mean(all_probas, axis=0)
        preds = (avg_probas >= 0.5).astype(int)

        print(f"    ✅ 预测完成，平均概率: {np.mean(avg_probas):.3f}")

        return avg_probas, preds, fold_details

    def backup_predict(self, X, fold_data, missing_features):
        """备用预测方法：使用简化的预处理"""
        try:
            model = fold_data['model']

            # 简化预处理：只做标准化
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_sc = scaler.fit_transform(X)

            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X_sc)[:, 1]
            else:
                # 对于没有predict_proba的模型，返回0.5
                probas = np.full(len(X), 0.5)

            return probas
        except:
            return None

    def predict_single_modality(self, data_path, modality):
        """预测单个模态"""
        if modality not in self.models:
            return None

        print(f"\n{'=' * 40}")
        print(f"📊 {modality} 预测")
        print(f"{'=' * 40}")

        # 加载数据
        df = pd.read_csv(data_path)
        print(f"数据形状: {df.shape}")

        # 检查是否有GT列
        gt_columns = []
        for col in df.columns:
            if col.upper() in ['GT', 'GOLD_STANDARD', 'LABEL', 'TARGET']:
                gt_columns.append(col)

        if gt_columns:
            # 使用第一个找到的GT列
            gt_column = gt_columns[0]
            y_true = df[gt_column].values
            print(f"✅ 找到GT列: {gt_column}, 阳性样本数: {np.sum(y_true == 1)}, 阴性样本数: {np.sum(y_true == 0)}")
        else:
            print("⚠️ 未找到GT列，将只进行预测不进行评估")
            y_true = None

        # 获取可用特征（排除诊断信息和GT列）
        available_features = []
        for col in df.columns:
            col_lower = col.lower()
            if ('diagnostics' not in col_lower and
                    col not in ['patient_id', 'patient_id_2', 'patient_id_a', 'patient_id_d'] and
                    col.upper() not in ['GT', 'GOLD_STANDARD', 'LABEL', 'TARGET']):
                available_features.append(col)

        print(f"可用特征数: {len(available_features)}")

        # 获取模型特征
        model_info = self.models[modality]
        model_features = model_info['feature_names']

        print(f"模型期望特征数: {len(model_features)}")
        print(f"模型特征示例: {model_features[:3]}")

        # 特征匹配
        mapping = self.find_best_feature_match(model_features, available_features)

        if not mapping:
            print(f"❌ {modality}: 未找到任何匹配的特征")
            return None

        # 准备数据
        X, present_features, missing_features = self.prepare_data_with_mapping(
            df, model_features, mapping
        )

        # 自适应预测
        probas, preds, fold_details = self.adaptive_predict(X, modality, present_features, missing_features)

        # ==============================
        # 关键修改：添加GT对齐检查
        # ==============================
        if y_true is not None:
            try:
                # 确保长度匹配
                min_len = min(len(y_true), len(probas))

                if min_len > 0:
                    print(f"\n📊 {modality} GT对齐检查:")
                    print(f"  GT样本数: {len(y_true)}")
                    print(f"  预测样本数: {len(probas)}")
                    print(f"  有效对齐样本数: {min_len}")

                    # 如果长度不匹配，进行调整
                    if min_len < len(y_true):
                        print(f"⚠️  警告: GT长度({len(y_true)}) > 预测长度({len(probas)}), 将截断GT")
                        y_true = y_true[:min_len]
                    elif min_len < len(probas):
                        print(f"⚠️  警告: 预测长度({len(probas)}) > GT长度({len(y_true)}), 将截断预测")
                        probas = probas[:min_len]
                        preds = preds[:min_len]
                        # 还需要调整fold_details中的预测
                        for fold_info in fold_details:
                            if 'probas' in fold_info and len(fold_info['probas']) > min_len:
                                fold_info['probas'] = fold_info['probas'][:min_len]

                    # 计算对齐后的性能
                    print(f"✅  成功对齐{min_len}个样本")

            except Exception as e:
                print(f"⚠️ GT对齐检查失败: {e}")

        # 如果有GT，计算性能指标
        performance_metrics = None
        if y_true is not None:
            try:
                # 确保长度匹配
                min_len = min(len(y_true), len(probas))
                y_true_valid = y_true[:min_len]
                probas_valid = probas[:min_len]
                preds_valid = preds[:min_len]

                # 计算性能指标
                auc = roc_auc_score(y_true_valid, probas_valid)
                accuracy = accuracy_score(y_true_valid, preds_valid)
                f1 = f1_score(y_true_valid, preds_valid)

                # 计算敏感性和特异性
                cm = confusion_matrix(y_true_valid, preds_valid)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # 阳性预测值
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # 阴性预测值
                else:
                    sensitivity = specificity = ppv = npv = 0

                performance_metrics = {
                    'AUC': auc,
                    'Accuracy': accuracy,
                    'F1_score': f1,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'PPV': ppv,
                    'NPV': npv,
                    'Confusion_matrix': cm.tolist(),
                    'n_samples': min_len,
                    'n_positive': int(np.sum(y_true_valid == 1)),
                    'n_negative': int(np.sum(y_true_valid == 0))
                }

                print(f"\n📈 {modality} 性能指标:")
                print(f"  - AUC: {auc:.3f}")
                print(f"  - Accuracy: {accuracy:.3f}")
                print(f"  - F1 Score: {f1:.3f}")
                print(f"  - Sensitivity: {sensitivity:.3f}")
                print(f"  - Specificity: {specificity:.3f}")
                print(f"  - PPV: {ppv:.3f}")
                print(f"  - NPV: {npv:.3f}")
                print(
                    f"  - 样本数: {min_len} (阳性: {int(np.sum(y_true_valid == 1))}, 阴性: {int(np.sum(y_true_valid == 0))})")

            except Exception as e:
                print(f"⚠️ 计算性能指标失败: {e}")

        # 返回结果
        return {
            'probas': probas,
            'preds': preds,
            'mapping': mapping,
            'present_features': present_features,
            'missing_features': missing_features,
            'X_shape': X.shape,
            'fold_details': fold_details,
            'gt_values': y_true,  # ✅ 确保包含GT
            'performance_metrics': performance_metrics,
            'model_info': {
                'model_name': model_info['model_name'],
                'performance': model_info['performance'],
                'model_path': model_info['path'],
                'expected_features': len(model_features)
            }
        }


def save_individual_oof_data(modality, result, output_dir):
    """
    为每个模态单独保存OOF数据
    """
    print(f"\n💾 正在保存 {modality} 的OOF数据...")

    # 创建模态特定的输出目录
    modality_oof_dir = os.path.join(output_dir, f"{modality}_oof")
    Path(modality_oof_dir).mkdir(parents=True, exist_ok=True)

    # 获取预期的特征数量
    expected_features = result.get('model_info', {}).get('expected_features', 20)

    # 构建OOF数据结构
    oof_data = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'modality': modality,
        'pred_probabilities': result['probas'].tolist() if isinstance(result['probas'], np.ndarray) else result[
            'probas'],
        'pred_classes': result['preds'].tolist() if isinstance(result['preds'], np.ndarray) else result['preds'],
        'gt_values': result['gt_values'].tolist() if result.get('gt_values') is not None else None,
        'fold_details': result.get('fold_details', []),
        'feature_mapping': result['mapping'],
        'present_features': result['present_features'],
        'missing_features': result['missing_features'],
        'model_info': result.get('model_info', {}),
        'performance_metrics': result.get('performance_metrics', {}),
        'metadata': {
            'n_samples': len(result['probas']),
            'n_features': result['X_shape'][1],
            'n_folds': len(result.get('fold_details', [])),
            'match_rate': len(result['present_features']) / expected_features if expected_features > 0 else 0,
            'match_count': len(result['present_features']),
            'missing_count': len(result['missing_features']),
            'expected_features': expected_features,
            'mean_probability': float(np.mean(result['probas'])),
            'positive_rate': float(np.mean(result['preds'] == 1)),
            'positive_count': int(np.sum(result['preds'] == 1)),
            'negative_count': int(np.sum(result['preds'] == 0))
        }
    }

    # 保存为pickle格式
    oof_pickle_path = os.path.join(modality_oof_dir, f"{modality}_oof_predictions.pkl")
    joblib.dump(oof_data, oof_pickle_path)
    print(f"✅ {modality} OOF数据已保存到: {oof_pickle_path}")

    # 同时保存为JSON格式（便于查看）
    try:
        json_data = {
            'timestamp': oof_data['timestamp'],
            'modality': modality,
            'metadata': oof_data['metadata'],
            'feature_mapping_count': len(oof_data['feature_mapping']),
            'present_features_count': len(oof_data['present_features']),
            'missing_features_count': len(oof_data['missing_features']),
            'performance_metrics': oof_data.get('performance_metrics', {}),
            'model_info': {
                'model_name': oof_data['model_info'].get('model_name'),
                'expected_features': oof_data['model_info'].get('expected_features'),
                'original_performance': oof_data['model_info'].get('performance', {})
            }
        }

        json_path = os.path.join(modality_oof_dir, f"{modality}_oof_summary.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ {modality} OOF摘要已保存到: {json_path}")

    except Exception as e:
        print(f"⚠️ 保存{modality} JSON摘要失败: {e}")

    # 创建详细的CSV文件
    try:
        n_samples = len(result['probas'])

        # 创建基础DataFrame
        df_modality = pd.DataFrame({
            'sample_id': [f"{modality}_{i + 1:03d}" for i in range(n_samples)],
            'pred_prob': result['probas'],
            'pred_class': result['preds'],
            'pred_label': ['Non-csPCa' if c == 0 else 'PI-RADS ≥4' for c in result['preds']],
            'confidence': np.abs(result['probas'] - 0.5) * 2
        })

        # 在保存OOF数据后，添加GT评估
        if result.get('gt_values') is not None and len(result['probas']) > 0:
            try:
                min_len = min(len(result['gt_values']), len(result['probas']))
                if min_len > 0:
                    print(f"\n📊 {modality} OOF文件GT评估:")
                    print(f"  对齐样本数: {min_len}")
                    print(f"  GT阳性数: {np.sum(result['gt_values'][:min_len] == 1)}")
                    print(f"  预测阳性数: {np.sum(result['preds'][:min_len] == 1)}")

                    # 保存到metadata中
                    if 'metadata' in oof_data:
                        oof_data['metadata']['gt_positive_count'] = int(np.sum(result['gt_values'][:min_len] == 1))
                        oof_data['metadata']['gt_negative_count'] = int(np.sum(result['gt_values'][:min_len] == 0))
                        oof_data['metadata']['aligned_samples'] = min_len

                    # 如果之前没有计算performance_metrics，现在计算
                    if 'performance_metrics' not in oof_data or not oof_data['performance_metrics']:
                        try:
                            from sklearn.metrics import roc_auc_score, accuracy_score
                            y_true = result['gt_values'][:min_len]
                            y_pred = result['preds'][:min_len]
                            y_probas = result['probas'][:min_len]

                            if len(np.unique(y_true)) > 1:  # 确保有正负样本
                                auc = roc_auc_score(y_true, y_probas)
                                accuracy = accuracy_score(y_true, y_pred)

                                oof_data['performance_metrics'] = {
                                    'AUC': float(auc),
                                    'Accuracy': float(accuracy),
                                    'n_samples': min_len,
                                    'n_positive': int(np.sum(y_true == 1)),
                                    'n_negative': int(np.sum(y_true == 0))
                                }
                                print(f"✅  计算OOF性能指标: AUC={auc:.3f}, Accuracy={accuracy:.3f}")
                        except Exception as e:
                            print(f"⚠️  计算OOF性能指标失败: {e}")

            except Exception as e:
                print(f"⚠️ OOF GT评估失败: {e}")

        # 添加每折的预测结果
        if 'fold_details' in result and result['fold_details']:
            for fold_idx, fold_info in enumerate(result['fold_details']):
                probas = fold_info.get('probas', [])
                if len(probas) >= n_samples:
                    df_modality[f'fold_{fold_idx + 1}_prob'] = probas[:n_samples]
                else:
                    # 如果长度不匹配，用NaN填充
                    df_modality[f'fold_{fold_idx + 1}_prob'] = np.nan

        # 保存CSV
        csv_path = os.path.join(modality_oof_dir, f"{modality}_detailed_oof_predictions.csv")
        df_modality.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ {modality} 详细OOF数据已保存到: {csv_path}")

        # 保存特征映射
        mapping_df = pd.DataFrame([
            {'model_feature': k, 'matched_feature': v}
            for k, v in result['mapping'].items()
        ])
        mapping_path = os.path.join(modality_oof_dir, f"{modality}_feature_mapping.csv")
        mapping_df.to_csv(mapping_path, index=False, encoding='utf-8-sig')
        print(f"✅ {modality} 特征映射已保存到: {mapping_path}")

        # 保存缺失特征
        if result['missing_features']:
            missing_df = pd.DataFrame({
                'missing_features': result['missing_features']
            })
            missing_path = os.path.join(modality_oof_dir, f"{modality}_missing_features.csv")
            missing_df.to_csv(missing_path, index=False, encoding='utf-8-sig')
            print(f"✅ {modality} 缺失特征已保存到: {missing_path}")

        # 保存每折预测统计
        if 'fold_details' in result and result['fold_details']:
            fold_data = []
            for fold_info in result['fold_details']:
                fold_data.append({
                    'fold_idx': fold_info['fold_idx'],
                    'mean_prob': np.mean(fold_info.get('probas', [0])),
                    'std_prob': np.std(fold_info.get('probas', [0])),
                    'model_type': fold_info.get('model_type', 'Unknown'),
                    'has_predict_proba': fold_info.get('has_predict_proba', False)
                })

            fold_df = pd.DataFrame(fold_data)
            fold_path = os.path.join(modality_oof_dir, f"{modality}_fold_statistics.csv")
            fold_df.to_csv(fold_path, index=False, encoding='utf-8-sig')
            print(f"✅ {modality} 折统计已保存到: {fold_path}")

        # 保存性能指标
        if result.get('performance_metrics'):
            perf_df = pd.DataFrame({
                'Metric': list(result['performance_metrics'].keys()),
                'Value': list(result['performance_metrics'].values())
            })
            perf_path = os.path.join(modality_oof_dir, f"{modality}_performance_metrics.csv")
            perf_df.to_csv(perf_path, index=False, encoding='utf-8-sig')
            print(f"✅ {modality} 性能指标已保存到: {perf_path}")

    except Exception as e:
        print(f"⚠️ 保存{modality} CSV数据失败: {e}")

    return modality_oof_dir


def save_combined_oof_data(results, output_dir, ensemble_probas=None, ensemble_preds=None, ensemble_gt=None):
    """
    保存组合的OOF数据（所有模态）
    """
    print(f"\n💾 正在保存组合OOF数据...")

    combined_oof_data = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'results': {},
        'ensemble_results': None,
        'summary': {}
    }

    # 保存各模态的详细结果
    for modality, result in results.items():
        # 获取预期的特征数量
        expected_features = result.get('model_info', {}).get('expected_features', 20)

        modality_oof = {
            'pred_probabilities': result['probas'].tolist() if isinstance(result['probas'], np.ndarray) else result[
                'probas'],
            'pred_classes': result['preds'].tolist() if isinstance(result['preds'], np.ndarray) else result['preds'],
            'gt_values': result.get('gt_values', None),
            'fold_details': result.get('fold_details', []),
            'feature_mapping': result['mapping'],
            'present_features': result['present_features'],
            'missing_features': result['missing_features'],
            'performance_metrics': result.get('performance_metrics', {}),
            'model_info': result.get('model_info', {}),
            'metadata': {
                'n_samples': len(result['probas']),
                'n_features': result['X_shape'][1],
                'n_folds': len(result.get('fold_details', [])),
                'match_rate': len(result['present_features']) / expected_features if expected_features > 0 else 0,
                'match_count': len(result['present_features']),
                'missing_count': len(result['missing_features']),
                'expected_features': expected_features
            }
        }
        combined_oof_data['results'][modality] = modality_oof

        # 添加到summary
        combined_oof_data['summary'][modality] = {
            'n_samples': len(result['probas']),
            'n_folds': len(result.get('fold_details', [])),
            'mean_probability': float(np.mean(result['probas'])),
            'positive_rate': float(np.mean(result['preds'] == 1)),
            'feature_match_rate': len(result['present_features']) / expected_features if expected_features > 0 else 0,
            'feature_match_count': len(result['present_features']),
            'expected_features': expected_features,
            'performance_metrics': result.get('performance_metrics', {})
        }

    # 保存集成结果
    if ensemble_probas is not None and ensemble_preds is not None:
        ensemble_performance = None
        if ensemble_gt is not None:
            try:
                min_len = min(len(ensemble_gt), len(ensemble_probas))
                y_true = ensemble_gt[:min_len]
                y_pred = ensemble_preds[:min_len]
                y_probas = ensemble_probas[:min_len]

                auc = roc_auc_score(y_true, y_probas)
                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)

                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                else:
                    sensitivity = specificity = ppv = npv = 0

                ensemble_performance = {
                    'AUC': auc,
                    'Accuracy': accuracy,
                    'F1_score': f1,
                    'Sensitivity': sensitivity,
                    'Specificity': specificity,
                    'PPV': ppv,
                    'NPV': npv,
                    'Confusion_matrix': cm.tolist(),
                    'n_samples': min_len,
                    'n_positive': int(np.sum(y_true == 1)),
                    'n_negative': int(np.sum(y_true == 0))
                }
            except Exception as e:
                print(f"⚠️ 计算集成性能指标失败: {e}")

        combined_oof_data['ensemble_results'] = {
            'probabilities': ensemble_probas.tolist() if isinstance(ensemble_probas, np.ndarray) else ensemble_probas,
            'classes': ensemble_preds.tolist() if isinstance(ensemble_preds, np.ndarray) else ensemble_preds,
            'gt_values': ensemble_gt.tolist() if ensemble_gt is not None else None,
            'n_samples': len(ensemble_probas),
            'positive_rate': float(np.mean(ensemble_preds == 1)),
            'mean_probability': float(np.mean(ensemble_probas)),
            'performance_metrics': ensemble_performance
        }

    # 保存到文件
    combined_oof_path = os.path.join(output_dir, "combined_oof_predictions.pkl")
    joblib.dump(combined_oof_data, combined_oof_path)
    print(f"✅ 组合OOF数据已保存到: {combined_oof_path}")

    # 同时保存为JSON格式
    try:
        json_data = {
            'timestamp': combined_oof_data['timestamp'],
            'summary': combined_oof_data['summary'],
            'ensemble_summary': {
                'n_samples': len(ensemble_probas) if ensemble_probas is not None else 0,
                'positive_rate': float(np.mean(ensemble_preds == 1)) if ensemble_preds is not None else 0,
                'mean_probability': float(np.mean(ensemble_probas)) if ensemble_probas is not None else 0,
                'performance_metrics': ensemble_performance
            } if ensemble_probas is not None else None
        }

        json_path = os.path.join(output_dir, "combined_oof_summary.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ 组合OOF摘要已保存到: {json_path}")

    except Exception as e:
        print(f"⚠️ 保存组合JSON摘要失败: {e}")

    return combined_oof_path


def create_detailed_combined_csv(results, output_dir, ensemble_probas=None, ensemble_preds=None, ensemble_gt=None):
    """
    创建详细的组合CSV文件（所有模态）
    """
    print(f"\n📊 正在生成详细组合OOF数据...")

    all_dfs = []

    for modality, result in results.items():
        n_samples = len(result['probas'])

        df_modality = pd.DataFrame({
            'patient_id': [f"{modality}_{i + 1:03d}" for i in range(n_samples)],
            'modality': modality,
            'pred_prob': result['probas'],
            'pred_class': result['preds'],
            'pred_label': ['Non-csPCa' if c == 0 else 'PI-RADS ≥4' for c in result['preds']]
        })

        # 如果有GT值，添加GT列
        if result.get('gt_values') is not None:
            gt_min_len = min(n_samples, len(result['gt_values']))
            df_modality['gt_class'] = np.nan
            df_modality['gt_label'] = ''
            df_modality.loc[:gt_min_len - 1, 'gt_class'] = result['gt_values'][:gt_min_len]
            df_modality.loc[:gt_min_len - 1, 'gt_label'] = [
                'Non-csPCa' if c == 0 else 'PI-RADS ≥4'
                for c in result['gt_values'][:gt_min_len]
            ]

        all_dfs.append(df_modality)

    # 合并所有模态的数据
    if all_dfs:
        detailed_df = pd.concat(all_dfs, ignore_index=True)

        # 添加集成结果（如果存在）
        if ensemble_probas is not None and ensemble_preds is not None:
            ensemble_df = pd.DataFrame({
                'patient_id': [f"INT_{i + 1:03d}" for i in range(len(ensemble_probas))],
                'modality': 'Ensemble',
                'pred_prob': ensemble_probas,
                'pred_class': ensemble_preds,
                'pred_label': ['Non-csPCa' if c == 0 else 'PI-RADS ≥4' for c in ensemble_preds]
            })

            # 添加集成GT（使用第一个模态的GT）
            if ensemble_gt is not None:
                gt_min_len = min(len(ensemble_probas), len(ensemble_gt))
                ensemble_df['gt_class'] = np.nan
                ensemble_df['gt_label'] = ''
                ensemble_df.loc[:gt_min_len - 1, 'gt_class'] = ensemble_gt[:gt_min_len]
                ensemble_df.loc[:gt_min_len - 1, 'gt_label'] = [
                    'Non-csPCa' if c == 0 else 'PI-RADS ≥4'
                    for c in ensemble_gt[:gt_min_len]
                ]

            detailed_df = pd.concat([detailed_df, ensemble_df], ignore_index=True)

        csv_path = os.path.join(output_dir, "combined_detailed_predictions.csv")
        detailed_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ 详细组合数据已保存到: {csv_path}")

        return detailed_df

    return None


def create_performance_summary_csv(results, ensemble_probas=None, ensemble_preds=None, ensemble_gt=None,
                                   output_dir=None):
    """创建性能指标汇总CSV"""
    print(f"\n📊 正在生成性能指标汇总...")

    performance_data = []

    for modality, result in results.items():
        perf_metrics = result.get('performance_metrics')
        if perf_metrics:
            perf_data = {
                'Modality': modality,
                'AUC': perf_metrics.get('AUC', np.nan),
                'Accuracy': perf_metrics.get('Accuracy', np.nan),
                'F1_Score': perf_metrics.get('F1_score', np.nan),
                'Sensitivity': perf_metrics.get('Sensitivity', np.nan),
                'Specificity': perf_metrics.get('Specificity', np.nan),
                'PPV': perf_metrics.get('PPV', np.nan),
                'NPV': perf_metrics.get('NPV', np.nan),
                'N_Samples': perf_metrics.get('n_samples', 0),
                'N_Positive': perf_metrics.get('n_positive', 0),
                'N_Negative': perf_metrics.get('n_negative', 0),
                'Feature_Match_Rate': len(result['present_features']) / len(
                    result['model_info'].get('expected_features', 20))
            }
            performance_data.append(perf_data)

    # 添加集成性能
    if ensemble_probas is not None and ensemble_preds is not None and ensemble_gt is not None:
        try:
            min_len = min(len(ensemble_gt), len(ensemble_probas))
            y_true = ensemble_gt[:min_len]
            y_pred = ensemble_preds[:min_len]
            y_probas = ensemble_probas[:min_len]

            auc = roc_auc_score(y_true, y_probas)
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred)

            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            else:
                sensitivity = specificity = ppv = npv = 0

            perf_data = {
                'Modality': 'Ensemble',
                'AUC': auc,
                'Accuracy': accuracy,
                'F1_Score': f1,
                'Sensitivity': sensitivity,
                'Specificity': specificity,
                'PPV': ppv,
                'NPV': npv,
                'N_Samples': min_len,
                'N_Positive': int(np.sum(y_true == 1)),
                'N_Negative': int(np.sum(y_true == 0)),
                'Feature_Match_Rate': np.mean(
                    [len(r['present_features']) / len(r['model_info'].get('expected_features', 20)) for r in
                     results.values()])
            }
            performance_data.append(perf_data)
        except Exception as e:
            print(f"⚠️ 生成集成性能汇总失败: {e}")

    if performance_data:
        perf_df = pd.DataFrame(performance_data)

        if output_dir:
            perf_path = os.path.join(output_dir, "performance_summary.csv")
            perf_df.to_csv(perf_path, index=False, encoding='utf-8-sig')
            print(f"✅ 性能指标汇总保存到: {perf_path}")

        return perf_df

    return None


def main():
    # ==============================
    # 1. 配置
    # ==============================
    MODEL_DIRS = {
        'T2': r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\4rd\T2_only",
        'DCE': r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\4rd\dce_only",
        'DWI': r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\4rd\dwi_only"
    }

    DATA_PATHS = {
        'T2': r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\aligned_results\T2_aligned.csv",
        'DCE': r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\aligned_results\dce_aligned.csv",
        'DWI': r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\aligned_results\dwi_aligned.csv"
    }

    OUTPUT_DIR = r"K:\PCa_2025\8-Radiomics-PCa\output_TZ\Test-Ex\predictions_new\external_test_final_fixed"
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("自适应MRI多序列预测系统 (各模态单独保存OOF数据，包含GT评估)")
    print("=" * 60)

    # ==============================
    # 2. 初始化
    # ==============================
    try:
        predictor = AdaptiveMRIPredictor(MODEL_DIRS)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    # ==============================
    # 3. 预测所有模态并单独保存OOF数据
    # ==============================
    print("\n🔮 开始多模态预测并保存OOF数据...")

    results = {}
    modality_oof_dirs = {}  # 保存各模态的OOF目录

    for modality, data_path in DATA_PATHS.items():
        result = predictor.predict_single_modality(data_path, modality)
        if result is not None:
            results[modality] = result

            # 为每个模态单独保存OOF数据
            modality_oof_dir = save_individual_oof_data(modality, result, OUTPUT_DIR)
            modality_oof_dirs[modality] = modality_oof_dir

    if not results:
        print("\n❌ 所有模态预测都失败！")
        return

    # ==============================
    # 4. 集成结果
    # ==============================
    ensemble_probas = None
    ensemble_preds = None
    ensemble_gt = None

    if len(results) > 1:
        print(f"\n{'=' * 60}")
        print("🤝 集成多模态结果")
        print(f"{'=' * 60}")

        # 找到最小样本数
        min_samples = min(len(r['probas']) for r in results.values())

        # 权重集成（使用模型AUC作为权重）
        ensemble_probas = np.zeros(min_samples)
        total_weight = 0

        for modality, result in results.items():
            probas = result['probas'][:min_samples]
            auc = predictor.models[modality]['performance'].get('AUC', 0.5)
            weight = auc

            ensemble_probas += probas * weight
            total_weight += weight

        if total_weight > 0:
            ensemble_probas = ensemble_probas / total_weight

        # 计算最终类别
        ensemble_preds = (ensemble_probas >= 0.5).astype(int)

        # 获取集成GT（使用第一个有GT的模态）
        for result in results.values():
            if result.get('gt_values') is not None:
                ensemble_gt = result['gt_values'][:min_samples]
                print(f"✅ 使用GT进行集成评估，样本数: {len(ensemble_gt)}")
                break

        # 如果有GT，计算集成性能
        if ensemble_gt is not None:
            try:
                min_len = min(len(ensemble_gt), len(ensemble_probas))
                y_true = ensemble_gt[:min_len]
                y_pred = ensemble_preds[:min_len]
                y_probas = ensemble_probas[:min_len]

                auc = roc_auc_score(y_true, y_probas)
                accuracy = accuracy_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                cm = confusion_matrix(y_true, y_pred)

                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                else:
                    sensitivity = specificity = ppv = npv = 0

                print(f"\n📈 集成性能指标:")
                print(f"  - AUC: {auc:.3f}")
                print(f"  - Accuracy: {accuracy:.3f}")
                print(f"  - F1 Score: {f1:.3f}")
                print(f"  - Sensitivity: {sensitivity:.3f}")
                print(f"  - Specificity: {specificity:.3f}")
                print(f"  - PPV: {ppv:.3f}")
                print(f"  - NPV: {npv:.3f}")
                print(f"  - 样本数: {min_len} (阳性: {int(np.sum(y_true == 1))}, 阴性: {int(np.sum(y_true == 0))})")

            except Exception as e:
                print(f"⚠️ 计算集成性能指标失败: {e}")

        # 保存集成结果
        df_ensemble = pd.DataFrame({
            'patient_id': [f"INT_{i + 1:03d}" for i in range(min_samples)],
            'ensemble_prob': ensemble_probas,
            'ensemble_class': ensemble_preds,
            'ensemble_label': ['Non-csPCa' if c == 0 else 'PI-RADS ≥4' for c in ensemble_preds],
            'confidence': np.abs(ensemble_probas - 0.5) * 2
        })

        # 如果有GT，添加到DataFrame
        if ensemble_gt is not None:
            gt_min_len = min(min_samples, len(ensemble_gt))
            df_ensemble['gt_class'] = np.nan
            df_ensemble['gt_label'] = ''
            df_ensemble.loc[:gt_min_len - 1, 'gt_class'] = ensemble_gt[:gt_min_len]
            df_ensemble.loc[:gt_min_len - 1, 'gt_label'] = [
                'Non-csPCa' if c == 0 else 'PI-RADS ≥4'
                for c in ensemble_gt[:gt_min_len]
            ]

        # 添加各模态的贡献
        for modality in results.keys():
            auc = predictor.models[modality]['performance'].get('AUC', 0.5)
            df_ensemble[f'{modality}_weight'] = auc

        ensemble_path = os.path.join(OUTPUT_DIR, "ensemble_predictions.csv")
        df_ensemble.to_csv(ensemble_path, index=False, encoding='utf-8-sig')
        print(f"✅ 集成结果保存到: {ensemble_path}")

        # 保存集成OOF数据
        if ensemble_probas is not None:
            ensemble_oof_dir = os.path.join(OUTPUT_DIR, "ensemble_oof")
            Path(ensemble_oof_dir).mkdir(parents=True, exist_ok=True)

            ensemble_performance = None
            if ensemble_gt is not None:
                try:
                    min_len = min(len(ensemble_gt), len(ensemble_probas))
                    y_true = ensemble_gt[:min_len]
                    y_pred = ensemble_preds[:min_len]
                    y_probas = ensemble_probas[:min_len]

                    auc = roc_auc_score(y_true, y_probas)
                    accuracy = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    cm = confusion_matrix(y_true, y_pred)

                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                    else:
                        sensitivity = specificity = ppv = npv = 0

                    ensemble_performance = {
                        'AUC': auc,
                        'Accuracy': accuracy,
                        'F1_score': f1,
                        'Sensitivity': sensitivity,
                        'Specificity': specificity,
                        'PPV': ppv,
                        'NPV': npv,
                        'Confusion_matrix': cm.tolist(),
                        'n_samples': min_len,
                        'n_positive': int(np.sum(y_true == 1)),
                        'n_negative': int(np.sum(y_true == 0))
                    }
                except Exception as e:
                    print(f"⚠️ 保存集成性能指标失败: {e}")

            ensemble_oof_data = {
                'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'modality': 'Ensemble',
                'probabilities': ensemble_probas.tolist(),
                'classes': ensemble_preds.tolist(),
                'gt_values': ensemble_gt.tolist() if ensemble_gt is not None else None,
                'n_samples': len(ensemble_probas),
                'positive_rate': float(np.mean(ensemble_preds == 1)),
                'mean_probability': float(np.mean(ensemble_probas)),
                'performance_metrics': ensemble_performance,
                'metadata': {
                    'modalities_used': list(results.keys()),
                    'weights': {modality: predictor.models[modality]['performance'].get('AUC', 0.5)
                                for modality in results.keys()}
                }
            }

            ensemble_pickle_path = os.path.join(ensemble_oof_dir, "ensemble_oof_predictions.pkl")
            joblib.dump(ensemble_oof_data, ensemble_pickle_path)
            print(f"✅ 集成OOF数据保存到: {ensemble_pickle_path}")

            # 保存集成详细结果
            ensemble_df = pd.DataFrame({
                'sample_id': [f"ENS_{i + 1:03d}" for i in range(len(ensemble_probas))],
                'ensemble_prob': ensemble_probas,
                'ensemble_class': ensemble_preds,
                'ensemble_label': ['Non-csPCa' if c == 0 else 'PI-RADS ≥4' for c in ensemble_preds]
            })

            if ensemble_gt is not None:
                gt_min_len = min(len(ensemble_probas), len(ensemble_gt))
                ensemble_df['gt_class'] = np.nan
                ensemble_df['gt_label'] = ''
                ensemble_df.loc[:gt_min_len - 1, 'gt_class'] = ensemble_gt[:gt_min_len]
                ensemble_df.loc[:gt_min_len - 1, 'gt_label'] = [
                    'Non-csPCa' if c == 0 else 'PI-RADS ≥4'
                    for c in ensemble_gt[:gt_min_len]
                ]

            ensemble_csv_path = os.path.join(ensemble_oof_dir, "ensemble_detailed_oof_predictions.csv")
            ensemble_df.to_csv(ensemble_csv_path, index=False, encoding='utf-8-sig')
            print(f"✅ 集成详细OOF数据保存到: {ensemble_csv_path}")

        # 统计
        pos_count = np.sum(ensemble_preds == 1)
        neg_count = np.sum(ensemble_preds == 0)

        print(f"\n📊 集成预测统计:")
        print(f"  - Total Sample: {min_samples}")
        print(f"  - Non-csPCa: {neg_count}")
        print(f"  - csPCa: {pos_count}")
        print(f"  - 平均概率: {np.mean(ensemble_probas):.3f}")
        print(f"  - 使用模态: {', '.join(results.keys())}")

    # ==============================
    # 5. 保存组合OOF数据和性能汇总
    # ==============================
    # 保存组合OOF数据
    combined_oof_path = save_combined_oof_data(results, OUTPUT_DIR, ensemble_probas, ensemble_preds, ensemble_gt)

    # 创建详细的组合CSV
    combined_df = create_detailed_combined_csv(results, OUTPUT_DIR, ensemble_probas, ensemble_preds, ensemble_gt)

    # 创建性能指标汇总
    perf_summary = create_performance_summary_csv(results, ensemble_probas, ensemble_preds, ensemble_gt, OUTPUT_DIR)

    # ==============================
    # 6. 保存各模态结果
    # ==============================
    print(f"\n{'=' * 60}")
    print("💾 保存各模态预测结果")
    print(f"{'=' * 60}")

    for modality, result in results.items():
        # 创建患者ID
        n_samples = len(result['probas'])
        patient_ids = [f"{modality}_{i + 1:03d}" for i in range(n_samples)]

        # 创建结果DataFrame
        df_result = pd.DataFrame({
            'patient_id': patient_ids,
            'pred_prob': result['probas'],
            'pred_class': result['preds'],
            'pred_label': ['Non-csPCa' if c == 0 else 'PI-RADS ≥4' for c in result['preds']],
            'confidence': np.abs(result['probas'] - 0.5) * 2
        })

        # 如果有GT，添加到DataFrame
        if result.get('gt_values') is not None:
            gt_min_len = min(n_samples, len(result['gt_values']))
            df_result['gt_class'] = np.nan
            df_result['gt_label'] = ''
            df_result.loc[:gt_min_len - 1, 'gt_class'] = result['gt_values'][:gt_min_len]
            df_result.loc[:gt_min_len - 1, 'gt_label'] = [
                'Non-csPCa' if c == 0 else 'PI-RADS ≥4'
                for c in result['gt_values'][:gt_min_len]
            ]

        # 保存预测结果
        result_path = os.path.join(OUTPUT_DIR, f"{modality}_predictions.csv")
        df_result.to_csv(result_path, index=False, encoding='utf-8-sig')
        print(f"✅ {modality}: 预测结果保存到 {result_path}")

    # ==============================
    # 7. 生成报告
    # ==============================
    print(f"\n{'=' * 60}")
    print("📋 生成最终报告")
    print(f"{'=' * 60}")

    def format_metric(value):
        """安全地格式化指标值"""
        if isinstance(value, (int, float)):
            return f"{value:.3f}"
        else:
            return str(value)

    report_content = f"""
    MRI多序列模型外部测试预测报告 (各模态单独OOF，含GT评估)
    ==========================================================
    预测时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    测试集: 外部数据
    输出目录: {OUTPUT_DIR}

    各模态OOF数据目录:
    """

    for modality, oof_dir in modality_oof_dirs.items():
        report_content += f"""
      {modality}: {oof_dir}"""

    report_content += f"""

    组合OOF数据: {combined_oof_path}

    预测总结:
    """

    for modality in DATA_PATHS.keys():
        if modality in results:
            result = results[modality]
            pos_count = np.sum(result['preds'] == 1)
            neg_count = np.sum(result['preds'] == 0)
            match_count = len(result['present_features'])
            total_features = result.get('model_info', {}).get('expected_features', 20)
            auc = predictor.models[modality]['performance'].get('AUC', 'N/A')
            n_folds = len(result.get('fold_details', []))

            # 获取性能指标
            perf_metrics = result.get('performance_metrics')
            if perf_metrics:
                test_auc = perf_metrics.get('AUC', 'N/A')
                test_acc = perf_metrics.get('Accuracy', 'N/A')
                test_sens = perf_metrics.get('Sensitivity', 'N/A')
                test_spec = perf_metrics.get('Specificity', 'N/A')
            else:
                test_auc = test_acc = test_sens = test_spec = 'N/A'

            report_content += f"""
    {modality}:
      - 训练集AUC: {format_metric(auc)}
      - 测试集AUC: {format_metric(test_auc)}
      - 测试集Accuracy: {format_metric(test_acc)}
      - 测试集Sensitivity: {format_metric(test_sens)}
      - 测试集Specificity: {format_metric(test_spec)}
      - Feature matching: {match_count}/{total_features} ({match_count / total_features * 100:.1f}%)
      - Non-csPCa: {neg_count} ({neg_count / len(result['preds']) * 100:.1f}%)
      - csPCa: {pos_count} ({pos_count / len(result['preds']) * 100:.1f}%)
      - Mean Predicted probability: {np.mean(result['probas']):.3f}
      - 交叉验证折数: {n_folds}
      - OOF数据目录: {modality_oof_dirs.get(modality, 'N/A')}
            """
        else:
            report_content += f"""
    {modality}: 预测失败
            """

    if len(results) > 1 and ensemble_probas is not None:
        min_samples = len(ensemble_probas)
        pos_count = np.sum(ensemble_preds == 1)
        neg_count = np.sum(ensemble_preds == 0)

        # 获取集成性能指标
        ensemble_perf = None
        # 检查是否有GT计算性能指标
        if ensemble_gt is not None:
            try:
                min_len = min(len(ensemble_gt), len(ensemble_probas))
                y_true = ensemble_gt[:min_len]
                y_pred = ensemble_preds[:min_len]
                y_probas = ensemble_probas[:min_len]

                if len(np.unique(y_true)) > 1:  # 确保有正负样本
                    auc = roc_auc_score(y_true, y_probas)
                    accuracy = accuracy_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    cm = confusion_matrix(y_true, y_pred)

                    if cm.shape == (2, 2):
                        tn, fp, fn, tp = cm.ravel()
                        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
                        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
                    else:
                        sensitivity = specificity = ppv = npv = 0

                    ensemble_perf = {
                        'AUC': auc,
                        'Accuracy': accuracy,
                        'F1_score': f1,
                        'Sensitivity': sensitivity,
                        'Specificity': specificity,
                        'PPV': ppv,
                        'NPV': npv
                    }
            except Exception as e:
                print(f"⚠️ 计算集成性能指标失败: {e}")

        if ensemble_perf:
            test_auc = ensemble_perf.get('AUC', 'N/A')
            test_acc = ensemble_perf.get('Accuracy', 'N/A')
            test_sens = ensemble_perf.get('Sensitivity', 'N/A')
            test_spec = ensemble_perf.get('Specificity', 'N/A')
        else:
            test_auc = test_acc = test_sens = test_spec = 'N/A'

        report_content += f"""
    集成预测:
      - 总样本: {min_samples}
      - 测试集AUC: {format_metric(test_auc)}
      - 测试集Accuracy: {format_metric(test_acc)}
      - 测试集Sensitivity: {format_metric(test_sens)}
      - 测试集Specificity: {format_metric(test_spec)}
      - Non-csPCa: {neg_count} ({neg_count / min_samples * 100:.1f}%)
      - csPCa: {pos_count} ({pos_count / min_samples * 100:.1f}%)
      - 平均概率: {np.mean(ensemble_probas):.3f}
      - 高风险样本 (概率 > 0.7): {np.sum(ensemble_probas > 0.7)}
      - 低风险样本 (概率 < 0.3): {np.sum(ensemble_probas < 0.3)}
      - 使用模态: {', '.join(results.keys())}
        """

    report_content += f"""

    OOF数据结构说明:
    ------------
    每个模态都有自己的OOF目录 ({modality}_oof/):
    1. {modality}_oof_predictions.pkl: 完整的OOF数据 (Python pickle格式)
    2. {modality}_oof_summary.json: OOF数据摘要 (JSON格式，便于查看)
    3. {modality}_detailed_oof_predictions.csv: 详细的OOF数据 (CSV格式)
    4. {modality}_feature_mapping.csv: 特征映射文件
    5. {modality}_missing_features.csv: 缺失特征列表
    6. {modality}_fold_statistics.csv: 每折预测统计
    7. {modality}_performance_metrics.csv: 性能指标文件

    组合OOF数据:
    1. combined_oof_predictions.pkl: 所有模态的组合OOF数据
    2. combined_oof_summary.json: 组合OOF数据摘要
    3. combined_detailed_predictions.csv: 所有模态的详细预测结果
    4. performance_summary.csv: 所有模态的性能指标汇总

    集成OOF数据 (如果有多模态):
    1. ensemble_oof/ensemble_oof_predictions.pkl: 集成预测OOF数据
    2. ensemble_oof/ensemble_detailed_oof_predictions.csv: 集成详细预测结果

    数据用途:
    - 模型性能评估: 基于GT计算AUC、准确率等指标
    - SHAP分析: 用于特征重要性解释
    - 模型性能验证: 用于后续模型性能评估
    - 特征重要性分析: 分析哪些特征对预测贡献最大
    - 结果可解释性研究: 理解模型预测的依据
    """

    print(report_content)

    # 保存报告
    report_path = os.path.join(OUTPUT_DIR, "final_prediction_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"✅ 报告保存到: {report_path}")

    # ==============================
    # 8. 可视化
    # ==============================
    try:
        print("\n📈 生成可视化图表...")

        # 根据是否有性能指标决定子图数量
        if any('performance_metrics' in r for r in results.values()):
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        colors = ['blue', 'green', 'red']

        # 1. 各模态预测概率分布
        ax = axes[0, 0] if 'performance_metrics' in results.values() else axes[0, 0]
        for i, (modality, result) in enumerate(results.items()):
            ax.hist(result['probas'], bins=20, alpha=0.5, color=colors[i],
                    label=modality, edgecolor='black')
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('Sample size')
        ax.set_title('Predicted probability distribution of each modality')
        ax.legend()
        ax.grid(alpha=0.3)

        # 2. 各模态预测对比
        ax = axes[0, 1] if 'performance_metrics' in results.values() else axes[0, 1]
        data_to_plot = []
        labels = []
        for modality, result in results.items():
            data_to_plot.append(result['probas'])
            labels.append(modality)

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        # 设置颜色
        for patch, color in zip(bp['boxes'], colors[:len(results)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)

        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1)
        ax.set_ylabel('Predicted probability')
        ax.set_title('Comparison of predictions across different modalities')
        ax.grid(alpha=0.3)

        # 3. 集成预测分布
        if len(results) > 1 and ensemble_probas is not None:
            ax = axes[1, 0] if 'performance_metrics' in results.values() else axes[1, 0]
            ax.hist(ensemble_probas, bins=20, alpha=0.7,
                    color='purple', edgecolor='black')
            ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2)
            ax.set_xlabel('Predicted probability')
            ax.set_ylabel('Sample size')
            ax.set_title('Integrated forecast probability distribution')
            ax.grid(alpha=0.3)

            # 4. 风险等级分布
            ax = axes[1, 1] if 'performance_metrics' in results.values() else axes[1, 1]
            risk_levels = []
            for prob in ensemble_probas:
                if prob < 0.3:
                    risk_levels.append('Low risk')
                elif prob < 0.7:
                    risk_levels.append('Medium risk')
                else:
                    risk_levels.append('High risk')

            risk_counts = pd.Series(risk_levels).value_counts()
            colors_risk = ['lightgreen', 'gold', 'lightcoral']
            risk_counts.plot(kind='pie', autopct='%1.1f%%', colors=colors_risk, ax=ax)
            ax.set_ylabel('')
            ax.set_title('Integrated prediction of risk level distribution')

        # 5. 性能指标对比（如果有GT）
        if any('performance_metrics' in r for r in results.values()):
            ax = axes[0, 2]
            perf_metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity']
            x = np.arange(len(perf_metrics))
            width = 0.2

            for i, (modality, result) in enumerate(results.items()):
                if 'performance_metrics' in result:
                    metrics = result['performance_metrics']
                    values = [metrics.get(metric, 0) for metric in perf_metrics]
                    ax.bar(x + i * width - width * (len(results) - 1) / 2, values, width,
                           label=modality, color=colors[i], alpha=0.7)

            ax.set_xlabel('Metrics')
            ax.set_ylabel('Score')
            ax.set_title('Performance metrics comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(perf_metrics)
            ax.legend()
            ax.grid(alpha=0.3, axis='y')
            ax.set_ylim(0, 1.1)

        # 6. 混淆矩阵可视化（如果有集成结果和GT）
        if (len(results) > 1 and ensemble_probas is not None and
                ensemble_gt is not None and 'performance_metrics' in results[list(results.keys())[0]]):
            ax = axes[1, 2]

            # 获取第一个模态的性能指标
            perf_metrics = results[list(results.keys())[0]]['performance_metrics']
            if 'Confusion_matrix' in perf_metrics:
                cm = np.array(perf_metrics['Confusion_matrix'])

                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.figure.colorbar(im, ax=ax)

                # 添加文本标签
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, format(cm[i, j], 'd'),
                                ha="center", va="center",
                                color="white" if cm[i, j] > thresh else "black")

                ax.set_xlabel('Predicted label')
                ax.set_ylabel('True label')
                ax.set_title('Confusion Matrix')
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Non-csPCa', 'csPCa'])
                ax.set_yticklabels(['Non-csPCa', 'csPCa'])

        plt.tight_layout()
        plot_path = os.path.join(OUTPUT_DIR, "prediction_analysis.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"✅ 可视化图表保存到: {plot_path}")

    except Exception as e:
        print(f"⚠️ 可视化失败: {e}")

    # ==============================
    # 9. 完成
    # ==============================
    print(f"\n{'=' * 60}")
    print("🎉 预测任务完成！")
    print(f"{'=' * 60}")

    print(f"\n📁 输出目录: {OUTPUT_DIR}")
    print(f"📄 生成的文件:")

    for modality in results.keys():
        print(f"  - {modality}_predictions.csv")
        print(f"  - {modality}_oof/ 目录 (包含该模态的所有OOF数据和性能指标)")

    if len(results) > 1 and ensemble_probas is not None:
        print(f"  - ensemble_predictions.csv")
        print(f"  - ensemble_oof/ 目录 (包含集成OOF数据)")

    print(f"  - combined_oof_predictions.pkl (组合OOF数据)")
    print(f"  - combined_detailed_predictions.csv (组合详细预测)")
    print(f"  - performance_summary.csv (性能指标汇总)")
    print(f"  - final_prediction_report.txt")
    print(f"  - prediction_analysis.png")

    print(f"\n📊 各模态OOF数据包含:")
    for modality in results.keys():
        print(f"  {modality}:")
        print(f"    - {modality}_oof_predictions.pkl (主要OOF数据)")
        print(f"    - {modality}_oof_summary.json (OOF摘要)")
        print(f"    - {modality}_detailed_oof_predictions.csv (详细OOF数据)")
        print(f"    - {modality}_feature_mapping.csv (特征映射)")
        print(f"    - {modality}_missing_features.csv (缺失特征)")
        print(f"    - {modality}_fold_statistics.csv (折统计)")
        print(f"    - {modality}_performance_metrics.csv (性能指标)")

    print(f"\n📊 临床提示:")
    for modality in results.keys():
        result = results[modality]
        high_risk = np.sum(result['probas'] > 0.7)
        if high_risk > 0:
            print(f"  {modality}: {high_risk}个高风险样本 (概率 > 0.7)")


if __name__ == "__main__":
    main()