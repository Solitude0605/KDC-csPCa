# extract_radiomics_from_bbox.py
"""
从 bounding box CSV 提取 T2WI 放射组学特征（无需预生成 mask）
输入：
  - image_dir: .nii.gz 图像文件夹
  - bbox_csv: 包含 patient_id, x_min, y_min, x_max, y_max 的 CSV
  - output_csv: 输出特征 CSV
  - 从nii中提取影像组学特征
"""

import sys
from pathlib import Path
import pandas as pd
import SimpleITK as sitk
import numpy as np
from radiomics import featureextractor
from tqdm import tqdm
import warnings
import yaml

warnings.filterwarnings("ignore")


def create_mask_from_bbox(image, x_min, y_min, x_max, y_max, z_start=None, z_end=None):
    """
    根据 2D bounding box 创建 3D 掩膜（在整个 Z 轴或指定范围内）
    假设 bbox 坐标是 (x, y) = (列, 行)，与 ITK/LPS 坐标一致
    """
    size = image.GetSize()
    width, height, depth = size[0], size[1], size[2]

    # 边界检查
    x_min = max(0, int(x_min))
    y_min = max(0, int(y_min))
    x_max = min(width, int(x_max))
    y_max = min(height, int(y_max))

    if z_start is None:
        z_start = 0
    if z_end is None:
        z_end = depth

    z_start = max(0, int(z_start))
    z_end = min(depth, int(z_end))

    # 创建全零掩膜
    mask_array = np.zeros((depth, height, width), dtype=np.uint8)

    # 在指定 Z 范围内填充矩形
    mask_array[z_start:z_end, y_min:y_max, x_min:x_max] = 1

    # 转换为 SimpleITK 图像（注意：ITK 使用 [x,y,z]，numpy 是 [z,y,x]）
    mask_sitk = sitk.GetImageFromArray(mask_array)
    mask_sitk.CopyInformation(image)  # 复用原图的 spacing, origin, direction
    return mask_sitk


def main():
    # 🔧 硬编码路径 —— 直接修改这里 👇
    IMAGE_DIR = r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\nii\t2"
    BBOX_CSV = r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\radiomics_csv\bboxes_class9th_patient.csv"
    OUTPUT_CSV = r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\radiomics_csv\radiomics_features_from_bbox_t2.csv"
    YAML_CONFIG = r"K:\PCa_2025\8-Radiomics-PCa\ML_TZ\extract_radiomics\t2wi_custom.yaml"

    image_dir = Path(IMAGE_DIR)
    bbox_csv = Path(BBOX_CSV)
    output_csv = Path(OUTPUT_CSV)
    yaml_config = Path(YAML_CONFIG)

    # 加载 bounding box 数据
    df_bbox = pd.read_csv(bbox_csv)
    print(f"📊 加载 {len(df_bbox)} 个 bounding boxes")

    # 初始化 extractor
    if not yaml_config.exists():
        print(f"❌ YAML 配置文件不存在: {yaml_config}")
        sys.exit(1)

    with open(yaml_config, encoding='utf-8') as config_file:
        config_dict = yaml.safe_load(config_file)

    extractor = featureextractor.RadiomicsFeatureExtractor(config_dict)
    print(f"✅ 使用配置: {yaml_config.name}")

    results = []
    for _, row in tqdm(df_bbox.iterrows(), total=len(df_bbox), desc="提取特征"):
        patient_id = row['patient_id_2']
        img_path = image_dir / f"{patient_id}.nii.gz"

        if not img_path.exists():
            print(f"⚠️ 图像不存在: {img_path}")
            continue

        try:
            image = sitk.ReadImage(str(img_path))
        except Exception as e:
            print(f"❌ 无法加载图像 {patient_id}: {e}")
            continue

        # 获取 bbox 坐标
        x_min, y_min = row['x_min'], row['y_min']
        x_max, y_max = row['x_max'], row['y_max']

        # 创建 3D 掩膜（默认在整个 Z 轴范围）
        mask = create_mask_from_bbox(image, x_min, y_min, x_max, y_max)

        # 可选：只在有病灶的 slice 范围内提取（如果你有 slice 信息）
        # 例如：mask = create_mask_from_bbox(image, x_min, y_min, x_max, y_max, z_start=10, z_end=15)

        try:
            feature_vector = extractor.execute(image, mask, label=1)
            feature_vector['patient_id'] = patient_id
            results.append(feature_vector)
        except Exception as e:
            print(f"❌ 特征提取失败 {patient_id}: {e}")

    if not results:
        print("❌ 未成功提取任何特征！")
        sys.exit(1)

    # 保存结果
    df_out = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"\n✅ 成功提取 {len(df_out)} 个样本的特征，已保存至:\n{output_csv}")


if __name__ == "__main__":
    main()