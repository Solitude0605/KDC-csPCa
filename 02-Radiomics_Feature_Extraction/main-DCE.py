#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 直接从DICOM文件中提取影像组学特征
from pathlib import Path
import pandas as pd
import pydicom  # 新增：用于读取原始尺寸
from data import dicom_processing
from tqdm import tqdm
import logging
logging.getLogger("radiomics").setLevel(logging.WARNING)

# === 配置路径 ===
csv_path = Path(r"K:\PCa_2025\8-Radiomics-PCa\data\radiomics_features_9th\ROI\patient_bboxes_class9_dwi.csv")
dcm_root = Path(r"K:\PCa_2025\5-Chen_Classify\676+135\test_D_676_800_filtered")
output_csv = Path(r"K:\PCa_2025\8-Radiomics-PCa\data\radiomics_features_9th\radiomics_csv\radiomics_features_dwi.csv")

output_csv.parent.mkdir(parents=True, exist_ok=True)


def main():
    df_bbox = pd.read_csv(csv_path)
    print(f"Loaded {len(df_bbox)} patients from CSV.")
    all_results = []

    for idx, row in tqdm(df_bbox.iterrows(), total=len(df_bbox), desc="Processing patients"):
        patient_id = row['patient_id']

        # 跳过无有效 bbox 的患者（你代码中用 -1 表示无效）
        if row['x_min'] == -1:
            tqdm.write(f"⚠️ Skipping {patient_id}: no valid ROI")
            continue

        x_min_512 = int(row['x_min'])
        y_min_512 = int(row['y_min'])
        x_max_512 = int(row['x_max'])
        y_max_512 = int(row['y_max'])

        dcm_dir = dcm_root / patient_id
        if not dcm_dir.exists():
            tqdm.write(f"❌ DICOM folder not found: {dcm_dir}")
            continue

        # === 关键：获取原始 DICOM 尺寸 ===
        dcm_files = sorted(dcm_dir.glob("*.dcm"))
        if not dcm_files:
            tqdm.write(f"❌ No DICOM files in {dcm_dir}")
            continue

        # 读第一个文件获取原始分辨率
        try:
            ds = pydicom.dcmread(dcm_files[0], force=True)  # force=True 防止部分头信息缺失报错
            orig_w = ds.Columns  # 宽度 (X)
            orig_h = ds.Rows  # 高度 (Y)
        except Exception as e:
            tqdm.write(f"❌ Failed to read DICOM header for {patient_id}: {e}")
            continue

        # === 坐标映射：512x512 → 原始 DICOM 空间 ===
        x_min_orig = int(round(x_min_512 * orig_w / 512))
        y_min_orig = int(round(y_min_512 * orig_h / 512))
        x_max_orig = int(round(x_max_512 * orig_w / 512))
        y_max_orig = int(round(y_max_512 * orig_h / 512))

        # 边界保护（防止越界）
        x_min_orig = max(0, x_min_orig)
        y_min_orig = max(0, y_min_orig)
        x_max_orig = min(orig_w, x_max_orig)
        y_max_orig = min(orig_h, y_max_orig)

        if x_min_orig >= x_max_orig or y_min_orig >= y_max_orig:
            tqdm.write(
                f"⚠️ Invalid ROI after mapping for {patient_id}: {x_min_orig, y_min_orig, x_max_orig, y_max_orig}")
            continue

        try:
            # 使用映射后的原始坐标裁剪
            image_3d = dicom_processing.load_and_crop_dcm(
                dcm_dir, x_min_orig, y_min_orig, x_max_orig, y_max_orig
            )
            tqdm.write(f"  ✓ Cropped ROI shape: {image_3d.shape}")

            features = dicom_processing.extract_radiomics_features(image_3d)
            features['patient_id'] = patient_id
            all_results.append(features)

        except Exception as e:
            tqdm.write(f"❌ Error processing {patient_id}: {e}")
            continue

    # 保存结果
    if all_results:
        df_features = pd.DataFrame(all_results)
        df_features.to_csv(output_csv, index=False)
        print(f"\n✅ Done! Radiomics features saved to: {output_csv}")
        print(f"Total extracted: {len(df_features)} patients")
    else:
        print("\n❌ No features extracted!")


if __name__ == "__main__":
    main()