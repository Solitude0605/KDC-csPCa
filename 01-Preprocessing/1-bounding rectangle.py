#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仅处理掩码路径：
1. 将所有 PNG 掩码 resize 到 512x512（NEAREST 插值）
2. 对每个患者，合并所有切片中类别 1~9 的区域，计算全局 bounding box
3. 输出坐标到 CSV
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
from tqdm import tqdm  # 用于进度条

# === 配置路径 ===
mask_root = Path(r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\ROI_mask\t2-mask-pre")
output_mask_root = Path(r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\t2-mask-pre-resized_512_masks")  # 输出掩码路径
bbox_csv = output_mask_root / r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\patient_bboxes_class9th.csv"

output_mask_root.mkdir(parents=True, exist_ok=True)

def resize_mask_to_512(input_path: Path, output_path: Path):
    """Resize mask using NEAREST to preserve label values"""
    img = Image.open(input_path)
    if img.mode == 'P':
        img = img.convert('L')
    resized = img.resize((512, 512), Image.NEAREST)
    resized.save(output_path)

def compute_global_bbox_for_patient(mask_dir: Path):
    """
    遍历患者所有掩码切片，合并 1~9 类，返回全局 (x_min, y_min, x_max, y_max)
    """
    mask_files = sorted(mask_dir.glob("*.png"))
    if not mask_files:
        return None

    all_x = []
    all_y = []

    for mask_file in mask_files:
        # 读取掩码
        mask_img = Image.open(mask_file).convert('L')
        mask_arr = np.array(mask_img)

        # 提取前景（1~9）   # 👈 提取整个全景
        # foreground = (mask_arr >= 1) & (mask_arr <= 9)

        # ✅ 只提取类别 9
        foreground = (mask_arr == 9)     # 👈 新增
        if not np.any(foreground):
            continue

        # 获取非零像素坐标
        rows, cols = np.where(foreground)
        all_y.extend(rows.tolist())
        all_x.extend(cols.tolist())

    if not all_x or not all_y:
        return None

    x_min, x_max = int(min(all_x)), int(max(all_x))
    y_min, y_max = int(min(all_y)), int(max(all_y))
    return x_min, y_min, x_max, y_max

def main():
    results = []

    # 获取所有患者目录并排序
    patients = sorted([p for p in mask_root.iterdir() if p.is_dir()])
    total_patients = len(patients)

    # 使用单个 tqdm 进度条
    with tqdm(total=total_patients, desc="Processing patients") as pbar:
        for patient_dir in patients:
            # print(f"\nProcessing patient: {patient_dir.name}")
            out_patient_dir = output_mask_root / patient_dir.name
            out_patient_dir.mkdir(exist_ok=True)

            # Step 1: Resize all masks to 512x512
            for mask_file in patient_dir.glob("*.png"):
                out_file = out_patient_dir / mask_file.name
                resize_mask_to_512(mask_file, out_file)

            # Step 2: Compute global bounding box
            bbox = compute_global_bbox_for_patient(out_patient_dir)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                results.append({
                    'patient_id': patient_dir.name,
                    'class_used': 9,  # 👈 新增
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max,
                    'width': x_max - x_min,
                    'height': y_max - y_min
                })
            else:
                print(f"  ⚠️ No foreground (1-9) found in {patient_dir.name}")
                results.append({
                    'patient_id': patient_dir.name,
                    'x_min': -1, 'y_min': -1, 'x_max': -1, 'y_max': -1,
                    'width': 0, 'height': 0
                })

            # 更新进度条
            pbar.update(1)

    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(bbox_csv, index=False)
    print(f"\n✅ Done! Bounding boxes saved to:\n{bbox_csv}")
    print(f"Resized masks saved to:\n{output_mask_root}")

if __name__ == "__main__":
    main()