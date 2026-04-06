# ======================================================================================
# 提取每位患者的肿瘤最长直径（mm）和体积（mm³） - 修复版
# ======================================================================================

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.measure import regionprops
from scipy.ndimage import binary_fill_holes, label
from tqdm import tqdm
import pydicom
import re

# 设置路径
mask_root_dir = Path(
    r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\ROI_mask\t2-mask-pre")
dicom_root_dir = Path(r"K:\PCa_2025\5-Chen_Classify\External_Xinqiao\Raw\t2")
output_csv = r"K:\PCa_2025\8-Radiomics-PCa\data\external_test\radiomics_csv\tu_longest_diameter_and_volume_per_patient_mm.csv"

TU_LABEL = 9  # 肿瘤标签


def debug_print(msg):
    """调试打印"""
    print(f"[DEBUG] {msg}")


def natural_sort_key(s):
    """自然排序"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]


def find_mask_files(folder_path):
    """查找文件夹中的所有掩码文件（支持多种命名）"""
    mask_files = []

    # 尝试多种可能的文件名模式
    patterns = [
        "image_*.png",  # image_001.png
        "*.png",  # 001.png, slice_001.png 等
        "*.jpg",  # 如果保存为jpg
        "*mask*.png",  # mask_001.png
    ]

    for pattern in patterns:
        found = list(folder_path.glob(pattern))
        if found:
            mask_files.extend(found)
            break

    # 去重
    mask_files = list(set(mask_files))

    if mask_files:
        # 自然排序
        mask_files.sort(key=natural_sort_key)
        debug_print(f"Found {len(mask_files)} mask files in {folder_path.name}")
    else:
        debug_print(f"No mask files found in {folder_path.name}")

    return mask_files


def preprocess_mask(mask):
    """预处理掩码图像"""
    if mask is None or mask.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    # 提取肿瘤区域
    tu_region = (mask == TU_LABEL).astype(np.uint8)

    # 检查是否有肿瘤像素
    tumor_pixels = np.sum(tu_region)
    if tumor_pixels == 0:
        debug_print(f"No tumor pixels (label={TU_LABEL}) found")
        return tu_region

    debug_print(f"Found {tumor_pixels} tumor pixels")

    try:
        # 填充孔洞
        tu_filled = binary_fill_holes(tu_region).astype(np.uint8)

        # 连接成分分析
        labels, num_features = label(tu_filled)

        if num_features == 0:
            debug_print("No connected components after filling")
            return tu_region

        # 找到最大的连通区域
        sizes = [(labels == i).sum() for i in range(1, num_features + 1)]
        max_label = np.argmax(sizes) + 1
        tu_cleaned = (labels == max_label).astype(np.uint8)

        # 形态学闭运算
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tu_closed = cv2.morphologyEx(tu_cleaned, cv2.MORPH_CLOSE, kernel)

        debug_print(f"Connected components: {num_features}, largest size: {sizes[max_label - 1]}")

        return tu_closed

    except Exception as e:
        debug_print(f"Preprocessing error: {e}")
        return tu_region


def get_dicom_spacing_and_thickness(dicom_folder):
    """从DICOM文件获取像素间距和切片厚度"""
    pixel_spacing = None
    slice_thickness = None
    dicom_files_found = False

    # 尝试多种DICOM文件扩展名
    extensions = ["*.dcm", "*.DCM", "*.ima", "*.IMA"]

    for ext in extensions:
        dcm_files = list(dicom_folder.glob(ext))
        if dcm_files:
            dicom_files_found = True
            debug_print(f"Found {len(dcm_files)} DICOM files in {dicom_folder.name}")

            # 只读取第一个有效的DICOM文件
            for dcm_file in dcm_files[:1]:  # 只检查第一个文件
                try:
                    ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)

                    # 获取像素间距
                    if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing:
                        pixel_spacing = float(ds.PixelSpacing[0])
                        debug_print(f"Pixel spacing from DICOM: {pixel_spacing} mm")

                    # 获取切片厚度
                    if hasattr(ds, 'SliceThickness'):
                        slice_thickness = float(ds.SliceThickness)
                        debug_print(f"Slice thickness from DICOM: {slice_thickness} mm")

                    # 获取切片间距（如果有）
                    slice_spacing = None
                    if hasattr(ds, 'SpacingBetweenSlices'):
                        slice_spacing = float(ds.SpacingBetweenSlices)
                        debug_print(f"Slice spacing from DICOM: {slice_spacing} mm")

                    # 如果切片厚度为0，使用切片间距
                    if slice_thickness == 0 and slice_spacing is not None:
                        slice_thickness = slice_spacing
                        debug_print(f"Using slice spacing as thickness: {slice_thickness} mm")

                    break

                except Exception as e:
                    debug_print(f"Error reading DICOM {dcm_file.name}: {e}")
                    continue

    if not dicom_files_found:
        debug_print(f"No DICOM files found in {dicom_folder}")

    return pixel_spacing, slice_thickness


def calculate_tumor_metrics(patient_folder, dicom_folder, sample_id):
    """计算一个患者的肿瘤指标"""
    # 1. 获取掩码文件
    mask_files = find_mask_files(patient_folder)

    if not mask_files:
        debug_print(f"No mask files found for {sample_id}")
        # 返回默认结果和状态
        result = {
            "sample_id": sample_id,
            "max_tu_diameter_mm": 0.0,
            "tumor_volume_mm3": 0.0,
            "tumor_volume_ml": 0.0,
            "max_tu_diameter_px": 0.0,
            "total_tumor_area_px": 0.0,
            "slices_with_tumor": 0,
            "total_slices": 0,
            "slice_with_max_diameter": "",
            "pixel_spacing_mm": 0.0,
            "slice_thickness_mm": 0.0,
            "status": "No mask files found"
        }
        return result, "No mask files found"

    # 2. 获取DICOM信息
    pixel_spacing, slice_thickness = get_dicom_spacing_and_thickness(dicom_folder)

    # 设置默认值（如果需要）
    if pixel_spacing is None or pixel_spacing <= 0:
        pixel_spacing = 0.625  # 默认像素间距
        debug_print(f"Using default pixel spacing: {pixel_spacing} mm")

    if slice_thickness is None or slice_thickness <= 0:
        slice_thickness = 3.0  # 默认切片厚度
        debug_print(f"Using default slice thickness: {slice_thickness} mm")

    # 3. 处理每个切片
    max_diameter_px = 0.0
    best_slice = ""
    total_tumor_area_px = 0.0
    slices_with_tumor = 0

    for img_file in mask_files:
        try:
            # 读取掩码
            mask = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                debug_print(f"Cannot read mask: {img_file.name}")
                continue

            # 预处理
            tu_region = preprocess_mask(mask)
            tumor_area = np.sum(tu_region)

            if tumor_area > 0:
                slices_with_tumor += 1

                try:
                    # 获取区域属性
                    props = regionprops(tu_region)
                    if props:
                        # 计算最大直径
                        diameter_px = props[0].feret_diameter_max

                        if diameter_px > max_diameter_px:
                            max_diameter_px = diameter_px
                            best_slice = img_file.stem

                        # 累加肿瘤面积
                        total_tumor_area_px += tumor_area

                        debug_print(f"Slice {img_file.name}: area={tumor_area}, diameter={diameter_px:.2f}px")
                except Exception as e:
                    debug_print(f"Error in regionprops for {img_file.name}: {e}")

        except Exception as e:
            debug_print(f"Error processing {img_file.name}: {e}")
            continue

    # 4. 计算结果
    if slices_with_tumor == 0:
        debug_print(f"No tumor slices found for {sample_id}")
        result = {
            "sample_id": sample_id,
            "max_tu_diameter_mm": 0.0,
            "tumor_volume_mm3": 0.0,
            "tumor_volume_ml": 0.0,
            "max_tu_diameter_px": 0.0,
            "total_tumor_area_px": 0.0,
            "slices_with_tumor": 0,
            "total_slices": len(mask_files),
            "slice_with_max_diameter": "",
            "pixel_spacing_mm": pixel_spacing,
            "slice_thickness_mm": slice_thickness,
            "status": "No tumor found"
        }
        return result, "No tumor found"

    # 计算物理量
    max_diameter_mm = max_diameter_px * pixel_spacing
    tumor_volume_mm3 = total_tumor_area_px * (pixel_spacing ** 2) * slice_thickness
    tumor_volume_ml = tumor_volume_mm3 / 1000.0

    result = {
        "sample_id": sample_id,
        "max_tu_diameter_mm": round(max_diameter_mm, 2),
        "tumor_volume_mm3": round(tumor_volume_mm3, 2),
        "tumor_volume_ml": round(tumor_volume_ml, 3),
        "max_tu_diameter_px": round(max_diameter_px, 2),
        "total_tumor_area_px": round(total_tumor_area_px, 2),
        "slices_with_tumor": slices_with_tumor,
        "total_slices": len(mask_files),
        "slice_with_max_diameter": best_slice,
        "pixel_spacing_mm": round(pixel_spacing, 3),
        "slice_thickness_mm": round(slice_thickness, 3),
        "status": "Success"
    }

    debug_print(f"Results for {sample_id}:")
    debug_print(f"  Max diameter: {max_diameter_mm:.2f} mm ({max_diameter_px:.2f} px)")
    debug_print(f"  Tumor volume: {tumor_volume_mm3:.2f} mm³ ({tumor_volume_ml:.3f} ml)")
    debug_print(f"  Tumor slices: {slices_with_tumor}/{len(mask_files)}")

    return result, "Success"


def main():
    """主函数"""
    # 创建输出目录
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    # 获取患者文件夹
    patient_folders = [f for f in mask_root_dir.iterdir() if f.is_dir()]

    print(f"🔍 Found {len(patient_folders)} patient folders")
    print("=" * 60)

    results = []
    failed_patients = []

    # 处理每个患者
    for patient_folder in tqdm(patient_folders, desc="Processing patients"):
        sample_id = patient_folder.name

        print(f"\n📊 Processing patient: {sample_id}")
        print("-" * 40)

        # 查找对应的DICOM文件夹
        dicom_folder = dicom_root_dir / sample_id
        if not dicom_folder.exists():
            # 尝试其他可能的命名
            dicom_folder = dicom_root_dir / sample_id.replace("_mask", "")

        if not dicom_folder.exists():
            debug_print(f"DICOM folder not found: {dicom_folder}")
            dicom_folder = dicom_root_dir / sample_id  # 仍然使用原路径

        try:
            # 计算肿瘤指标
            result, status = calculate_tumor_metrics(patient_folder, dicom_folder, sample_id)

            if result is not None:
                results.append(result)

                if result["slices_with_tumor"] == 0:
                    print(f"⚠️  No tumor found for {sample_id}")
                else:
                    print(
                        f"✅ Processed {sample_id}: {result['max_tu_diameter_mm']:.2f} mm, {result['tumor_volume_ml']:.3f} ml")
            else:
                failed_patients.append(sample_id)
                print(f"❌ Failed to process {sample_id}: {status}")

        except Exception as e:
            debug_print(f"Error processing {sample_id}: {e}")
            failed_patients.append(sample_id)
            print(f"❌ Error processing {sample_id}: {str(e)}")

    # 保存结果
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)

        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS SUMMARY")
        print("=" * 60)

        # 统计
        total_patients = len(results)
        patients_with_tumor = len([r for r in results if r["slices_with_tumor"] > 0])
        patients_without_tumor = len([r for r in results if r["slices_with_tumor"] == 0])

        print(f"📈 Total patients processed: {total_patients}")
        print(f"✅ Patients with tumor: {patients_with_tumor}")
        print(f"⚠️  Patients without tumor: {patients_without_tumor}")

        if patients_with_tumor > 0:
            # 计算统计信息
            diameters = [r["max_tu_diameter_mm"] for r in results if r["slices_with_tumor"] > 0]
            volumes = [r["tumor_volume_ml"] for r in results if r["slices_with_tumor"] > 0]

            print(f"\n📏 Tumor Diameter Statistics:")
            print(f"   Range: {min(diameters):.2f} - {max(diameters):.2f} mm")
            print(f"   Average: {np.mean(diameters):.2f} mm")
            print(f"   Median: {np.median(diameters):.2f} mm")

            print(f"\n📦 Tumor Volume Statistics:")
            print(f"   Range: {min(volumes):.3f} - {max(volumes):.3f} ml")
            print(f"   Average: {np.mean(volumes):.3f} ml")
            print(f"   Median: {np.median(volumes):.3f} ml")

            # 添加分类统计
            print(f"\n📊 Size Categories:")
            small_tumors = len([d for d in diameters if d <= 10])
            medium_tumors = len([d for d in diameters if 10 < d <= 30])
            large_tumors = len([d for d in diameters if d > 30])

            print(f"   Small (≤10 mm): {small_tumors} patients")
            print(f"   Medium (10-30 mm): {medium_tumors} patients")
            print(f"   Large (>30 mm): {large_tumors} patients")

        print(f"\n💾 Results saved to: {output_csv}")

        # 保存失败的患者列表
        if failed_patients:
            failed_file = Path(output_csv).parent / "failed_patients.txt"
            with open(failed_file, 'w') as f:
                for patient in failed_patients:
                    f.write(f"{patient}\n")
            print(f"📝 Failed patients saved to: {failed_file}")

        # 可选：保存统计摘要
        summary_file = Path(output_csv).parent / "statistics_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("Tumor Statistics Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total patients: {total_patients}\n")
            f.write(f"Patients with tumor: {patients_with_tumor}\n")
            f.write(f"Patients without tumor: {patients_without_tumor}\n")

            if patients_with_tumor > 0:
                f.write(
                    f"\nDiameter (mm): min={min(diameters):.2f}, max={max(diameters):.2f}, avg={np.mean(diameters):.2f}\n")
                f.write(f"Volume (ml): min={min(volumes):.3f}, max={max(volumes):.3f}, avg={np.mean(volumes):.3f}\n")

        print(f"📊 Statistics summary saved to: {summary_file}")

    else:
        print("❌ No results to save!")


if __name__ == "__main__":
    main()