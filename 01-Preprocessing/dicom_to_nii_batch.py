# dicom_to_nii_batch.py (硬编码路径版 + 保留并显示层厚/间距)
"""
批量将 DICOM 文件夹转换为 NIfTI 格式，并保留原始 spacing / direction 等几何信息。
- 输入：每个患者一个子文件夹，内含 .dcm 文件
- 输出：每个患者一个 .nii.gz 文件（含正确 header）
"""

import sys
from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import warnings
import json
sys.stdout.flush()

warnings.filterwarnings("ignore")


def load_dicom_series(patient_dir):
    """从 DICOM 文件夹加载 3D 图像（自动选择最长的序列）"""
    try:
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(patient_dir))

        if not series_ids:
            print(f"⚠️ 无有效 DICOM 序列: {patient_dir}")
            return None

        # 获取每个序列的文件列表
        series_file_names = {}
        for sid in series_ids:
            files = reader.GetGDCMSeriesFileNames(str(patient_dir), sid)
            series_file_names[sid] = files

        # 选择文件最多的序列（通常是 T2）
        selected_series = max(series_file_names, key=lambda k: len(series_file_names[k]))
        dicom_names = series_file_names[selected_series]

        print(f"📁 {patient_dir.name}: 共 {len(series_ids)} 个序列，选择 ID={selected_series} ({len(dicom_names)} 张)")

        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # ✅ 关键：DICOM 的 spacing、origin、direction 已自动读入 image 对象
        return image

    except Exception as e:
        print(f"❌ 加载失败 {patient_dir}: {e}")
        return None


def main():
    # 🔧 在这里直接修改你的输入/输出路径 👇
    INPUT_DIR = r"K:\PCa_2025\5-Chen_Classify\676+135\test_2_676"
    OUTPUT_DIR = r"K:\PCa_2025\8-Radiomics-PCa\output\nii\t2"

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取所有患者子文件夹
    patient_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not patient_dirs:
        print(f"❌ 未找到任何患者文件夹: {input_dir}")
        sys.exit(1)

    print(f"📁 找到 {len(patient_dirs)} 个患者，开始转换...")

    success_count = 0
    failed_list = []
    metadata_log = {}  # 用于记录每个成功患者的 spacing 等信息

    for patient_dir in tqdm(patient_dirs, desc="DICOM → NIfTI"):
        patient_id = patient_dir.name
        output_path = output_dir / f"{patient_id}.nii.gz"

        if output_path.exists():
            continue

        image = load_dicom_series(patient_dir)
        if image is None:
            failed_list.append((patient_id, "加载失败"))
            continue

        try:
            # ✅ 写入 NIfTI（自动包含 spacing, origin, direction）
            sitk.WriteImage(image, str(output_path))

            # 📌 提取并记录关键元数据
            spacing = tuple(round(x, 4) for x in image.GetSpacing())      # (x, y, z) 间距，z 即层间距
            origin = tuple(round(x, 4) for x in image.GetOrigin())
            direction = tuple(round(x, 6) for x in image.GetDirection())  # 9 元组（3x3 矩阵）
            size = image.GetSize()

            metadata_log[patient_id] = {
                "spacing_mm": spacing,
                "origin_mm": origin,
                "direction_cosine": direction,
                "size_voxels": size,
                "output_path": str(output_path)
            }

            success_count += 1

        except Exception as e:
            failed_list.append((patient_id, f"写入失败: {str(e)}"))
            print(f"❌ 写入失败 {patient_id}: {e}")

    # 📝 保存元数据日志（方便后续 radiomics 配置）
    metadata_file = output_dir / "conversion_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_log, f, indent=2, ensure_ascii=False)

    # 打印失败汇总
    if failed_list:
        print(f"\n⚠️ 共 {len(failed_list)} 个患者失败，前10个示例:")
        for pid, reason in failed_list[:10]:
            print(f"  - {pid}: {reason}")
        with open(output_dir / "failed_patients.txt", "w", encoding="utf-8") as f:
            for pid, reason in failed_list:
                f.write(f"{pid}\t{reason}\n")

    print(f"\n✅ 成功转换 {success_count}/{len(patient_dirs)} 个患者")
    print(f"📁 输出目录: {output_dir}")
    print(f"📄 元数据已保存至: {metadata_file}")

    # 💡 示例：打印一个患者的 spacing（层厚 ≈ spacing[2]）
    if metadata_log:
        first_pid = next(iter(metadata_log))
        sp = metadata_log[first_pid]["spacing_mm"]
        print(f"\nℹ️ 示例 - 患者 {first_pid} 的体素间距 (mm): X={sp[0]}, Y={sp[1]}, Z={sp[2]}")
        print(f"   → 层厚/层间距 ≈ {sp[2]:.3f} mm")


if __name__ == "__main__":
    main()