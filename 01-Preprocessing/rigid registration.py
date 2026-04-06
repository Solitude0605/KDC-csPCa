import os
import pydicom
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


def dicom_to_sitk(dicom_path):
    """读取单张 DICOM 为 SimpleITK Image"""
    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    image = sitk.GetImageFromArray(pixel_array)

    # 设置物理信息（如果存在）
    if hasattr(ds, 'PixelSpacing') and ds.PixelSpacing:
        spacing = [float(ds.PixelSpacing[1]), float(ds.PixelSpacing[0])]  # (row, col) → (y, x)
        image.SetSpacing(spacing)
    else:
        image.SetSpacing([1.0, 1.0])

    if hasattr(ds, 'ImagePositionPatient') and ds.ImagePositionPatient:
        origin = [float(ds.ImagePositionPatient[0]), float(ds.ImagePositionPatient[1])]
        image.SetOrigin(origin)
    else:
        image.SetOrigin([0.0, 0.0])

    return image, ds


def resample_to_512x512(image):
    """
    将 2D 图像重采样为 512x512 像素，保持原始物理范围（FOV）不变。
    即：新 spacing = 原 FOV / 512
    """
    original_size = image.GetSize()  # (W, H)
    original_spacing = image.GetSpacing()  # (sx, sy)

    # 计算原始物理尺寸（FOV）
    fov_x = original_size[0] * original_spacing[0]
    fov_y = original_size[1] * original_spacing[1]

    # 新尺寸固定为 512x512
    new_size = (512, 512)
    # 新 spacing = FOV / 512
    new_spacing = (fov_x / 512.0, fov_y / 512.0)

    # 构建目标图像规格
    reference_image = sitk.Image(new_size, image.GetPixelID())
    reference_image.SetOrigin(image.GetOrigin())
    reference_image.SetSpacing(new_spacing)
    reference_image.SetDirection(image.GetDirection())

    # 重采样（线性插值）
    resampled = sitk.Resample(
        image,
        reference_image,
        sitk.Transform(),  # identity
        sitk.sitkLinear,
        0.0,
        image.GetPixelID()
    )
    return resampled


def rigid_registration_no_center_init(fixed_image, moving_image):
    """
    刚性配准，不使用几何中心初始化，而是从单位变换开始 + 互信息优化
    """
    R = sitk.ImageRegistrationMethod()

    # 异模态推荐：互信息
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.01)  # 加速

    # 优化器
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    R.SetOptimizerScalesFromPhysicalShift()

    # 初始变换：单位变换（无平移、无旋转）
    initial_transform = sitk.Euler2DTransform()
    initial_transform.SetIdentity()
    R.SetInitialTransform(initial_transform, inPlace=True)

    R.SetInterpolator(sitk.sitkLinear)

    # 执行配准
    final_transform = R.Execute(fixed_image, moving_image)
    print(f"配准完成 | 度量值: {R.GetMetricValue():.4f} | 变换参数: {final_transform.GetParameters()}")

    # 应用变换
    registered = sitk.Resample(
        moving_image,
        fixed_image,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_image.GetPixelID()
    )
    return registered, final_transform


def save_registered_dicom(original_ds, registered_image, output_path):
    """保存为 DICOM，保留元数据"""
    reg_array = sitk.GetArrayFromImage(registered_image)
    # 转回原始数据类型（假设为 uint16）
    if original_ds.BitsAllocated == 16:
        reg_array = np.clip(reg_array, 0, 65535).astype(np.uint16)
    else:
        reg_array = np.clip(reg_array, 0, 255).astype(np.uint8)

    new_ds = original_ds.copy()
    new_ds.PixelData = reg_array.tobytes()
    new_ds.Rows, new_ds.Columns = reg_array.shape

    # 更新描述
    if 'SeriesDescription' in new_ds:
        new_ds.SeriesDescription += "_RegToT2_512"
    else:
        new_ds.SeriesDescription = "RegisteredToT2_512"

    new_ds.save_as(output_path)

# ... [前面的函数 dicom_to_sitk, resample_to_512x512, rigid_registration_no_center_init, save_registered_dicom 保持不变] ...

def save_single_image_png(sitk_image, output_path):
    """将 SimpleITK 图像保存为归一化的 PNG 图像"""
    arr = sitk.GetArrayFromImage(sitk_image)
    # 归一化到 [0, 1]
    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)  # 防止除零
    plt.imsave(output_path, arr_norm, cmap='gray')
    print(f"✅ 单图已保存: {output_path}")


def save_comparison_figure(t2_img, dce_before_reg, dce_after_reg, dwi_before_reg, dwi_after_reg, output_path):
    """保存组合图（无网格线，仅刻度数字）"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    def normalize_image(img_array):
        min_val = np.min(img_array)
        max_val = np.max(img_array)
        if max_val - min_val == 0:
            return np.zeros_like(img_array)
        return (img_array - min_val) / (max_val - min_val)

    def plot_image(ax, img, title):
        arr = sitk.GetArrayFromImage(img)
        norm_arr = normalize_image(arr)
        ax.imshow(norm_arr, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(np.arange(0, 512, 100))
        ax.set_yticks(np.arange(0, 512, 100))
        ax.tick_params(axis='both', which='major', labelsize=8)

    plot_image(axes[0, 0], t2_img, 'T2 (512x512)')
    plot_image(axes[0, 1], dce_before_reg, 'dce (Before Reg)')
    plot_image(axes[0, 2], dce_after_reg, 'dce (After Reg)')
    plot_image(axes[1, 0], t2_img, 'T2 (512x512)')
    plot_image(axes[1, 1], dwi_before_reg, 'DWI (Before Reg)')
    plot_image(axes[1, 2], dwi_after_reg, 'DWI (After Reg)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 组合图已保存至: {output_path}")
    plt.close(fig)


# ==============================
# 主流程
# ==============================
if __name__ == "__main__":
    T2_PATH = r"K:\PCa_2026\Article\放射组学\初稿\图片\861450_20240707\FILE75.dcm"
    dce_PATH = r"K:\PCa_2026\Article\放射组学\初稿\图片\861450_20240707\FILE222.dcm"
    DWI_PATH = r"K:\PCa_2026\Article\放射组学\初稿\图片\861450_20240707\FILE159.dcm"

    OUT_DIR = r"K:\PCa_2026\Article\放射组学\初稿\图片\rigid"
    os.makedirs(OUT_DIR, exist_ok=True)  # 确保输出目录存在

    OUT_dce = os.path.join(OUT_DIR, "dce_registered_to_T2.dcm")
    OUT_DWI = os.path.join(OUT_DIR, "DWI_registered_to_T2.dcm")

    # 单图 PNG 路径
    PNG_T2      = os.path.join(OUT_DIR, "T2_512.png")
    PNG_dce_B   = os.path.join(OUT_DIR, "dce_before_reg.png")
    PNG_dce_A   = os.path.join(OUT_DIR, "dce_after_reg.png")
    PNG_DWI_B   = os.path.join(OUT_DIR, "DWI_before_reg.png")
    PNG_DWI_A   = os.path.join(OUT_DIR, "DWI_after_reg.png")
    COMPARISON  = os.path.join(OUT_DIR, "comparison.png")

    # 1. 读取原始图像
    t2_img, t2_ds = dicom_to_sitk(T2_PATH)
    dce_img, dce_ds = dicom_to_sitk(dce_PATH)
    dwi_img, dwi_ds = dicom_to_sitk(DWI_PATH)

    # 2. 重采样为 512x512
    print("🔄 正在重采样为 512x512...")
    t2_512 = resample_to_512x512(t2_img)
    dce_512 = resample_to_512x512(dce_img)
    dwi_512 = resample_to_512x512(dwi_img)

    # 3. 刚性配准
    print("🎯 配准 dce → T2...")
    dce_reg, _ = rigid_registration_no_center_init(t2_512, dce_512)
    print("🎯 配准 DWI → T2...")
    dwi_reg, _ = rigid_registration_no_center_init(t2_512, dwi_512)

    # 4. 保存 DICOM 结果
    save_registered_dicom(dce_ds, dce_reg, OUT_dce)
    save_registered_dicom(dwi_ds, dwi_reg, OUT_DWI)

    # ✅ 新增：保存单张 PNG 图像
    print("💾 正在保存单张 PNG 图像...")
    save_single_image_png(t2_512, PNG_T2)
    save_single_image_png(dce_512, PNG_dce_B)
    save_single_image_png(dce_reg, PNG_dce_A)
    save_single_image_png(dwi_512, PNG_DWI_B)
    save_single_image_png(dwi_reg, PNG_DWI_A)

    # 5. 保存组合图
    save_comparison_figure(t2_512, dce_512, dce_reg, dwi_512, dwi_reg, COMPARISON)

    print("✅ 全部完成！")