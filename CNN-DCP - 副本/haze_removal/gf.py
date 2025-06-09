import numpy as np
import scipy as sp
import scipy.ndimage
#高效的引导滤波(Guided Filter)实现，主要用于图像处理中的边缘保持平滑和细节增强。

def box(img, r):
    """
    O(1)时间复杂度的高效盒式滤波(均值滤波)
    参数:
        img - 至少2维的图像数据
        r - 滤波半径
    返回:
        滤波后的图像
    """
    (rows, cols) = img.shape[:2]
    imDst = np.zeros_like(img)

    # 垂直方向累积和计算
    tile = [1] * img.ndim
    tile[0] = r
    imCum = np.cumsum(img, 0)  # 沿垂直方向累积和

    # 处理图像顶部、中部和底部的不同情况
    imDst[0:r + 1, :, ...] = imCum[r:2 * r + 1, :, ...]
    imDst[r + 1:rows - r, :, ...] = imCum[2 * r + 1:rows, :, ...] - imCum[0:rows - 2 * r - 1, :, ...]
    imDst[rows - r:rows, :, ...] = np.tile(imCum[rows - 1:rows, :, ...], tile) - imCum[rows - 2 * r - 1:rows - r - 1, :,
                                                                                 ...]

    # 水平方向累积和计算
    tile = [1] * img.ndim
    tile[1] = r
    imCum = np.cumsum(imDst, 1)  # 沿水平方向累积和

    # 处理图像左侧、中部和右侧的不同情况
    imDst[:, 0:r + 1, ...] = imCum[:, r:2 * r + 1, ...]
    imDst[:, r + 1:cols - r, ...] = imCum[:, 2 * r + 1: cols, ...] - imCum[:, 0: cols - 2 * r - 1, ...]
    imDst[:, cols - r: cols, ...] = np.tile(imCum[:, cols - 1:cols, ...], tile) - imCum[:,
                                                                                  cols - 2 * r - 1: cols - r - 1, ...]

    return imDst


def _gf_color(I, p, r, eps, s=None):
    """
    彩色引导滤波(使用RGB三通道引导图像)
    参数:
        I - 引导图像(RGB三通道)
        p - 待滤波图像(单通道)
        r - 窗口半径
        eps - 正则化参数(控制平滑程度)
        s - 下采样因子(用于快速引导滤波)
    返回:
        滤波后的图像
    """
    fullI = I
    fullP = p
    if s is not None:
        # 下采样加速处理
        I = sp.ndimage.zoom(fullI, [1 / s, 1 / s, 1], order=1)
        p = sp.ndimage.zoom(fullP, [1 / s, 1 / s], order=1)
        r = round(r / s)

    h, w = p.shape[:2]
    N = box(np.ones((h, w)), r)  # 计算每个窗口内的像素数

    # 计算各通道的均值
    mI_r = box(I[:, :, 0], r) / N
    mI_g = box(I[:, :, 1], r) / N
    mI_b = box(I[:, :, 2], r) / N

    mP = box(p, r) / N  # 待滤波图像的均值

    # 计算I*p的均值
    mIp_r = box(I[:, :, 0] * p, r) / N
    mIp_g = box(I[:, :, 1] * p, r) / N
    mIp_b = box(I[:, :, 2] * p, r) / N

    # 计算协方差
    covIp_r = mIp_r - mI_r * mP
    covIp_g = mIp_g - mI_g * mP
    covIp_b = mIp_b - mI_b * mP

    # 计算引导图像各通道间的方差和协方差
    var_I_rr = box(I[:, :, 0] * I[:, :, 0], r) / N - mI_r * mI_r
    var_I_rg = box(I[:, :, 0] * I[:, :, 1], r) / N - mI_r * mI_g
    var_I_rb = box(I[:, :, 0] * I[:, :, 2], r) / N - mI_r * mI_b

    var_I_gg = box(I[:, :, 1] * I[:, :, 1], r) / N - mI_g * mI_g
    var_I_gb = box(I[:, :, 1] * I[:, :, 2], r) / N - mI_g * mI_b

    var_I_bb = box(I[:, :, 2] * I[:, :, 2], r) / N - mI_b * mI_b

    a = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            # 构建协方差矩阵
            sig = np.array([
                [var_I_rr[i, j], var_I_rg[i, j], var_I_rb[i, j]],
                [var_I_rg[i, j], var_I_gg[i, j], var_I_gb[i, j]],
                [var_I_rb[i, j], var_I_gb[i, j], var_I_bb[i, j]]
            ])
            covIp = np.array([covIp_r[i, j], covIp_g[i, j], covIp_b[i, j]])
            # 求解线性方程组得到系数a
            a[i, j, :] = np.linalg.solve(sig + eps * np.eye(3), covIp)

    b = mP - a[:, :, 0] * mI_r - a[:, :, 1] * mI_g - a[:, :, 2] * mI_b

    # 计算a和b的均值
    meanA = box(a, r) / N[..., np.newaxis]
    meanB = box(b, r) / N

    if s is not None:
        # 上采样恢复原始尺寸
        meanA = sp.ndimage.zoom(meanA, [s, s, 1], order=1)
        meanB = sp.ndimage.zoom(meanB, [s, s], order=1)

    # 计算最终输出
    q = np.sum(meanA * fullI, axis=2) + meanB

    return q


def _gf_gray(I, p, r, eps, s=None):
    """
    灰度引导滤波(快速版本)
    参数:
        I - 引导图像(单通道)
        p - 待滤波图像(单通道)
        r - 窗口半径
        eps - 正则化参数
        s - 下采样因子(用于加速)
    返回:
        滤波后的图像
    """
    if s is not None:
        # 下采样加速
        Isub = sp.ndimage.zoom(I, 1 / s, order=1)
        Psub = sp.ndimage.zoom(p, 1 / s, order=1)
        r = round(r / s)
    else:
        Isub = I
        Psub = p

    (rows, cols) = Isub.shape
    N = box(np.ones([rows, cols]), r)

    # 计算各种统计量
    meanI = box(Isub, r) / N
    meanP = box(Psub, r) / N
    corrI = box(Isub * Isub, r) / N
    corrIp = box(Isub * Psub, r) / N
    varI = corrI - meanI * meanI
    covIp = corrIp - meanI * meanP

    # 计算线性系数
    a = covIp / (varI + eps)
    b = meanP - a * meanI

    # 计算系数均值
    meanA = box(a, r) / N
    meanB = box(b, r) / N

    if s is not None:
        # 上采样恢复原始尺寸
        meanA = sp.ndimage.zoom(meanA, s, order=1)
        meanB = sp.ndimage.zoom(meanB, s, order=1)

    # 计算最终输出
    q = meanA * I + meanB
    return q


def _gf_colorgray(I, p, r, eps, s=None):
    """
    自动选择彩色或灰度引导滤波
    根据引导图像的通道数自动选择适当的滤波方法
    """
    if I.ndim == 2 or I.shape[2] == 1:
        return _gf_gray(I, p, r, eps, s)
    elif I.ndim == 3 and I.shape[2] == 3:
        return _gf_color(I, p, r, eps, s)
    else:
        print("Invalid guide dimensions:", I.shape)


def guided_filter(I, p, r, eps, s=None):
    """
    引导滤波主接口
    对多通道输入图像进行逐通道引导滤波
    参数:
        I - 引导图像(1或3通道)
        p - 待滤波图像(n通道)
        r - 窗口半径
        eps - 正则化参数
        s - 下采样因子(用于快速滤波)
    返回:
        滤波后的图像
    """
    if p.ndim == 2:
        p3 = p[:, :, np.newaxis]

    out = np.zeros_like(p3)
    for ch in range(p3.shape[2]):
        out[:, :, ch] = _gf_colorgray(I, p3[:, :, ch], r, eps, s)
    return np.squeeze(out) if p.ndim == 2 else out


def test_gf():
    """
    引导滤波测试函数
    演示如何使用引导滤波
    """
    import imageio
    # 读取测试图像
    cat = imageio.imread('cat.bmp').astype(np.float32) / 255
    tulips = imageio.imread('tulips.bmp').astype(np.float32) / 255

    r = 8  # 窗口半径
    eps = 0.05  # 正则化参数

    # 对cat图像进行引导滤波
    cat_smoothed = guided_filter(cat, cat, r, eps)
    cat_smoothed_s4 = guided_filter(cat, cat, r, eps, s=4)  # 使用下采样加速

    # 保存结果
    imageio.imwrite('cat_smoothed.png', cat_smoothed)
    imageio.imwrite('cat_smoothed_s4.png', cat_smoothed_s4)

    # 对tulips图像进行逐通道引导滤波
    tulips_smoothed4s = np.zeros_like(tulips)
    for i in range(3):
        tulips_smoothed4s[:, :, i] = guided_filter(tulips, tulips[:, :, i], r, eps, s=4)
    imageio.imwrite('tulips_smoothed4s.png', tulips_smoothed4s)

    # 不使用下采样的版本
    tulips_smoothed = np.zeros_like(tulips)
    for i in range(3):
        tulips_smoothed[:, :, i] = guided_filter(tulips, tulips[:, :, i], r, eps)
    imageio.imwrite('tulips_smoothed.png', tulips_smoothed)