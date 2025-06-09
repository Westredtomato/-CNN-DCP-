#基于暗通道先验(Dark Channel Prior, DCP)和引导滤波(Guided Filter)的图像去雾算法实现
import PIL.Image as Image
import skimage.io as io
import numpy as np
import time
from gf import guided_filter  # 使用显式相对导入 # 导入引导滤波实现
import matplotlib.pyplot as plt

class HazeRemoval(object):
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        """
        初始化去雾器参数
        参数:
            omega: 透射率调整参数(默认0.95)
            t0: 透射率下限阈值(默认0.1)
            radius: 暗通道计算半径(默认7)
            r: 引导滤波半径(默认20)
            eps: 引导滤波正则化参数(默认0.001)
        """
        self.omega = omega
        self.t0 = t0
        self.radius = radius
        self.r = r
        self.eps = eps

    def open_image(self, img_path):
        """
        加载并准备图像数据
        参数:
            img_path: 输入图像路径
        """
        img = Image.open(img_path)
        self.src = np.array(img).astype(np.double) / 255.  # 归一化到[0,1]
        self.rows, self.cols, _ = self.src.shape  # 图像尺寸
        # 初始化各中间结果矩阵
        self.dark = np.zeros((self.rows, self.cols), dtype=np.double)  # 暗通道
        self.Alight = np.zeros((3), dtype=np.double)  # 大气光
        self.tran = np.zeros((self.rows, self.cols), dtype=np.double)  # 透射率
        self.dst = np.zeros_like(self.src, dtype=np.double)  # 输出图像

    def get_dark_channel(self, radius=7):
        """
        计算暗通道(Dark Channel Prior)
        参数:
            radius: 局部区域半径(默认7)
        """
        print("开始计算暗通道...")
        start = time.time()
        tmp = self.src.min(axis=2)  # 取RGB三通道最小值
        # 计算每个像素的局部最小值
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - radius)
                rmax = min(i + radius, self.rows - 1)
                cmin = max(0, j - radius)
                cmax = min(j + radius, self.cols - 1)
                self.dark[i, j] = tmp[rmin:rmax + 1, cmin:cmax + 1].min()
        print("耗时:", time.time() - start)

    def get_air_light(self):
        """估计大气光(Air Light)"""
        print("开始估计大气光...")
        start = time.time()
        flat = self.dark.flatten()
        flat.sort()
        num = int(self.rows * self.cols * 0.001)  # 取前0.1%最亮的像素
        threshold = flat[-num]
        tmp = self.src[self.dark >= threshold]  # 获取最亮区域像素
        tmp.sort(axis=0)
        self.Alight = tmp[-num:, :].mean(axis=0)  # 计算大气光
        print("耗时:", time.time() - start)

    def get_transmission(self, radius=7, omega=0.95):
        """
        计算透射率(Transmission)
        参数:
            radius: 局部区域半径(默认7)
            omega: 透射率调整参数(默认0.95)
        """
        print("开始计算透射率...")
        start = time.time()
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0, i - radius)
                rmax = min(i + radius, self.rows - 1)
                cmin = max(0, j - radius)
                cmax = min(j + radius, self.cols - 1)
                # 计算归一化后的局部最小值
                pixel = (self.src[rmin:rmax + 1, cmin:cmax + 1] / self.Alight).min()
                self.tran[i, j] = 1. - omega * pixel  # 计算透射率
        print("耗时:", time.time() - start)

    def guided_filter(self, r=60, eps=0.001):
        """
        应用引导滤波优化透射率
        参数:
            r: 滤波半径(默认60)
            eps: 正则化参数(默认0.001)
        """
        print("开始引导滤波优化...")
        start = time.time()
        self.gtran = guided_filter(self.src, self.tran, r, eps)
        print("耗时:", time.time() - start)

    def recover(self, t0=0.1):
        """
        恢复无雾图像
        参数:
            t0: 透射率下限阈值(默认0.1)
        """
        print("开始去雾处理...")
        start = time.time()
        self.gtran[self.gtran < t0] = t0  # 设置透射率下限
        t = self.gtran.reshape(*self.gtran.shape, 1).repeat(3, axis=2)
        # 根据大气散射模型恢复图像
        self.dst = (self.src.astype(np.double) - self.Alight) / t + self.Alight
        self.dst *= 255  # 恢复[0,255]范围
        self.dst[self.dst > 255] = 255  # 处理溢出
        self.dst[self.dst < 0] = 0
        self.dst = self.dst.astype(np.uint8)
        print("耗时:", time.time() - start)

    def show(self):
        """保存中间结果和最终结果"""
        import cv2
        # 保存各阶段结果(用于调试和分析)
        cv2.imwrite("img/src.jpg", (self.src * 255).astype(np.uint8)[:, :, (2, 1, 0)])
        cv2.imwrite("img/dark.jpg", (self.dark * 255).astype(np.uint8))
        cv2.imwrite("img/tran.jpg", (self.tran * 255).astype(np.uint8))
        cv2.imwrite("img/gtran.jpg", (self.gtran * 255).astype(np.uint8))
        cv2.imwrite("img/dst.jpg", self.dst[:, :, (2, 1, 0)])
        io.imsave("test.jpg", self.dst)  # 保存最终结果

def remove_haze(image_path):
    """
    去雾处理接口函数
    参数:
        image_path: 输入图像路径
    返回:
        去雾后的图像(numpy数组)
    """
    hr = HazeRemoval()  # 创建去雾器实例
    hr.open_image(image_path)  # 加载图像
    hr.get_dark_channel()  # 计算暗通道
    hr.get_air_light()  # 估计大气光
    hr.get_transmission()  # 计算透射率
    hr.guided_filter()  # 引导滤波优化
    hr.recover()  # 恢复无雾图像
    return hr.dst  # 返回去雾结果