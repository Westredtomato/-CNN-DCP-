# 导入必要的库
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))  # 确保可以找到haze_removal
import torch
from PIL import Image, ImageTk  # PIL用于图像处理，ImageTk用于Tkinter图像显示
from torchvision import transforms  # 提供图像预处理功能
from haze_removal.haze_removal import remove_haze  # 从自定义模块导入去雾函数
from classifier.model import CNNModel  # 导入自定义的CNN模型类
import os
import tkinter as tk  # 用于构建GUI界面
from tkinter import filedialog, messagebox  # 文件对话框和消息提示框
#基于PyTorch和Tkinter的图形用户界面(GUI)应用程序，主要用于检测图像是否有雾，并对有雾图像进行去雾处理。

# 加载预训练模型
model = CNNModel()  # 初始化CNN模型实例
model.load_state_dict(torch.load('model.pth'))  # 加载预训练权重文件
model.eval()  # 将模型设置为评估模式（关闭dropout等训练专用层）


# 图像分类预测函数
def predict(image_path):
    """
    判断输入图像是否有雾
    参数:
        image_path: 图像文件路径
    返回:
        "Hazy"(有雾)或"Clear"(无雾)
    """
    image = Image.open(image_path)  # 使用PIL打开图像文件

    # 定义图像预处理流程
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图像大小为128x128像素
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化处理
    ])

    # 应用预处理并添加batch维度（模型输入要求）
    image = transform(image).unsqueeze(0)
    output = model(image)  # 使用模型进行预测
    prediction = (output.squeeze() > 0.5).float()  # 使用0.5作为分类阈值
    return "Hazy" if prediction.item() == 1 else "Clear"  # 返回分类结果


# 图像去雾处理函数
def process_haze_removal(image_path):
    """
    对有雾图像执行去雾处理
    参数:
        image_path: 图像文件路径
    返回:
        去雾后的PIL图像对象
    """
    print("正在执行去雾处理...")
    output_image = remove_haze(image_path)  # 调用去雾算法

    # 将numpy数组输出转换为PIL图像格式
    output_image = Image.fromarray(output_image)
    return output_image


# 主图像处理函数
def process_image(image_path):
    """
    图像处理主流程：先分类，再根据需要去雾
    参数:
        image_path: 图像文件路径
    返回:
        (处理后的图像对象, 状态消息)
    """
    print("开始图像分类...")
    prediction = predict(image_path)  # 先判断图像是否有雾

    if prediction == "Hazy":
        print("检测到有雾图像，执行去雾...")
        output_image = process_haze_removal(image_path)  # 执行去雾处理
        return output_image, "去雾处理完成"
    else:
        print("图像无雾，无需处理")
        return Image.open(image_path), "图像无雾"  # 直接返回原图


# GUI文件选择函数
def open_file():
    """
    打开文件对话框并处理选中的图像
    """
    # 设置支持的文件类型
    file_path = filedialog.askopenfilename(
        title="选择图像文件",
        filetypes=[("图像文件", "*.png;*.jpg;*.jpeg")]
    )

    if file_path:
        # 处理图像并获取结果
        output_image, message = process_image(file_path)

        # 显示处理结果消息
        messagebox.showinfo("处理结果", message)

        # 准备显示图像
        img = output_image
        img.thumbnail((300, 300))  # 生成缩略图（保持宽高比）
        img = ImageTk.PhotoImage(img)  # 转换为Tkinter兼容格式

        # 更新GUI显示
        result_label.config(image=img)
        result_label.image = img  # 保持引用防止被垃圾回收

        # 启用下载按钮并绑定当前图像
        download_button.config(
            state="normal",
            command=lambda: save_image(output_image)
        )


# 图像保存函数
def save_image(image):
    """
    保存处理后的图像到指定路径
    参数:
        image: 要保存的PIL图像对象
    """
    # 打开保存文件对话框
    output_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG文件", "*.png"), ("所有文件", "*.*")],
        title="保存处理结果"
    )

    if output_path:
        image.save(output_path)  # 保存图像文件
        messagebox.showinfo("保存成功", f"文件已保存到: {output_path}")


# ===== GUI界面构建 =====
root = tk.Tk()  # 创建主窗口
root.title("图像去雾处理系统")  # 设置窗口标题
root.geometry("500x600")  # 设置初始窗口大小
root.resizable(False, False)  # 禁止调整窗口大小
root.config(bg="#f0f0f0")  # 设置背景颜色

# 界面样式配置
button_style = {
    "font": ("Arial", 12),
    "width": 15,
    "height": 2,
    "bg": "#4CAF50",  # 按钮背景色
    "fg": "white",  # 文字颜色
    "relief": "raised"
}

# 主标题标签
title_label = tk.Label(
    root,
    text="图像去雾处理系统",
    font=("Arial", 16, "bold"),
    bg="#f0f0f0",
    fg="#333333"
)
title_label.pack(pady=20)

# 选择文件按钮
select_button = tk.Button(
    root,
    text="选择图像",
    command=open_file,
    **button_style
)
select_button.pack(pady=10)

# 下载按钮（初始状态为禁用）
download_button = tk.Button(
    root,
    text="保存图像",
    state="disabled",
    **button_style
)
download_button.pack(pady=10)

# 图像显示区域
result_label = tk.Label(
    root,
    bg="#ffffff",  # 白色背景
    relief="sunken",  # 凹陷边框效果
    bd=2  # 边框宽度
)
result_label.pack(pady=20, padx=20, fill="both", expand=True)

# 状态栏标签
status_label = tk.Label(
    root,
    text="就绪 | 请选择图像文件",
    font=("Arial", 10),
    bg="#e0e0e0",
    fg="#555555",
    anchor="w"
)
status_label.pack(fill="x", pady=10)

# 启动主事件循环
root.mainloop()