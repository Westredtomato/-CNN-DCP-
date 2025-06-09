#单张图像雾霾预测的，它使用训练好的CNN模型来判断输入图像是否有雾霾
import torch
from PIL import Image
from torchvision import transforms
import sys

# 添加项目路径到系统路径（可能需要根据实际项目结构调整）
sys.path.append(r'C:\Users\westone\Desktop\CNN-DCP\classifier')

from model import CNNModel  # 从model.py导入自定义CNN模型

# ========== 模型加载 ==========
model = CNNModel()  # 初始化模型结构
model.load_state_dict(torch.load('model.pth'))  # 加载训练好的模型权重
model.eval()  # 设置为评估模式(关闭dropout等)


# ========== 预测函数 ==========
def predict(image_path):
    """
    对单张图像进行雾霾预测
    参数:
        image_path: 输入图像的路径
    返回:
        "Hazy"(有雾霾) 或 "Clear"(无雾霾)
    """
    # 1. 图像加载
    image = Image.open(image_path)  # 使用PIL打开图像

    # 2. 图像预处理(必须与训练时相同)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整大小到128x128
        transforms.ToTensor(),  # 转换为PyTorch张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
    ])

    # 3. 应用预处理并添加batch维度
    image = transform(image).unsqueeze(0)  # 从[C,H,W]变为[1,C,H,W]

    # 4. 模型预测
    with torch.no_grad():  # 禁用梯度计算
        output = model(image)  # 前向传播，得到预测值

    # 5. 结果解析(二分类)
    prediction = (output.squeeze() > 0.5).float()  # 使用0.5作为阈值

    # 6. 返回可读结果
    return "Hazy" if prediction.item() == 1 else "Clear"  # 1表示有雾霾，0表示无雾霾


# 示例使用（实际使用时可以通过命令行参数传入图像路径）
if __name__ == "__main__":
    image_path = "example.png"  # 替换为你的图像路径
    result = predict(image_path)
    print(f"Prediction: {result}")