import torch
import torch.nn as nn

#定义卷积神经网络(CNN)模型，用于雾霾图像分类任务
class CNNModel(nn.Module):
    def __init__(self):
        """初始化CNN模型结构"""
        super(CNNModel, self).__init__()

        # 第一卷积层: 输入3通道(RGB), 输出32通道, 3x3卷积核
        self.conv1 = nn.Conv2d(
            in_channels=3,  # 输入通道数(RGB图像)
            out_channels=32,  # 输出特征图数量
            kernel_size=3,  # 卷积核大小
            stride=1,  # 步长
            padding=1  # 边缘填充(保持尺寸)
        )

        # 第二卷积层: 32→64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # 第三卷积层: 64→128通道
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # 全连接层1: 128*16*16 → 512
        self.fc1 = nn.Linear(128 * 16 * 16, 512)

        # 全连接层2: 512 → 1 (二分类输出)
        self.fc2 = nn.Linear(512, 1)

        # Sigmoid激活函数(将输出压缩到0-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """定义前向传播流程"""
        # 第一卷积块: Conv → ReLU → MaxPool
        x = torch.relu(self.conv1(x))  # [batch, 3, 128,128] → [batch, 32, 128,128]
        x = torch.max_pool2d(x, 2)  # → [batch, 32, 64,64]

        # 第二卷积块
        x = torch.relu(self.conv2(x))  # → [batch, 64, 64,64]
        x = torch.max_pool2d(x, 2)  # → [batch, 64, 32,32]

        # 第三卷积块
        x = torch.relu(self.conv3(x))  # → [batch, 128, 32,32]
        x = torch.max_pool2d(x, 2)  # → [batch, 128, 16,16]

        # 展平层: 准备全连接
        x = x.view(-1, 128 * 16 * 16)  # → [batch, 32768]

        # 全连接层1
        x = torch.relu(self.fc1(x))  # → [batch, 512]

        # 输出层
        x = self.fc2(x)  # → [batch, 1]
        x = self.sigmoid(x)  # → [0-1]概率值
        return x

    def load_model(model_path):
        """
        加载预训练模型的工具函数
        参数:
            model_path: 模型权重文件路径(.pth)
        返回:
            加载好的模型实例
        """
        model = CNNModel()  # 创建模型实例
        model.load_state_dict(torch.load(model_path))  # 加载权重
        model.eval()  # 设置为评估模式
        return model