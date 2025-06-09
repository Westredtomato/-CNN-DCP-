import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNNModel  # 从model.py导入自定义的CNN模型
#使用PyTorch框架实现的卷积神经网络(CNN)训练程序，用于将图像分类为"有雾霾"或"无雾霾"
# ========== 数据预处理 ==========
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 将所有图像调整为128x128像素
    transforms.ToTensor(),  # 将PIL图像转换为PyTorch张量
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]范围
])

# ========== 数据加载 ==========
# 训练数据集路径 (注意: 需要替换为你自己的路径)
train_dataset = datasets.ImageFolder(
    r'C:\Users\westone\Desktop\CNN-DCP\classifier\data\train',
    transform=transform
)
# 测试数据集路径
test_dataset = datasets.ImageFolder(
    r'C:\Users\westone\Desktop\CNN-DCP\classifier\data\test',
    transform=transform
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # 训练集: 批量32, 打乱顺序
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  # 测试集: 批量32, 不打乱顺序

# ========== 模型初始化 ==========
model = CNNModel()  # 实例化自定义CNN模型

# ========== 损失函数和优化器 ==========
criterion = nn.BCELoss()  # 二分类交叉熵损失函数 (Binary Cross Entropy)
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器, 学习率0.001

# ========== 训练循环 ==========
num_epochs = 10  # 训练轮数
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0

    # 遍历训练集所有批次
    for inputs, labels in train_loader:
        optimizer.zero_grad()  # 清空梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs.squeeze(), labels.float())  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
        running_loss += loss.item()  # 累计损失

    # 打印每轮的平均损失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

# ========== 模型保存 ==========
torch.save(model.state_dict(), 'model.pth')  # 保存模型权重
print("Model saved as 'model.pth'")

# ========== 模型测试 ==========
model.eval()  # 设置模型为评估模式
correct = 0
total = 0

with torch.no_grad():  # 禁用梯度计算(节省内存)
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float()  # 以0.5为阈值进行二分类
        total += labels.size(0)  # 累计总样本数
        correct += (predicted == labels.float()).sum().item()  # 累计正确预测数

# 打印测试准确率
print(f'Test Accuracy: {100 * correct / total:.2f}%')