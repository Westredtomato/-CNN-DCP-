import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNNModel  # 从model.py导入自定义的CNN模型
#用于评估训练好的雾霾检测CNN模型性能
def evaluate():
    # ========== 数据预处理 ==========
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图像大小为128x128
        transforms.ToTensor(),  # 转换为PyTorch张量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到[-1,1]范围
    ])

    # ========== 加载测试数据集 ==========
    # 注意：这里的路径需要根据实际情况修改
    test_dataset = datasets.ImageFolder(
        r'C:\Users\westone\Desktop\CNN-DCP\classifier\data\test',
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,  # 每批处理32张图像
        shuffle=False  # 测试集不需要打乱顺序
    )

    # ========== 设备设置 ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动检测GPU

    # ========== 加载模型 ==========
    model = CNNModel()  # 初始化模型结构
    model.load_state_dict(torch.load('model.pth'))  # 加载训练好的权重
    model.to(device)  # 将模型移动到GPU(如果可用)
    model.eval()  # 设置为评估模式(关闭dropout等)

    # ========== 评估模型 ==========
    correct = 0  # 正确预测的样本数
    total = 0  # 总样本数

    with torch.no_grad():  # 禁用梯度计算(节省内存)
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # 数据移动到GPU

            outputs = model(inputs)  # 前向传播
            predicted = (outputs > 0.5).float()  # 以0.5为阈值进行二分类

            total += labels.size(0)  # 累计总样本数
            correct += (predicted.view(-1) == labels).sum().item()  # 累计正确预测数

    # ========== 计算结果 ==========
    accuracy = correct / total  # 计算准确率
    accuracy_percentage = accuracy * 100  # 转换为百分比
    print(f"Test Accuracy: {accuracy_percentage:.2f}%")  # 格式化输出


if __name__ == "__main__":
    evaluate()  # 执行评估函数