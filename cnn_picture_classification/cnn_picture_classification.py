import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from simple_cnn import SimpleCNN

def main():
    # ========================================================
    # 第一步：准备数据
    # ========================================================

    # 1.1 定义数据预处理流程：像素值 [0,255] → Tensor [0,1] → 标准化到 [-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 1.2 加载训练集 (60,000 张) 和测试集 (10,000 张)
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                             shuffle=False, num_workers=2)

    # 1.3 定义类别名称（索引 0~9 对应的标签）
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")

    # ========================================================
    # 第二步：构建模型
    # ========================================================

    # 2.1 实例化 CNN 模型
    model = SimpleCNN()
    print(model)

    # 2.2 选择运算设备（优先使用 GPU）
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"正在 {device} 上进行训练。")

    # 2.3 定义损失函数（交叉熵，适用于多分类）
    criterion = nn.CrossEntropyLoss()

    # 2.4 定义优化器（Adam，自适应学习率，默认 lr=0.001）
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ========================================================
    # 第三步：训练模型
    # ========================================================

    num_epochs = 5
    train_losses = []

    print("开始训练...")
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 3.1 取出一个 batch 的数据，搬到对应设备上
            inputs, labels = data[0].to(device), data[1].to(device)

            # 3.2 梯度清零（防止上一轮梯度累积）
            optimizer.zero_grad()

            # 3.3 前向传播：输入图片 → 模型输出 10 个类别的得分
            outputs = model(inputs)

            # 3.4 计算损失：对比预测得分与真实标签
            loss = criterion(outputs, labels)

            # 3.5 反向传播：根据损失计算每个参数的梯度
            loss.backward()

            # 3.6 更新参数：优化器根据梯度调整模型权重
            optimizer.step()

            running_loss += loss.item()
            if i % 300 == 299:
                print(f'[Epoch {epoch + 1}, Batch {i + 1:5d}] loss: {running_loss / 300:.3f}')
                train_losses.append(running_loss / 300)
                running_loss = 0.0

    print('训练完成。')

    # ========================================================
    # 第四步：评估模型
    # ========================================================

    model.eval()
    correct = 0
    total = 0

    # 4.1 在测试集上计算准确率（关闭梯度计算以节省内存）
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'模型在 10,000 张测试图像上的准确率: {100 * correct / total:.2f}%')

    # ========================================================
    # 第五步：可视化预测结果
    # ========================================================

    # 5.1 取一个 batch 的测试图片进行预测
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)

    # 5.2 展示前 6 张图片的预测结果与真实标签
    images = images.numpy()
    plt.figure(figsize=(10, 8))
    for i in range(6):
        plt.subplot(2, 3, i+1)
        img = images[i] / 2 + 0.5  # 反标准化：[-1,1] → [0,1]
        plt.imshow(np.transpose(img, (1, 2, 0)), cmap='gray')
        plt.title(f"预测: {classes[predicted[i]]}\n(真实: {classes[labels[i]]})")
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()