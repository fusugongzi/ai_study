import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # --- 卷积层 1 ---
        # 输入通道=1 (灰度图), 输出通道=32, 卷积核大小=3x3, 步长=1, 填充=1 (保持尺寸)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        # 池化层 1: 2x2 最大池化，尺寸减半 (28x28 -> 14x14)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- 卷积层 2 ---
        # 输入通道=32, 输出通道=64, 卷积核=3x3, 步长=1, 填充=1
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu2 = nn.ReLU()
        # 池化层 2: 2x2 最大池化 (14x14 -> 7x7)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # --- 全连接层 ---
        # 展平数据：经过两层池化后，图像变成了 64 个 7x7 的特征图。
        # 输入神经元 = 64 * 7 * 7
        self.flatten = nn.Flatten()
        
        # 第一个全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.relu3 = nn.ReLU()
        
        # 输出层：输出 10 个类别的得分
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 定义数据在网络中的流向
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x