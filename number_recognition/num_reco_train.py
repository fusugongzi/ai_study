# 训练模型，将模型权重参数保存到model下

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 配置超参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
learning_rate = 0.001
epochs = 3

# 2. 数据准备：缩放到 0-1 之间并进行标准化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 搭建神经网络 (CNN)
class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), # 输入1通道(灰度), 输出16通道
            nn.ReLU(),
            nn.MaxPool2d(2),                            # 28x28 -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)                             # 14x14 -> 7x7
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)                          # 最终输出10类 (0-9)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = DigitNet().to(device)

# 4. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 训练模型
print(f"正在使用 {device} 进行训练...")
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()           # 梯度清零
        output = model(data)            # 前向传播
        loss = criterion(output, target)# 计算误差
        loss.backward()                 # 反向传播
        optimizer.step()                # 更新权重
        
        if batch_idx % 200 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

# 6. 测试模型准确率
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

print(f"\n测试集准确率: {100. * correct / len(test_loader.dataset)}%")

# 假设你的模型变量名是 model
torch.save(model.state_dict(), "model/digit_model.pth")
print("模型权重已保存到本地：model/digit_model.pth")