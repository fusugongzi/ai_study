import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# 1. 必须定义和训练时完全一样的模型结构
class DigitNet(nn.Module):
    def __init__(self):
        super(DigitNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# 2. 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitNet().to(device)
model.load_state_dict(torch.load("model/digit_model.pth", map_location=device))
model.eval()  # 必须切换到评估模式！

# 3. 图像预处理函数
def predict_digit(image_path):
    # 打开图片并转为灰度图
    img = Image.open(image_path).convert('L')
    
    # 核心预处理步骤
    transform = transforms.Compose([
        transforms.Resize((28, 28)),      # 强制缩放到 28x28
        transforms.ToTensor(),            # 转为张量 (0-1)
        transforms.Normalize((0.1307,), (0.3081,)) # 使用 MNIST 的标准差和均值
    ])
    
    # 增加 Batch 维度 (1, 1, 28, 28)
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # 4. 执行预测
    with torch.no_grad():
        output = model(img_tensor)
        # 找到得分最高的索引，即为预测数字
        prediction = output.argmax(dim=1).item()
        
    return prediction

# --- 使用示例 ---
# 假设你有一张名为 'my_number.png' 的图片
result = predict_digit('my_number.png')
print(f"模型识别结果是: {result}")