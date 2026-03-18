from ultralytics import YOLO
import cv2

# 1. 加载预训练的 YOLOv8 模型 (基于 CNN 架构)
# 'yolov8n.pt' 是纳米级模型，速度极快，适合 CPU 运行
model = YOLO('yolov8n.pt') 

# 2. 对图像进行目标检测
# 你可以替换为本地图片路径，如 'my_car.jpg'
results = model('https://ultralytics.com/images/bus.jpg')

# 3. 解析并展示结果
for result in results:
    # 展示检测后的可视化图像
    result.show() 
    
    # 打印检测到的所有目标信息
    for box in result.boxes:
        class_id = int(box.cls[0])
        label = model.names[class_id]
        confidence = box.conf[0].item()
        coords = box.xyxy[0].tolist() # [左上x, 左上y, 右下x, 右下y]
        
        print(f"发现目标: {label}, 置信度: {confidence:.2f}, 坐标: {coords}")