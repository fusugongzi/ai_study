import cv2
import numpy as np
import re
from paddleocr import PaddleOCR

PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤川青藏琼宁'
PLATE_RE = re.compile(rf'[{PROVINCES}][A-Z][·.\-]?[A-Z0-9]{{4,6}}')


def clean_plate_text(text):
    """去除 OCR 识别中的干扰符号（中间点、空格等），只保留汉字/字母/数字"""
    return re.sub(r'[·.\-\s]', '', text)


def detect_plate_region(img_path):
    """通过颜色检测定位车牌区域（支持蓝牌、绿牌、黄牌）"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    masks = [
        cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255])),   # 蓝牌
        cv2.inRange(hsv, np.array([35, 80, 80]), np.array([85, 255, 255])),      # 绿牌
        cv2.inRange(hsv, np.array([15, 80, 80]), np.array([35, 255, 255])),      # 黄牌
    ]
    mask = masks[0] | masks[1] | masks[2]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = w / h if h > 0 else 0
        area = w * h
        if 2.0 < ratio < 6.0 and area > 1000 and area > best_area:
            best, best_area = (x, y, w, h), area

    if best:
        x, y, w, h = best
        pad = 10
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)
        return img[y1:y2, x1:x2]
    return None


def extract_plate_from_ocr(res):
    """从 OCR 结果中按正则过滤并合并车牌文字"""
    texts = res['rec_texts']
    scores = np.array(res['rec_scores'])
    boxes = np.array(res['rec_boxes'])

    for i, t in enumerate(texts):
        cleaned = clean_plate_text(t)
        if PLATE_RE.search(cleaned):
            return PLATE_RE.search(cleaned).group(), float(scores[i])

    n = len(texts)
    used = [False] * n
    for i in range(n):
        if used[i] or not texts[i].strip():
            continue
        group = [i]
        used[i] = True
        cy = (boxes[i][1] + boxes[i][3]) / 2
        h = max(boxes[i][3] - boxes[i][1], 1)

        for j in range(n):
            if used[j] or not texts[j].strip():
                continue
            cy_j = (boxes[j][1] + boxes[j][3]) / 2
            if abs(cy - cy_j) < h * 2:
                group.append(j)
                used[j] = True

        group.sort(key=lambda idx: boxes[idx][0])
        combined = clean_plate_text(''.join(texts[idx] for idx in group))
        m = PLATE_RE.search(combined)
        if m:
            avg_score = float(np.mean(scores[group]))
            return m.group(), avg_score

    return None, 0.0


def upscale_plate(img, target_height=150):
    """将裁剪的车牌图放大到合适尺寸，提高 OCR 识别率"""
    h, w = img.shape[:2]
    if h >= target_height:
        return img
    scale = target_height / h
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def recognize(img_path):
    ocr = PaddleOCR(use_textline_orientation=True, lang='ch')

    plate_img = detect_plate_region(img_path)
    if plate_img is not None:
        print("=== 检测到车牌区域，对裁剪区域进行 OCR ===")
        plate_img = upscale_plate(plate_img)
        result = list(ocr.predict(plate_img))
        found = False
        for res in result:
            for text, score in zip(res['rec_texts'], res['rec_scores']):
                if text.strip():
                    print(f"文字: {text}, 置信度: {score:.4f}")
                    found = True
        if found:
            for res in result:
                plate, confidence = extract_plate_from_ocr(res)
                if plate:
                    print(f"  >>> 车牌号: {plate}, 置信度: {confidence:.4f}")
                    return
        print("  裁剪区域未识别出文字，回退到全图识别...")

    print("=== 对全图进行 OCR 并过滤车牌 ===")
    result = list(ocr.predict(img_path))
    for res in result:
        plate, confidence = extract_plate_from_ocr(res)
        if plate:
            print(f"  >>> 车牌号: {plate}, 平均置信度: {confidence:.4f}")
            return
    print("  未检测到车牌")


recognize('car_plate_1.jpg')