import cv2
import face_recognition
import pickle
import numpy as np
from pathlib import Path


DEFAULT_DAT_DIR = Path(__file__).resolve().parent / "dat"

# 录入数据，将人脸图片转化为向量
def record_face(name, image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    if len(face_locations) == 0:
        print("未在图片中检测到人脸，请更换照片重试。")
        return

    if len(face_locations) > 1:
        print(f"检测到 {len(face_locations)} 张人脸，将使用第一张。")

    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    DEFAULT_DAT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DEFAULT_DAT_DIR / f"{name}_face_encoding.dat"
    with open(out_path, "wb") as f:
        pickle.dump(face_encodings[0], f)
    print(f"成功从照片录入 {name} 的特征数据！已保存到: {out_path}")

# 识别人脸
def start_recognition(image_path, dat_dir: Path = DEFAULT_DAT_DIR, tolerance: float = 0.5):
    dat_dir = Path(dat_dir)
    dat_files = sorted(dat_dir.glob("*_face_encoding.dat"))

    if not dat_files:
        print(f"未找到人脸特征文件（*_face_encoding.dat），请先录入。目录: {dat_dir}")
        return None

    known_encodings = []
    known_names = []
    for p in dat_files:
        try:
            with open(p, "rb") as f:
                known_encodings.append(pickle.load(f))
            known_names.append(p.name.replace("_face_encoding.dat", ""))
        except Exception:
            continue

    if not known_encodings:
        print(f"目录下特征文件不可用。目录: {dat_dir}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图片: {image_path}")
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    if len(face_encodings) == 0:
        print("未在图片中检测到人脸。")
        return None

    hits = []
    for face_encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_idx = int(np.argmin(distances))
        if float(distances[best_idx]) <= tolerance:
            hits.append(known_names[best_idx])

    if len(face_encodings) == 1:
        return hits[0] if hits else None
    return hits

if __name__ == "__main__":
    record_face("ZhangSan", "face_imgs/a.jpg")
    record_face("LiWu", "face_imgs/b.jpg")
    record_face("WangJiu", "face_imgs/c.jpg")

    print(start_recognition("face_imgs/d.jpg"))