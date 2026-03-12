# 有一个应用场景，根据打斗镜头数和亲吻镜头数 来区分一个电影是爱情片还是动作片
# 已有一些例子，这种场景使用knn算法比较合适

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 1. 准备训练数据 (特征：[打斗镜头数, 亲吻镜头数])
# 标签：0 代表 动作片, 1 代表 爱情片
X_train = np.array([
    [100, 5],   # 动作片 A
    [95, 10],   # 动作片 B
    [105, 8],   # 动作片 C
    [5, 80],    # 爱情片 D
    [10, 95],   # 爱情片 E
    [2, 110]    # 爱情片 F
])
y_train = np.array([0, 0, 0, 1, 1, 1])

# 2. 初始化 KNN 模型，设置 K=3 (看最近的3个邻居)
knn = KNeighborsClassifier(n_neighbors=3)

# 3. 训练模型 (其实就是把坐标点存起来)
knn.fit(X_train, y_train)

# 4. 预测新电影
# 假设有一部新电影 X：有 98 个打斗镜头，2 个亲吻镜头
new_movie = np.array([[98, 2]])
prediction = knn.predict(new_movie)

# 5. 输出结果
movie_type = "动作片" if prediction[0] == 0 else "爱情片"
print(f"这部新电影的预测类型是：{movie_type}")