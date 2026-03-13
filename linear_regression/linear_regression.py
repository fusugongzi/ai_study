import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 房价预测，一元线性回归，假设房子价格和面积是一元的关系
# 并不是所有的y都满足y=wx+b,当预测值与真实值之间存在偏差时，这个差值被称为残差，追求的是最小化均方误差。
def house_price_prediction_v1():
    # 1. 模拟生成数据 (设置随机种子保证结果可复现)
    np.random.seed(42)
    # 假设房屋面积在 50 到 150 平方米之间
    X = np.random.rand(50, 1) * 100 + 50 
    # 设置真实规律：y = 2.1 * x + 15 (单价2.1万，基础地价15万)
    # 加上 np.random.randn(50, 1) * 15 作为“噪声”（如装修、楼层差异）
    y = 2.1 * X + 15 + np.random.randn(50, 1) * 15

    # 2. 创建并训练线性回归模型
    model = LinearRegression()
    model.fit(X, y)

    # 3. 查看模型学到的参数
    w = model.coef_[0][0]
    b = model.intercept_[0]
    print(f"模型学到的公式为: y = {w:.2f}x + {b:.2f}")

    # 4. 进行预测
    test_area = np.array([[110]])
    predicted_price = model.predict(test_area)
    print(f"预测 110 平方米的房价为: {predicted_price[0][0]:.2f} 万元")

    # 5. 可视化结果
    plt.scatter(X, y, color='blue', label='Actual Data (with noise)')
    plt.plot(X, model.predict(X), color='red', label='Linear Regression Line')
    plt.xlabel('Area (sqm)')
    plt.ylabel('Price (10k RMB)')
    plt.legend()
    plt.show()

# 房价预测，多元线性回归，假设房子价格和面积、距离是一元的关系
def house_price_prediction_v2():
    # 1. 准备数据 (特征：[面积, 距离])
    # 假设我们有 5 套房子的数据
    X = np.array([
        [60, 2],   # 60平，离中心2公里
        [85, 5],   # 85平，离中心5公里
        [100, 1],  # 100平，离中心1公里
        [120, 10], # 120平，离中心10公里
        [150, 15]  # 150平，离中心15公里
    ])

    # 实际成交价 (万元)
    y = np.array([150, 170, 240, 210, 260])

    # 2. 训练模型
    model = LinearRegression()
    model.fit(X, y)

    # 3. 输出学习到的系数
    w1, w2 = model.coef_
    b = model.intercept_

    print(f"学到的公式: y = {w1:.2f} * 面积 + {w2:.2f} * 距离 + {b:.2f}")

    # 4. 预测一套新房
    # 面积 110平，距离市中心 3 公里
    new_house = np.array([[110, 3]])
    prediction = model.predict(new_house)

    print(f"预测 110平/距离3km 的房价为: {prediction[0]:.2f} 万元")

if __name__ == "__main__":
    house_price_prediction_v2()