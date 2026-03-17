# 导入必要的库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# 1. 加载数据集
iris = load_iris()
X = iris.data  # 特征：花萼长度、宽度，花瓣长度、宽度
y = iris.target  # 标签：三种不同的鸢尾花

# 2. 划分训练集和测试集（80% 训练，20% 测试）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 创建决策树模型
# 这里使用基尼系数 (gini) 作为准则，你也可以换成 'entropy'
clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)

# 4. 训练模型
clf.fit(X_train, y_train)

# 5. 评估模型
score = clf.score(X_test, y_test)
print(f"模型的准确率为: {score:.2%}")

# 6. 可视化决策树
plt.figure(figsize=(12, 8))
plot_tree(clf, 
          filled=True, 
          feature_names=iris.feature_names, 
          class_names=iris.target_names,
          rounded=True)
plt.show()