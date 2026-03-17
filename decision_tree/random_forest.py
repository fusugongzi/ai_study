from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 2. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. 创建随机森林模型
# n_estimators: 森林中树的数量（默认100）
# max_features: 寻找最佳分割时考虑的随机特征数量
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

# 4. 训练模型
rf_clf.fit(X_train, y_train)

# 5. 预测与评估
y_pred = rf_clf.predict(X_test)
print(f"随机森林的准确率: {accuracy_score(y_test, y_pred):.2%}")

# 6. 查看特征重要性
# 这是随机森林最有用的功能之一，能告诉你哪些特征对分类贡献最大
for name, importance in zip(iris.feature_names, rf_clf.feature_importances_):
    print(f"特征: {name:20} 重要性: {importance:.4f}")