from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def iris_data():
	"""
	读取IRIS数据集
	:return:
	"""
	iris = load_iris()
	print("鸢尾花数据集：\n", iris)
	print("鸢尾花描述：\n", iris["DESCR"])
	print("鸢尾花特征：\n", iris.feature_names)
	print("查看特征值：\n", iris.data, iris.data.shape)

	x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=22)
	print("训练集数目：\n", x_train.shape)
	print("测试集数目：\n", x_test.shape)
	return x_train, x_test, y_train, y_test


def normal(x_train, x_test):
	"""
	对数据集进行标准化
	:param x_train: 训练集
	:param x_test: 测试集
	:return:
	"""
	estimator = Normalizer()
	x_train = estimator.fit_transform(x_train)
	x_test = estimator.transform(x_test)
	print("处理后训练集：\n", x_train.shape)
	print("处理后测试集：\n", x_test)
	return x_train, x_test


def decision(x_train, x_test, y_train, y_test, param_dict):
	estimator = DecisionTreeClassifier()
	estimator = GridSearchCV(estimator, param_grid=param_dict, cv=5)
	estimator.fit(x_train, y_train)
	# 准确率
	score = estimator.score(x_test, y_test)
	prediction = estimator.predict(x_test)
	print("准确率：\n", score)
	print("最佳结果：\n", estimator.best_score_)
	print("最佳估计器: \n", estimator.best_estimator_)
	print("最佳参数：\n", estimator.best_params_)
	print("预测值：\n", prediction)
	print("真实值与预测值比较：\n", prediction == y_test)
	return None


if __name__ == '__main__':
	x_tr, x_te, y_tr, y_te = iris_data()
	x_tr, x_te = normal(x_tr, x_te)
	# Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
	param_dict = {"criterion": ["gini", "entropy"]}
	decision(x_tr, x_te, y_tr, y_te, param_dict)
