import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz


def load_data():
	# 1. 读取数据
	df = pd.read_csv("train.csv")
	# 2. 筛选有用数据特征
	x = df[["Pclass", "Age", "Sex"]]
	y = df["Survived"]
	# 3. 补充空缺值,以此特征下的平均值填补NULL
	# df.fillna()填补空缺值
	x["Age"].fillna(x["Age"].mean(), inplace=True)
	# 4. 将df转化为字典类型
	x = x.to_dict(orient="records")
	# 5. 划分数据集
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=22)
	# print("训练集： \n", x)
	# print("测试集： \n", y)
	return x_train, x_test, y_train, y_test


def feature(x_train, x_test):
	transfer = DictVectorizer()
	x_train = transfer.fit_transform(x_train)
	x_test = transfer.transform(x_test)

	"""
	get_feature_names(self)         [source]
	Returns a list of feature names, ordered by their indices.
	If one-of-K coding is applied to categorical features, this will include the constructed feature names but not the original ones.
	"""
	name = transfer.get_feature_names()
	# print(name)
	print(x_train)
	return x_train, x_test, name


def decision(x_train, x_test, y_train, y_test, name):
	estimator = DecisionTreeClassifier(criterion="entropy")
	estimator.fit(x_train, y_train)
	# 准确率
	score = estimator.score(x_test, y_test)
	prediction = estimator.predict(x_test)
	print("准确率：\n", score)
	print("预测值：\n", prediction)
	print("真实值与预测值比较：\n", prediction == y_test)
	export_graphviz(estimator, out_file="E:\python project\Machine Learning\stitanic_tree.dot", feature_names=name)
	return None


if __name__ == '__main__':
	x_tr, x_te, y_tr, y_te = load_data()
	x_tr, x_te, f_name = feature(x_tr, x_te)
	decision(x_tr, x_te, y_tr, y_te, f_name)
