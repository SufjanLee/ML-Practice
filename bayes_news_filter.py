from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV


def fetch_data():
	"""
	使用fetch方法获取稍大的数据集，并做数据集的划分
	:return: 训练集和测试集的特征值和目标值
	"""
	news = fetch_20newsgroups(data_home="E:\python project\Machine Learning", subset="all")
	x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.30, random_state=22)
	return x_train, x_test, y_train, y_test


def feature_extraction(x_train, x_test):
	"""
	使用TF-IDF方法对文本内容进行提取
	:param x_train: 训练集特征值
	:param x_test: 测试集特征
	:return:
	"""
	transfer = TfidfVectorizer()
	x_train = transfer.fit_transform(x_train)
	x_test = transfer.transform(x_test)
	return x_train, x_test


def bayes(x_train, y_train, x_test, y_test, param_grid):
	"""
	使用贝叶斯-MultinomialNB计算概率，并使用GridsearchCV来选择最优的模型参数
	:param x_train:
	:param y_train:
	:param x_test:
	:param y_test:
	:param param_grid: 需要选择的最优参数
	:return:
	"""
	estimator = MultinomialNB()
	estimator = GridSearchCV(estimator, param_grid=param_grid, cv=5)
	estimator.fit(x_train, y_train)

	# 准确率和预测值
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
	x_tr, x_te, y_tr, y_te = fetch_data()
	x_tr, x_te = feature_extraction(x_tr, x_te)

	param_dict = {"alpha": [0.8, 0.85, 0.9, 0.95, 1.00, 1.05, 1.10, 1.15]}
	bayes(x_tr, y_tr, x_te, y_te, param_grid=param_dict)


