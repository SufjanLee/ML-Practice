import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def read_data():
	# 1.读取数据
	data = pd.read_csv("train.csv")

	# 2. 过滤部分数据，缩小数据集
	data = data.query("x < 2.5 & x > 2 & y < 1.5 & y >1.0")

	# 3. 转换时间格式
	time_value = pd.to_datetime(data["time"], unit="s")
	time = pd.DatetimeIndex(time_value)
	data["day"] = time.day
	data["weekday"] = time.weekday
	data["hour"] = time.hour

	# 4. 过滤签到次数较少的数据
	place_count = data.groupby("place_id").count()["row_id"]
	tem = place_count[place_count >= 30]
	tem1 = data["place_id"].isin(tem.index.values)

	# 5. 保存最终数据
	data_final = data[tem1]

	return data_final


def pre(data_final):
	# 1. 提取特征值
	x = data_final[["x", "y", "accuracy", "day", "weekday", "hour"]]

	# 2. 提取目标值
	y = data_final["place_id"]

	# 3. 数据集划分
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=22)

	# 4. 对训练集和测试集进行标准化
	estimator = StandardScaler()
	x_train = estimator.fit_transform(x_train)
	x_test = estimator.transform(x_test)
	return x_train, x_test, y_train, y_test


def knn_grid(x_train, y_train, x_test, y_test, param_dict):
	# 1. 实例化估计器
	estimator = KNeighborsClassifier()
	# 网格搜索最优参数
	estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
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
	data = read_data()
	x_tr, x_te, y_tr, y_te = pre(data)
	param_dict = {"n_neighbors": [3, 5, 7, 9, 11, 13, 15]}
	# "algorithm": ["auto", "kd_tree", "ball_tree"], "p": [1, 2, 3, 4, 5]
	knn_grid(x_tr, y_tr, x_te, y_te, param_dict)

