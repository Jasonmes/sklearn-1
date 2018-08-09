#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Jason Mess
# 导入鸢尾植物数据集，新闻数据集
from sklearn.datasets import load_iris, fetch_20newsgroups
# 数据划分
from sklearn.model_selection import train_test_split


# 划分鸢尾植物数据集
def iris_tailed_plant():
    """
    加载鸢尾植物数据集
    :return:
    bunch：堆
    iris中文指鸢尾植物
    这里存储了其萼片和花瓣的长宽，一共4个属性，鸢尾植物又分三类。
    与之相对，iris里有两个属性iris.data，iris.target
    data里是一个矩阵，每一列代表了萼片或花瓣的长宽
    一共4列，每一列代表某个被测量的鸢尾植物，一共采样了150条记录，

    print(iris_bunch.data.shape)
    所以查看这个矩阵的形状iris_bunch.data.shape，返回：(150, 4)
    """
    iris_bunch = load_iris()

    """
    sepal length:萼片
    petal length:花瓣
    1：打印特征值名称
    2：打印特征值
    3：打印目标值名称
    """
    # print(iris_bunch.feature_names)
    # print(iris_bunch.data)
    # print(iris_bunch.target_names)

    """
    数据集
    训练集和测试集
    train_test_split经常用到
    """
    x_train, x_test_feature, y_train, y_test_target = train_test_split(iris_bunch.data,
                                                                       iris_bunch.target,
                                                                       test_size=0.25)
    print('训练值特征值:\n', x_train)
    print('训练值目标值:\n', y_train)
    print('测试集特征值:\n', x_test_feature)
    print('测试集目标值:\n', y_test_target)


def load_news():
    """
    # 划分数据集
    :return:
    """
    fetch_news = fetch_20newsgroups(data_home='./data/', subset='all')
    """
    打印新闻内容
    打印标签值
    """
    # print(fetch_news.data[2])
    # print(fetch_news.target)
    x_train, x_test_feature, y_train, y_test_target = train_test_split(fetch_news.data,
                                                                       fetch_news.target,
                                                                       test_size=0.25)
    print('训练值特征值:\n', x_train)
    print('训练值目标值:\n', y_train)
    print('测试集特征值:\n', x_test_feature)
    print('测试集目标值:\n', y_test_target)


if __name__ == '__main__':
    # 鸢尾植物数据划分
    # iris_tailed_plant()

    # 新闻数据划分
    load_news()

