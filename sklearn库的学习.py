#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Jason Mess
# 字典特征抽取
from sklearn.feature_extraction import DictVectorizer
# 文本特征抽取和词频抽取
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba
# 特征预处理:归一化，标准化
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Imputer
# 缺失值的处理要导入 numpy和 imputer
import numpy as np
# 特征选择和特征降维：导入方差阈值还有PCA分解
# 删除低方差
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA

"""
特征工程是什么？
特征工程是将原始数据转换为更好的代表预测模型的潜在问题的特征的过程，
从而提高了对未知数据的预测准确性
"""

"""
字典特征抽取
对字典数据进行特征值化
对类别特征进行one-hot编码
"""


def dict_feature_extraction():
    """
    字典特征抽取
    :return: None
    """
    # 实例化
    dv = DictVectorizer(sparse=False)

    # 调用函数转换
    data = dv.fit_transform([{"city": "北京", "smog": 100},
                             {"city": "上海", "smog": 60},
                             {"city": "深圳", "smog": 30}])

    # 打印特征名称
    print(dv.get_feature_names())
    print(data)

    '''
    print(data.toarray())
    在false下，这样会报错：
    'numpy.ndarray' object has no attribute 'toarray'
    '''
    # print(type(data))

    # sparse 的 参数 是 True 的时候，输出的数据不方便看，要转换
    # print(data.toarray())
    # return None

    # 把one-hot编码恢复为字典
    # print(dv.inverse_transform(data))


def cur_ch_text():
    ch_1 = "一、不要别人点什么，就做什么"
    ch_2 = "二、推销自己"
    ch_3 = "三、学会带领团队"

    # 分段后把生成器转换为列表
    # 然后用空格连接起来
    c1 = ' '.join(list(jieba.cut(ch_1)))
    c2 = ' '.join(list(jieba.cut(ch_2)))
    c3 = ' '.join(list(jieba.cut(ch_3)))

    # print(c1, '\n', c2, '\n', c3)

    return c1, c2, c3


def text_feature_extraction():
    """
    :return None
    """

    En_text = ["life is short, use python", 'life is short, what about Lisp?']
    Ch_text = ["生命 短暂，用 python", "生命 短暂，lisp 怎么样？"]  # 单个字或者单个字母不计入统计
    c1, c2, c3 = cur_ch_text()

    cv = CountVectorizer()
    # data = cv.fit_transform(En_text)
    data = cv.fit_transform([c1, c2, c3])
    # print(data)
    print(cv.get_feature_names())
    print(data.toarray())

    # 得到的是一个单词出现的次数
    return


def word_frequency_extraction():
    """
    :return None
    """
    c1, c2, c3 = cur_ch_text()

    tf_idf = TfidfVectorizer()
    # stop_words 用来排除不需要的单词
    #    Tfidf = TfidfVectorizer(stop_words=['不要', '什么'])

    # data = cv.fit_transform(En_text)
    data = tf_idf.fit_transform([c1, c2, c3])
    # print(data)
    print(tf_idf.get_feature_names())
    print(data.toarray())

    return


# normalized: 归一化适用于精确小数聚集的情况
def normalized():
    """
    对数据进行归一化
    :return: None
    """
    mms = MinMaxScaler(feature_range=[0, 1])
    data = mms.fit_transform([[90, 2, 10, 40],
                              [60, 4, 15, 45],
                              [75, 3, 13, 46]])

    print(data)
    return None


# 标准化不容易受到异常点的处理，相对于归一法
def standard_sca():
    """
    :return: None
    """
    standard_s = StandardScaler()
    data = standard_s.fit_transform([[90, 2, 10, 40],
                                     [60, 4, 15, 45],
                                     [75, 3, 13, 46]])
    print(data)
    print("mean:平均值", '\n', standard_s.mean_, '\n', 'var:方差', '\n', standard_s.var_)
    '''
    # RecursionError: 超出最大递归深度
    # RecursionError: maximum recursion depth exceeded
    '''
    return None


def imputer_test():
    """
    # 对缺失值的处理
    :return:None
    missing_values:返回一个整数 默认是NaN
    strategy: 一个字符串 默认是mean (mean/median/most_freguent)
    axis: 0,1 (默认是0)
    """
    imp = Imputer(missing_values='NaN', strategy='median', axis=0)
    data = imp.fit_transform([[90, 2, 10, 40],
                              [np.nan, 4, 15, 45],
                              [75, 3, np.nan, 46]])  # nan1 = 60  nan2 = 13
    print(data)
    return None


def var_thre():
    """
    删除低方差特征
    :return: None
    threshold 的取值范围
    threshold : float, optional
        Features wi th a training-set variance lower than this threshold will
        be removed. The default is to keep all features with non-zero variance,
        i.e. remove the features that have the same value in all samples.
    """

    var_thr = VarianceThreshold(threshold=0.4)
    data = var_thr.fit_transform([[0, 2, 0, 3],
                                  [0, 1, 4, 3],
                                  [0, 1, 1, 3]])
    print(data)
    print(var_thr.variances_)


def pca_decomposition():
    """
    主成分分解降维
    :return: None
    """
    p_dec = PCA(n_components=0.95)
    data = p_dec.fit_transform([[2, 8, 4, 5],
                                [6, 3, 0, 8],
                                [5, 4, 9, 1]])
    print(data)


if __name__ == '__main__':
    # 字典特征抽取
    # dict_feature_extraction()

    # 文本特征抽取
    # text_feature_extraction()

    # 中文抽取
    # cur_ch_text()
    # word_frequency_extraction()

    # 归一化：normalized
    # normalized()

    # 标准化
    # standard_sca()

    # 缺失值的处理
    # imputer_test()

    # 删除地方差特征：特征选择
    var_thre()

    # 主成分分解降维
    # pca_decomposition()