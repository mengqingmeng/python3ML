from math import sqrt
import numpy as np
from collections import Counter
class KNNClassifier:
    def __init__(self,k):
        assert k >=1,'k must be valid'
        self.k = k
        self.x_train = None
        self.y_train = None
    
    def fit(self,x_train,y_train):
        assert x_train.shape[0] == y_train.shape[0],'the size of x_train must be equal to the size of y_train'
        assert self.k <= x_train.shape[0],'the size of x_train must be at least k'
        self._x_train = x_train
        self._y_train = y_train
        return self

    def predict(self,x_predict):
        #训练数据集和标签不能为空
        assert self._x_train is not None and self._y_train is not None,'data must fit before predict'
        #预测数据特征值和训练数据特征值数量相同3
        assert x_predict.shape[1] == self._x_train.shape[1],'the feature number of x_predict must be equal to x_train'
        y_predict = [self._predict(x) for x in x_predict]
        return np.array(y_predict)

    #给定单个预测数据，实行预测，并返回结果
    def _predict(self,x):
        assert x.shape[0] == self._x_train.shape[1],'the feature number of x must be equal to x_train'
        distance = [ sqrt(np.sum((i-x)**2)) for i in self._x_train]
        nearest = np.argsort(distance)
        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)
        return votes.most_common(1)[0][0]