import numpy as np
class StandardScaler:
    def __init__(self):
        #均值
        self.mean_ = None;
        #方差
        self.scale_ = None;
    def fit(self,X):
        assert X.ndim == 2, 'the demension of X must be 2'
        #计算每列的均值，结果在数组中
        self.mean_ = np.array([np.mean(X[:,i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:,i]) for i in range(X.shape[1])])
        return self

    def transform(self,X):
        assert X.ndim == 2 ,'the demension of X must be 2'
        assert self.mean_ is not None and self.scale_ is not None ,'must fit before transform'
        assert X.shape[1] == len(self.mean_),'the feature number of x must be equal to mean_ and std_'
        resX = np.empty(shape=X.shape,dtype=float)
        #遍历列,计算列后的结果 放在列中
        for col in range(X.shape[1]):
            resX[:,col] = (X[:,col]-self.mean_[col]) / self.scale_[col] 
        return resX