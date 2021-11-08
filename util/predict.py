import numpy as np
import sympy as sp

BAYES = 1 #ベイズ推定
MLE = 0   #最尤推定

class PredictedDist:
    """
    本クラスは予測分布を提供するクラス。
    tの精度パラメータβと、ベイズ推定で用いる事前分布の精度パラメータαを入力する。
    その後、訓練データを本クラスのメソッドfit()で提供する事で、学習を行う。
    predict()メソッドを使用する事で、学習結果と入力されたxを用いて、目的変数tの予測分布のパラメータを提供する。
    αとβはデフォルト値として、α=5*10**(-3) β=11.1とする。
    """
    def __init__(self, alpha=5*10**(-3), beta=11.1, mode = MLE):
        self.alpha = alpha
        self.beta = beta
        self.mode = mode

    # 訓練データから学習を行う。
    def fit(self, X, t):
        if self.mode == MLE:
            self.w    = np.linalg.inv(X.T@X) @ X.T @ t
            self.sigma = 1/X.shape[0] * np.sum(np.power((X@self.w - t),2), axis=0)
        else:
            self.accuracy = self.alpha * np.identity(X.shape[1]) + self.beta*X.T @ X
            self.mu = self.beta * np.linalg.inv(self.accuracy) @ X.T @ t
    
    # 予測分布を算出する
    def predict(self, X):
        if self.mode == MLE:
            y = X@self.w
            y_std = np.sqrt(self.sigma)
        else:
            y = X.dot(self.mu.T)
            y_var = 1 / self.beta + np.sum(X.dot(np.linalg.inv(self.accuracy)) * X, axis=1)
            y_std = np.sqrt(y_var)
        return y, y_std