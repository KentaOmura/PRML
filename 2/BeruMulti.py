import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import random

MLE   = 0
BAYSE = 1

# ベルヌーイ分布(二項分布)
class BernoulliDistribution:
    def __init__(self, a=1, b=1, mode=MLE):
        self.u = -1
        self.a = a
        self.b = b
        self.mode = mode

    def fit(self,X):
        if self.mode == MLE:
            self.u = np.sum(X)/X.size
        else:
            m = np.sum(X)
            l = X.size - m
            u = np.linspace(0, 1, 100)
            self.u = sp.gamma(m+l+self.a+self.b)/(sp.gamma(m+self.a)*sp.gamma(l+self.b)) * np.power(u, m+self.a-1) * np.power(1-u, l+self.b-1)
            self.a += m
            self.b += l
        return self.u
    
    # 次に表が出る確率
    def estimate(self, X):
        if self.mode == MLE:
            return self.fit(X)
        else:
            m = np.sum(X)
            l = X.size - m
            result = (m+self.a) /(m+l+self.a+self.b)
            self.fit(X)
            return result

# 多項分布
class MultinomialDistribution:
    def __init__(self, K, alpha = None, mode=MLE):
        self.u = -1
        self.K = K
        if alpha == None:
            self.alpha = np.zeros(self.K)
        else:
            self.alpha = alpha
        self.mode  = mode
    
    # Xの列がデータ 行はデータの数
    def fit(self, X):
        if self.mode == MLE:
            self.u = np.sum(X,axis=0) / X.shape[0]
        else:
            m = np.sum(X, axis=0)
            u = np.full(self.K, 1/self.K)
            # 事前分布はディリクレ分布を使用する。
            self.u = sp.gamma(np.sum(m)+np.sum(self.alpha))/np.prod(sp.gamma(self.alpha + m)) * np.prod(np.power(self.u, self.alpha+m-1)) 
            # 逐次処理用
            self.alpha += m
        return self.u

    # クラス毎の確率を返す
    def estimate(self,X):
        return self.fit(X)

def main2():
    data = np.array([1,0,0,0,0]).reshape(1,-1)
    data2 = np.array([1,0,0,0,0]).reshape(1,-1)

    model = MultinomialDistribution(K=5,mode=BAYSE)
    model.fit(data2)
    print(model.estimate(data))

def main():
    # コイン投げで5回表が出たとした時のデータ
    data = np.array([1,1,1,1,1])
    data2 = np.array([1,1,1,1,1])

    # 分布の形状を描画
    u = np.linspace(0, 1, 100)
    # 事前分布
    a = 2
    b = 2
    by = sp.gamma(a+b)/(sp.gamma(a)*sp.gamma(b)) * np.power(u, a-1) * np.power(1-u, b-1)
    plt.subplot2grid((2,2),(0,0))
    plt.plot(u,by,label="Prior")
    plt.xlabel("u")
    # 尤度関数
    li = []
    for i in np.nditer(u):
        li.append(sp.binomial(data.size, np.sum(data)) * (np.power(i,np.sum(data))*np.power(1-i, data.size - np.sum(data))))
    plt.subplot2grid((2,2),(0,1))
    plt.plot(u, li, label="Likelihood")
    plt.xlabel("u")
    # 事後分布
    model = BernoulliDistribution(a=a,b=b,mode=BAYSE)
    ay = model.fit(data)
    plt.subplot2grid((2,2),(1,0),colspan=2)
    plt.plot(u, ay, label="ex-post")
    plt.xlabel("u")
    plt.show()

    # x=1の時の確率
    model2 = BernoulliDistribution()
    print(model.estimate(data)*100)
    print(model2.estimate(data)*100) # 最尤推定を使用した結果

main2()