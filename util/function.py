import numpy as np

def createData(f,low, high, size):
    x = np.random.uniform(low=low, high=high, size=size)
    t = f(x) + np.random.normal(loc=0, scale=1/11.1, size=size)
    return x,t