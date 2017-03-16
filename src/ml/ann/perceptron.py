import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from ml.util import extend, model, logistic

'''
使用感知器法则训练一个线性函数对数据集进行划分
设数据集为: {(x1,t1),(x2,t2), ... ,(xn,tn)}
1) 如果数据集是线性可分的
感知器法则可以正确分类所有样本，过程收敛
2) 如果数据集是线性不可分的
可以使用梯度下降算法(delta法则)找到一个最优线性函数，使得这个线性函数对整个数据集划分的效果最好
参见gradient_descent.py
'''
e = np.e
ln = np.log

def perceptron_rule(datas, iterations=100, step=0.1):
    '''
    感知器法则
    对于一个线性可分的数据集: {(x1,t1),(x2,t2), ... ,(xn,tn)}
    oi = sign(wxi)
    ∆w = λ(ti - oi)xi
    w = w + ∆w
    循环这个过程至所有数据均正确分类，w不再改变，过程收敛
    '''
    w = np.random.rand(3)
    print('init w is: ' + str(w))
    for i in range(iterations):
        step = 1 / (i + 1)
        raw_w = np.array(w)
        for x,t in datas:
            x = np.array([1] + x)
            o = np.sign(w.dot(x))
            dw = (t - o) * x
            w = w + step * dw
        print('w is: ' + str(w))
        if (w == raw_w).all():
            break
    return w

