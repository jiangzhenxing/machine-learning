import numpy as np
from functools import reduce
from ml.util import extend, model, logistic

'''
使用梯度下降算法训练一个线性函数对数据集进行划分
设数据集为: {(x1,t1),(x2,t2), ... ,(xn,tn)}
如果数据集是线性不可分的
可以使用梯度下降算法(delta法则)找到一个最优线性函数，使得这个线性函数对整个数据集划分的效果最好
对最优的度量可以使用不同的标准
LMS最小均方差: ∑(ti-wxi)**2，是一个比较常用的误差度量方式，最优即使数据集方差最小
LOGISTIC逻辑函数，最优即使样本相应类别逻辑概率的乘积最大
'''

e = np.e
ln = np.log

def grad_descent_regression(datas, iteration=100, step=0.01, initw=0, cost='LMS', verbose=0):
    '''
    对于一个线性不可分的数据集: {(x1,t1),(x2,t2), ... ,(xn,tn)}
    找到一个线性函数，使得这个线性函数对整个数据集划分的误差最小
    cost: 误差度量方式
    LMS指最小均方差: ∑(ti-oi)**2，是一个比较常用的误差度量方式
    对数据集中的每个样本xi:
    oi = wxi
    cost = ∑1/2(ti-oi)**2 = ∑1/2(ti-wxi)**2
    ▽ w = ∑-(ti-wxi)xi = ∑(wxi-ti)xi
    w = w + λ▽ w
    重复这个过程至w收敛
    LOGISTIC指逻辑函数，即使样本相应类别逻辑概率的乘积最大
    Pi = 1 / (1 + e**(-wxi))
    cost = ln(∏Pi**ti(1-Pi)**(1-ti)) = ∑ln(Pi**ti(1-Pi)**(1-ti)) = -∑[tiln(1+e**(-wxi)) + (1-ti)ln(1+e**wxi)]
    ▽ w = ∑[ti/(1+e**wxi) - (1-ti)/(1+e**(-wxi))]xi
    '''
    if initw == 0: w = np.zeros(len(datas[0]) + 1)
    elif initw == 'random': w = np.random.rand(len(datas[0]) + 1)
    else: w = np.array(initw)
    print('init w is ' + str(w))

    for i in range(1,iteration):
        delta = np.zeros(len(datas[0]) + 1)
        step0 = step / (ln(i) + 1)
        for x,t in datas:
            x = np.array([1] + x)
            # print('x is: ' + str(x) + ', w is: ' + str(w))
            if cost == 'LMS':
                delta += (w.dot(x) - t) * x
            elif cost == 'LOGISTIC':
                if t == -1: t = 0
                delta += (t / (1 + e ** w.dot(x)) - (1 - t) / (1 + e ** -w.dot(x))) * x
        if model(delta) < 1e-20: continue
        gradient = delta

        if cost == 'LMS':
            w = w - step0 * gradient # 梯度向量方向为误差上升最快方向，故取负为误差下降最快方向
            if verbose == 1:
                error = sum([(w.dot(np.array([1] + x))) ** 2 - t for x,t in datas ])
        elif cost == 'LOGISTIC':
            w = w + step0 * gradient # 梯度向量方向为样本对应类别概率乘积上升最快方向
            #error = -sum([ ln(1 + e ** -w.dot(np.array([1] + x))) if t == 1 else ln(1 + e ** w.dot(np.array([1] + x))) for x,t in datas ])
            if verbose == 1:
                error = reduce(lambda x,y: x * y, [ 1 / (1 + e ** -w.dot(np.array([1] + x))) if t == 1 else 1 / (1 + e ** w.dot(np.array([1] + x))) for x,t in datas ])
        if verbose == 1:
            print('step0 is ' + str(step0) + ', delta is ' + str(delta) + ', w is ' + str(w) + \
                ', slope is ' + str(- w[1] / w[2]) + ', error is ' + str(error))

    return w

def stoch_grad_descent_regression(datas, iteration=100, step=0.01, initw=0, cost='LMS', verbose=0):
    '''
    随机梯度下降与梯度下降相似，
    只是对数据集中的每个样本xi，梯度向量的方向取(ti-wxi)xi，然后根据此梯度向量的方向更新w，不需要根据整个数据集样本的误差计算梯度向量
    这相当于为每个样本定义一个误差函数：cost = (ti - oi) ** 2 = (ti - wxi) ** 2 和梯度向量(ti - wxi)xi
    逻辑回归中目标函数为：-tiln(1 + e ** -wxi) - (1 - ti)ln(1 + e ** wxi)
    梯度向量为：(ti / (1 + e ** wxi) - (1 - ti) / (1 + e ** -wxi))xi
    '''
    if initw == 0: w = np.zeros(len(datas[0]) + 1)
    elif initw == 'random': w = np.random.rand(len(datas[0]) + 1)
    else: w = np.array(initw)
    print('init w is ' + str(w))

    for i in range(1,iteration):
        step0 = step / (ln(i) + 1)
        for x,t in datas:
            x = extend(x)
            if cost == 'LMS':
                delta = (w.dot(x) - t) * x
                w = w - step0 * delta # 梯度向量方向为误差上升最快方向，故取负为误差下降最快方向
            elif cost == 'LOGISTIC':
                delta = (1 / (1 + e ** w.dot(x)) if t == 1 else -1 / (1 + e ** -w.dot(x))) * x
                w = w + step0 * delta # 梯度向量方向为概率乘积上升最快方向

        if verbose == 1:
            if cost == 'LMS':
                error = sum([ 0.5 * (w.dot(extend(x)) - t) ** 2 for x,t in datas ])
            elif cost == 'LOGISTIC':
                error = sum([ ln(logistic(w.dot(extend(x)))) if t == 1 else ln(1 - logistic(w.dot(extend(x)))) for x,t in datas ])
            #error = reduce(lambda x,y: x * y, [ logistic(w.dot(extend(x))) if t == 1 else 1 - logistic(w.dot(extend(x))) for x,t in datas ])
            print('step0 is ' + str(step0) + ', delta is ' + str(delta) + ', w is ' + str(w) + \
                    ', slope is ' + str(- w[1] / w[2]) + ', error is ' + str(error))
        
    return w
