import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

'''
训练一个线性函数对数据集进行划分
设数据集为: {(x1,t1),(x2,t2), ... ,(xn,tn)}
1) 如果数据集是线性可分的
感知器法则可以正确分类所有样本，过程收敛
2) 如果数据集是线性不可分的
可以使用梯度下降算法(delta法则)找到一个最优线性函数，使得这个线性函数对整个数据集划分的效果最好
对最优的度量可以使用不同的标准
LMS最小均方差: ∑(ti-wxi)**2，是一个比较常用的误差度量方式，最优即使数据集方差最小
LOGISTIC逻辑函数，最优即使样本相应类别逻辑概率的乘积最大
'''
e = np.e
ln = np.log
extend = lambda x: np.array([1] + x)    # 扩展样本数据，增加常数项
model = lambda x: np.sqrt(x.dot(x))	# 向量模长
logistic = lambda x: 1 / (1 + e ** -x)	# 逻辑函数
classify = lambda x: 1 if x >= 0 else -1    # 分类函数

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

def grad_desent_regression(datas, iteration=100, step=1, initw=0, cost='LMS'):
    '''
    对于一个线性不可分的数据集: {(x1,t1),(x2,t2), ... ,(xn,tn)}
    找到一个线性函数，使得这个线性函数对整个数据集划分的误差最小
    cost: 误差度量方式
    LMS指最小均方差: ∑(ti-oi)**2，是一个比较常用的误差度量方式
    对数据集中的每个样本xi:
    oi = wxi
    cost = ∑(ti-oi)**2 = ∑(ti-wxi)**2
    ▽ w = ∑2(ti-wxi)xi
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

    for i in range(iteration):
        delta = np.zeros(len(datas[0]) + 1)
        step0 = step / (i * 5 + 1)
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
            error = sum([ w.dot(np.array([1] + x)) - t for x,t in datas ])
        elif cost == 'LOGISTIC':
            w = w + step0 * gradient # 梯度向量方向为样本对应类别概率乘积上升最快方向
            #error = -sum([ ln(1 + e ** -w.dot(np.array([1] + x))) if t == 1 else ln(1 + e ** w.dot(np.array([1] + x))) for x,t in datas ])
            error = reduce(lambda x,y: x * y, [ 1 / (1 + e ** -w.dot(np.array([1] + x))) if t == 1 else 1 / (1 + e ** w.dot(np.array([1] + x))) for x,t in datas ])
        print('step0 is ' + str(step0) + ', delta is ' + str(delta) + ', w is ' + str(w) + \
                ', slope is ' + str(- w[1] / w[2]) + ', error is ' + str(error))

    return w

def stoch_grad_desent_regression(datas, iteration=100, step=1, initw=0, cost='LMS'):
    '''
    随机梯度下降与梯度下降相似，
    只是对数据集中的每个样本xi，梯度向量的方向取(ti-wxi)xi，然后根据此梯度向量的方向更新w，不需要根据整个数据集样本的误差计算梯度向量
    这相当于为每个样本定义一个误差函数：cost = (ti - ti) ** 2 = (ti - wxi) ** 2 和梯度向量(ti - wxi)xi
    逻辑回归中目标函数为：-tiln(1 + e ** -wxi) - (1 - ti)ln(1 + e ** wxi)
    梯度向量为：(ti / (1 + e ** wxi) - (1 - ti) / (1 + e ** -wxi))xi
    '''
    if initw == 0: w = np.zeros(len(datas[0]) + 1)
    elif initw == 'random': w = np.random.rand(len(datas[0]) + 1)
    else: w = np.array(initw)
    print('init w is ' + str(w))

    for i in range(iteration):
        step0 = step / (i * 2 + 1)
        for x,t in datas:
            x = extend(x)
            if cost == 'LMS':
                delta = (w.dot(x) - t) * x
                w = w - step0 * delta # 梯度向量方向为误差上升最快方向，故取负为误差下降最快方向
            elif cost == 'LOGISTIC':
                delta = (1 / (1 + e ** w.dot(x)) if t == 1 else -1 / (1 + e ** -w.dot(x))) * x
                w = w + step0 * delta # 梯度向量方向为概率乘积上升最快方向
        if cost == 'LMS':
            error = sum([ 0.5 * (w.dot(extend(x)) - t) ** 2 for x,t in datas ])
        elif cost == 'LOGISTIC':
            error = sum([ ln(logistic(w.dot(extend(x)))) if t == 1 else ln(1 - logistic(w.dot(extend(x)))) for x,t in datas ])
            #error = reduce(lambda x,y: x * y, [ logistic(w.dot(extend(x))) if t == 1 else 1 - logistic(w.dot(extend(x))) for x,t in datas ])
        print('step0 is ' + str(step0) + ', delta is ' + str(delta) + ', w is ' + str(w) + \
                ', slope is ' + str(- w[1] / w[2]) + ', error is ' + str(error))
        
    return w

def scatter_datas(datas):
    # 绘制原始数据点
    data1 = [ d[0] for d in datas if d[1] == 1 ] 
    data2 = [ d[0] for d in datas if d[1] == -1 ]
    
    plt.scatter(*list(zip(*data1)), s=25, c='g', alpha=0.5)
    plt.scatter(*list(zip(*data2)), s=25, c='r', alpha=0.5)

# 一个以直线 y=x 为界的线性可分的数据集
datas1 = [ ([x,y],1) if x>y else ([x,y],-1) for x in range(10) for y in range(10) if x!=y ]
# 一个线性不可分的数据集，大至以y=x分隔，在直线y=x附近有线性不可分的点
datas2 = [ ([x,y],1) if x>y else ([x,y],-1) for x in range(10) for y in range(10) if np.abs(x - y) > 1 ]
datas2.extend([([9,8],-1), ([8,9],1), ([8,7],1), ([7,8],-1), ([7,6],-1), ([6,7],1), ([6,5],1), ([5,6],-1), \
               ([5,4],-1), ([4,5],1), ([4,3],1), ([3,4],-1), ([3,2],-1), ([2,3],1), ([2,1],1), ([1,2],-1), \
               ([1,0],-1), ([0,1],1), ])

datas = datas1                  # 在这里选择使用的数据集
w = perceptron_rule(datas)     # 这个方法要求使用线性可分的数据集
#w = grad_desent_regression(datas, iteration=10, step=0.1, cost='LOGISTIC')
#w = stoch_grad_desent_regression(datas, step=0.02, iteration=100, initw=0, cost='LMS')
scatter_datas(datas)
print('w is: ' + str(w))

# 绘制线性回归的直线
x = np.linspace(0, 10)
y = -(w[0] + w[1] * x) / w[2]

line, = plt.plot(x, y, '--', linewidth=1)
line.set_dashes([1,0])

# 计算误差和错误率
result = [ (w.dot(extend(x)),t) for x,t in datas ]
error = sum([ (o - t) ** 2 for o,t in result ]) / 2     # LMS的误差
error_rate = sum([ 1 if classify(o) != t else 0 for o,t in result ]) / len(result)
print('error is: ' + str(error))
print('error rate is: ' + str(error_rate))

'''
# 绘制逻辑回归数据点
x = [ w.dot(extend(d[0])) for d in datas ]
y = list(map(logistic, x))
print('x is: \n' + str(x))
print('y is: \n' + str(y))
plt.scatter(x, y, s=10, c='r', alpha=0.5)
'''

plt.show()


