import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

'''
使用梯度下降算法训练一个线性函数对数据集进行划分
设数据集为:
{(x1,t1),(x2,t2), ... ,(xn,tn)}
模型输出: oi = wxi
1) 如果数据集是线性可分的
梯度下降算法可以正确分类所有样本，过程收敛
2) 如果数据集是线性不可分的
可以找到一个最优线性函数，使得这个线性函数对整个数据集划分的效果最好
对最优的度量可以使用不同的标准
LMS最小均方差: ∑(ti-oi)**2，是一个比较常用的误差度量方式，最优即使数据集方差最小
LOGISTIC逻辑函数，最优即使样本相应类别逻辑概率的乘积最大
'''
e = np.e
ln = np.log

def grad_des_line_separate(datas, step=0.1):
    '''
    对于一个线性可分的数据集: {(x1,t1),(x2,t2), ... ,(xn,tn)}
    oi = sign(wxi)
    ∆w = λ(ti - oi)xi
    w = w + ∆w
    循环这个过程至所有数据均正确分类，w不再改变，过程收敛
    '''
    w = np.random.rand(3)
    print('init w is: ' + str(w))
    for i in range(10000):
        raw_w = np.array(w)
        for x,t in datas:
            x = np.array([1] + x)
            o = np.sign(w.dot(x))
#            print('o:' + str(o) + ' t:' + str(t) + ' x:' + str(x))
            dw = (t - o) * x
#            print('dw: ' + str(dw))
            w = w + step * dw
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
    ti = wxi
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
            if cost == 'LMS':
                delta += (w.dot(x) - t) * x
            elif cost == 'LOGISTIC':
                if t == -1: t = 0
                delta += (t / (1 + e ** w.dot(x)) - (1 - t) / (1 + e ** -w.dot(x))) * x
        if model(delta) < 1e-20: continue
        gradient = normal(delta)     # 单位梯度向量

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
                if model(delta) < 1e-20: continue
                w = w - step0 * normal(delta) # 梯度向量方向为误差上升最快方向，故取负为误差下降最快方向
            elif cost == 'LOGISTIC':
                delta = (1 / (1 + e ** w.dot(x)) if t == 1 else -1 / (1 + e ** -w.dot(x))) * x
                if model(delta) < 1e-20: continue
                w = w + step0 * normal(delta) # 梯度向量方向为概率乘积上升最快方向
        if cost == 'LMS':
            error = sum([ w.dot(extend(x)) - t for x,t in datas ])
        elif cost == 'LOGISTIC':
            error = sum([ ln(logistic(w.dot(extend(x)))) if t == 1 else ln(1 - logistic(w.dot(extend(x)))) for x,t in datas ])
            #error = reduce(lambda x,y: x * y, [ logistic(w.dot(extend(x))) if t == 1 else 1 - logistic(w.dot(extend(x))) for x,t in datas ])
        print('step0 is ' + str(step0) + ', delta is ' + str(delta) + ', w is ' + str(w) + \
                ', slope is ' + str(- w[1] / w[2]) + ', error is ' + str(error))
        
    return w

extend = lambda x: np.array([1] + x)
# 向量模长
model = lambda x: np.sqrt(x.dot(x))
# 单位向量
normal = lambda x: x / model(x)
# 逻辑函数
logistic = lambda x: 1 / (1 + e ** -x)

# 一个以直线 y=x 为界的线性可分的数据集
#datas = [ ([x,y],1) if x>y else ([x,y],-1) for x in range(10) for y in range(10) if x!=y ]
# 一个线性不可分的数据集，大至以y=x分隔，在直线y=x附近有线性不可分的点
datas = [ ([x,y],1) if x>y else ([x,y],-1) for x in range(10) for y in range(10) if np.abs(x - y) > 1 ]
datas.extend([([9,8],-1), ([8,9],1), ([8,7],1), ([7,8],-1), ([7,6],-1), ([6,7],1), ([6,5],1), ([5,6],-1), \
              ([5,4],-1), ([4,5],1), ([4,3],1), ([3,4],-1), ([3,2],-1), ([2,3],1), ([2,1],1), ([1,2],-1), ([1,0],-1), ([0,1],1),])
#w = grad_des_line_separate(datas)  # 这个方法要求使用线性可分的数据集
#w = grad_desent_regression(datas, iteration=10, cost='LOGISTIC')
w = stoch_grad_desent_regression(datas, iteration=200, cost='LOGISTIC')
print('w is: ' + str(w))


# 绘制原始数据点
data1 = [ d[0] for d in datas if d[1] == 1 ] 
data2 = [ d[0] for d in datas if d[1] == -1 ]

colors = np.full(len(data1), 'g', dtype=np.str)
plt.scatter(*list(zip(*data1)), s=25, c=colors, alpha=0.5)

colors = np.full(len(data2), 'r', dtype=np.str)
plt.scatter(*list(zip(*data2)), s=25, c=colors, alpha=0.5)

# 绘制线性回归的直线
x = np.linspace(0, 10)
y = -(w[0] + w[1] * x) / w[2]

line, = plt.plot(x, y, '--', linewidth=1)
line.set_dashes([1,0])
'''
# 绘制逻辑回归函数图
x = [ w.dot(extend(d[0])) for d in datas ]
y = list(map(logistic, x))
print('x is: \n' + str(x))
print('y is: \n' + str(y))
plt.scatter(x, y, s=10, c='r', alpha=0.5)
'''

plt.show()


