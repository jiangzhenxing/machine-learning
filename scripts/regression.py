#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from ml.ann import perceptron_rule, grad_descent_regression, stoch_grad_descent_regression
from ml.util import scatter_datas, extend
from sklearn import svm

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

classify = lambda x: 1 if x >= 0 else -1    # 分类函数

# 一个以直线 y=x 为界的线性可分的数据集
datas1 = [ ([x,y],1) if x>y else ([x,y],-1) for x in range(10) for y in range(10) if x!=y ]
# 一个线性不可分的数据集，大至以y=x分隔，在直线y=x附近有线性不可分的点
datas2 = [ ([x,y],1) if x>y else ([x,y],-1) for x in range(10) for y in range(10) if np.abs(x - y) > 1 ]
datas2.extend([([9,8],-1), ([8,9],1), ([8,7],1), ([7,8],-1), ([7,6],-1), ([6,7],1), ([6,5],1), ([5,6],-1), \
               ([5,4],-1), ([4,5],1), ([4,3],1), ([3,4],-1), ([3,2],-1), ([2,3],1), ([2,1],1), ([1,2],-1), \
               ([1,0],-1), ([0,1],1), ])
# 一个以y=x分隔的不对称点集
datas3 = [([1,10],1), 
          ([5,6],1), 
          ([1,3],1), ([2,5],1), 
          ([3,10],1), 
          ([2,1],-1), 
          ([4,2],-1), 
          ([6,5],-1), 
          ([8,4],-1), ([8,5],-1),
         ]
LGST = 'LOGISTIC'
LMS = 'LMS'

datas = datas3          # 在这里选择使用的数据集
cost = LGST            # 选择损失函数
dataslr = [(d,(l+1)/2) for d,l in datas] # 类型转化为0/1
print(dataslr)
w = grad_descent_regression(dataslr, iteration=200, step=0.1, cost=LGST)
w2 = stoch_grad_descent_regression(datas, iteration=200, step=0.01, cost=LMS, verbose=1)

data = [d for d,l in datas]
label = [l for d,l in datas]
svc = svm.SVC(kernel='linear', max_iter=200)
svc.fit(data, label)
print('support_vectors:', svc.support_vectors_)
p = [[i,i*2] for i in range(1,9,1)]
d = svc.decision_function(p)
b = [(x,y-h) for (x,y),h in zip(p,d)]
line, = plt.plot(*zip(*b), 'm--', linewidth=1)
line.set_dashes([1,0])

#w = stoch_grad_descent_regression(datas, step=0.02, iteration=100, initw=0, cost='LMS')
scatter_datas(plt, datas)
print('w is: ' + str(w))
print('w2 is:', w2)

# 绘制线性回归的直线
x = np.linspace(1, 8)
y = -(w[0] + w[1] * x) / w[2]

line, = plt.plot(x, y, 'b--', linewidth=1)
line.set_dashes([1,0])

x2 = np.linspace(1, 7)
y2 = -(w2[0] + w2[1] * x2) / w2[2]
# w1x + w2y + w0 = 1
# y = -(w1x + w0 - 1)/w2
x3 = np.linspace(0, 4)
y3 = -(w2[0] + w2[1] * x3 - 1) / w2[2]
x4 = np.linspace(5, 9)
y4 = -(w2[0] + w2[1] * x4 + 1) / w2[2]

line, = plt.plot(x2, y2, 'c--', linewidth=1)
line.set_dashes([1,0])

line, = plt.plot(x3, y3, 'c--', linewidth=1)
line, = plt.plot(x4, y4, 'c--', linewidth=1)

#line.set_dashes([1,0])

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


