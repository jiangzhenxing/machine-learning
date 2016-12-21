import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

'''
使用梯度下降算法训练一个线性函数对数据集进行划分
设数据集为: 
{(x1,o1),(x2,o2), ... ,(xn,on)}
1) 如果数据集是线性可分的
梯度下降算法可以正确分类所有样本，过程收敛
2) 如果数据集是线性不可分的
可以找到一个线性函数，使得这个线性函数对整个数据集划分的误差最小
对误差的度量可以使用不同的标准
LMS最小均方差: ∑(oi-ti)**2，是一个比较常用的误差度量方式
LOGISTIC逻辑函数，逻辑回归中使用样本相应逻辑概率的乘积来定义误差: ∏Pi
'''

def grad_des_line_separate(datas, step=0.1):
	'''
	对于一个线性可分的数据集: {(x1,o1),(x2,o2), ... ,(xn,on)}
	ti = sign(wxi)
	∆w = λ(oi - ti)xi
	w = w + ∆w
	循环这个过程至所有数据均正确分类，w不再改变，过程收敛
	'''
	w = np.random.rand(3)
	print('init w is: ' + str(w))
	for i in range(10000):
		raw_w = np.array(w)
		for x,o in datas:
			x = np.array([1] + x)
			t = np.sign(w.dot(x))
#			print('o:' + str(o) + ' t:' + str(t) + ' x:' + str(x))
			dw = (o - t) * x
#			print('dw: ' + str(dw))
			w = w + step * dw
		if (w == raw_w).all():
			break
	return w

def grad_desent_regression(datas, iteration=100, step=1, initw=0, cost='LMS'):
	'''
	对于一个线性不可分的数据集: {(x1,o1),(x2,o2), ... ,(xn,on)}
	找到一个线性函数，使得这个线性函数对整个数据集划分的误差最小
	cost: 误差度量方式
	LMS指最小均方差: ∑(oi-ti)**2，是一个比较常用的误差度量方式
	对数据集中的每个样本xi:
	ti = wxi
	cost = ∑(oi-ti)**2 = ∑(oi-wxi)**2
	▽ w = ∑2(oi-wxi)xi
	w = w + λ▽ w
	重复这个过程至w收敛
	LOGISTIC指逻辑函数，逻辑回归中使用样本相应逻辑概率的乘积来定义误差: ∏Pi
	Pi = 1 / (1 + e**(-wxi))
	cost = ln(∏Pi**oi(1-Pi)**(1-oi)) = ∑ln(Pi**oi(1-Pi)**(1-oi)) = -∑[oiln(1+e**(-wxi)) + (1-oi)ln(1+e**wxi)]
	▽ w = ∑[oi/(1+e**wxi) - (1-oi)/(1+e**(-wxi))]xi
	'''
	if initw == 0: w = np.zeros(len(datas[0]) + 1)
	elif initw == 'random': w = np.random.rand(len(datas[0]) + 1)
	else: w = np.array(w)
	print('init w is ' + str(w))

	e = np.e
	ln = np.log

	for i in range(iteration):
		delta = np.zeros(len(datas[0]) + 1)
		step0 = step / (i * 2 + 1)
		for x,o in datas:
			x = np.array([1] + x)
			if cost == 'LMS':
				delta += (w.dot(x) - o) * x
			elif cost == 'LOGISTIC':
				if o == -1: o = 0
				delta += (o / (1 + e ** w.dot(x)) - (1 - o) / (1 + e ** -w.dot(x))) * x
		gradient = delta / np.sqrt(delta.dot(delta)) # 单位梯度向量

		if cost == 'LMS':
			w = w - step0 * gradient # 梯度向量方向为误差上升最快方向，故取负为误差下降最快方向
			error = sum([ w.dot(np.array([1] + x)) - o for x,o in datas ])
		elif cost == 'LOGISTIC':
			w = w + step0 * gradient # 梯度向量方向为样本对应类别概率乘积上升最快方向
			#error = -sum([ ln(1 + e ** -w.dot(np.array([1] + x))) if o == 1 else ln(1 + e ** w.dot(np.array([1] + x))) for x,o in datas ])
			error = reduce(lambda x,y: x * y, [ 1 / (1 + e ** -w.dot(np.array([1] + x))) if o == 1 else 1 / (1 + e ** w.dot(np.array([1] + x))) for x,o in datas ])
		print('step0 is ' + str(step0) + ', delta is ' + str(delta) + ', w is ' + str(w) + \
				', slope is ' + str(- w[1] / w[2]) + ', error is ' + str(error))

	return w

def stoch_grad_desent_regression(datas, iteration=100, step=1, initw=0, cost='LMS'):
	'''
	随机梯度下降与梯度下降相似，
	只是对数据集中的每个样本xi，梯度向量的方向取(oi-wxi)xi，
	然后根据此梯度向量的方向更新w，不需要根据整个数据集样本的误差计算梯度向量
	相当于为每个样本定义一个误差函数：cost = (oi-ti)**2 = (oi-wxi)**2 和梯度向量
	'''
	if initw == 0: w = np.zeros(len(datas[0]) + 1)
	elif initw == 'random': w = np.random.rand(len(datas[0]) + 1)
	else: w = np.array(w)
	print('init w is ' + str(w))

	for i in range(iteration):
		step0 = step / (i * 2 + 1)
		for x,o in datas:
			x = np.array([1] + x)
			delta = (w.dot(x) - o) * x 
			gradient = delta / np.sqrt(delta.dot(delta)) # 单位梯度向量
			w = w - step0 * gradient # 梯度向量方向为误差上升最快方向，故取负为误差下降最快方向
		error = sum([ w.dot(np.array([1] + x)) - o for x,o in datas ])
		print('step0 is ' + str(step0) + ', delta is ' + str(delta) + ', w is ' + str(w) + \
				', slope is ' + str(- w[1] / w[2]) + ', error is ' + str(error))

	return w

# 一个以直线 y=x 为界的线性可分的数据集
datas = [ ([x,y],1) if x>y else ([x,y],-1) for x in range(10) for y in range(10) if x!=y ]
data1 = [ d[0] for d in datas if d[1] == 1 ]
data2 = [ d[0] for d in datas if d[1] == -1 ]

x,y = list(zip(*data1))
colors = np.full(len(x), 'g', dtype=np.str)
plt.scatter(x, y, s=25, c=colors, alpha=0.5)

x,y = list(zip(*data2))
colors = np.full(len(x), 'r', dtype=np.str)
plt.scatter(x, y, s=25, c=colors, alpha=0.5)

#w = grad_des_line_separate(datas)
w = grad_desent_regression(datas, iteration=250, cost='LOGISTIC')
#w = stoch_grad_desent_regression(datas, iteration=100)
print('w is: ' + str(w))
x = np.linspace(0, 10)
y = -(w[0] + w[1] * x) / w[2]

line, = plt.plot(x, y, '--', linewidth=1)
line.set_dashes([1,0])

plt.show()


