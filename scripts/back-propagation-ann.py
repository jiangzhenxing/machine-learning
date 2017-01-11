import numpy as np
import matplotlib.pyplot as plt 
import logging

from functools import reduce
from operator import add
from operator import mul

'''
使用反向传播算法训练前馈神经网络
'''

e = np.e
ln = np.log
extend = lambda x: np.array([1] + x)        # 扩展样本数据，增加常数项
model = lambda x: np.sqrt(x.dot(x))         # 向量模长
# normal = lambda x: x if model(x) == 0 else x / model(x)         # 单位向量
logistic = lambda x: 1 / (1 + e ** -x)      # 逻辑函数
classify = lambda x: 1 if x >= 0 else -1    # 分类函数

logging.basicConfig(level=logging.INFO, 
                    format='[%(asctime)s line:%(lineno)d %(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')

class Node:
    '''
    神经网络的结点
    包括权重向量，输出函数，导数，陨失函数
    '''
    def __init__(self, weight, index, type='sigmoid'):
        self.weight = weight
        self.index = index
        if type == 'sigmoid':
            self.output = lambda x: logistic(self.weight.dot(x))
            self.derivative = lambda x: self.output(x) * (1 - self.output(x))
            self.cost = lambda x,t: self.output(x) if t == 1 else 1 - self.output(x)
        elif type == 'linear':
            self.output = lambda x: self.weight.dot(x)
            self.derivative = lambda x: self.weight
            self.cost = lambda o,t: 0.5 * (o - t) ** 2
            self.cost_derivative = lambda o,t: o - t

    def update_weight(self, delta):
        self.weight += delta


def simple_back_propagation_ann(datas, num_output=3, num_hidden=5, iterations=100, step=1):
    '''
    一个使用反向传播算法训练的简单的前馈神经网络
    num_output: 输出结点的数量
    num_hidden: 隐藏结点的数量
    隐藏结点为sigmoid结点
    输出结点为普通线性输出单元: out = wx
    误差函数为LMS函数: 0.5 * ∑ (o - t) ** 2
    '''
    # 输出结点
    outputs = [ Node(weight=np.random.rand(num_hidden + 1), index=i, type='linear') for i in range(num_output) ]
    # 隐藏结点
    hiddens = [ Node(weight=np.random.rand(len(datas[0]) + 1), index=i, type='sigmoid') for i in range(num_hidden) ]

    for i in range(iterations):
        step0 = step / (i * 2 + 1)
        for x,t in datas:
            x = extend(x)
            logging.debug('x is: ' + str(x) + ', t is: ' + str(t))

            # 计算隐藏结点的输出
            out_hiddens = extend([ node.output(x) for node in hiddens ])
            logging.debug('out_hiddens: ' + str(out_hiddens))

            # 计算输出结点的输出
            outs = [ node.output(out_hiddens) for node in outputs ]
            logging.debug('outs: ' + str(outs))

            # 计算输出单元的梯度向量
            gradient_outputs = [ (outs[node.index] - t) * out_hiddens for node in outputs ]
            logging.debug('gradient_outputs: ' + str(gradient_outputs))

            # 输出结点损失函数关于隐藏结点输出的导数
            d_cost_hidden = [ node.cost_derivative(outs[node.index], t) * node.derivative(out_hiddens)[1:] for node in outputs ]
            logging.debug('d_cost_hidden: ' + str(d_cost_hidden))

            # 隐藏结点关于输入权重的导数
            d_hidden_weight = [ node.derivative(x) * x for node in hiddens ]
            logging.debug('d_hidden_weight: ' + str(d_hidden_weight))

            # 陨失函数(各输出结点关于某隐藏结点陨失之和)关于输入权重的导数(与该隐藏结点关于输入权重的导数相乘)
            gradient_hiddens = list(map(mul, reduce(add, d_cost_hidden), d_hidden_weight))
            logging.debug('gradient_hiddens: ' + str(gradient_hiddens))

            # 更新权值向量(方向取梯度向量的反方向为误差下降最快方向)
            foreach(Node.update_weight, outputs, -step0 * np.array(gradient_outputs))
            foreach(Node.update_weight, hiddens, -step0 * np.array(gradient_hiddens))

            '''
            for gradient, node in zip(gradient_outputs, outputs):
                node.update_weight(step0 * -gradient)

            for gradient, node in zip(gradient_hiddens, hiddens):
                node.update_weight(step0 * -gradient)
            '''

        logging.info('step is: ' + str(step0))
        logging.info('out weight is: ' + str([ node.weight for node in outputs ]))
        logging.info('hidden weight is: ' + str([ node.weight for node in hiddens ]))

    # 神经网络的输出函数
    def output(x):
        x = extend(x)
        out_hiddens = extend([ node.output(x) for node in hiddens ])
        outs = [ node.output(out_hiddens) for node in outputs ]
        return outs

    return output

# 对多个输出结点的输出进行汇总输出
def classify_outputs(outs):
    return classify(sum(outs))

def foreach(function, *iterators):
    for args in zip(*iterators):
        function(*args)

def scatter_datas(datas):
    # 绘制原始数据点
    data1 = [ d[0] for d in datas if d[1] == 1 ]
    data2 = [ d[0] for d in datas if d[1] == -1 ]

    if len(data1) > 0:
        plt.scatter(*list(zip(*data1)), s=25, c='g', alpha=0.5)

    if len(data2) > 0:
        plt.scatter(*list(zip(*data2)), s=25, c='r', alpha=0.5)

# 一个以直线 y=x 为界的线性可分的数据集
datas1 = [ ([x,y],1) if x>y else ([x,y],-1) for x in range(10) for y in range(10) if x!=y ]
# 一个线性不可分的数据集，大至以y=x分隔，在直线y=x附近有线性不可分的点
datas2 = [ ([x,y],1) if x>y else ([x,y],-1) for x in range(10) for y in range(10) if np.abs(x - y) > 1 ]
datas2.extend([([9,8],-1), ([8,9],1), ([8,7],1), ([7,8],-1), ([7,6],-1), ([6,7],1), ([6,5],1), ([5,6],-1), \
               ([5,4],-1), ([4,5],1), ([4,3],1), ([3,4],-1), ([3,2],-1), ([2,3],1), ([2,1],1), ([1,2],-1), \
               ([1,0],-1), ([0,1],1), ])

datas = np.array(datas2)                    # 选择数据集
ann = simple_back_propagation_ann(datas, num_output=3, num_hidden=5, iterations=100, step=0.1)
results = [ (ann(x),t) for x,t in datas ]
logging.info('results is: ')
logging.info(results)

classifys = [ (classify_outputs(out),t) for out,t in results ]
logging.info('分类结果为: ')
logging.info(classifys)

# 计算误差和误差率
#error = sum([ (o - t) ** 2 for o,t in outputs ]) / 2
#logging.debug('error is: ' + str(error))        # error is: 0.172200922886

# 计算错误率
error_rate = sum([ 1 if c != t else 0 for c,t in classifys ]) / len(classifys)
logging.info('error rate is: ' + str(error_rate))

# 误分类的数据
error_classify_datas = datas[[ i for i,(c,t) in enumerate(classifys) if c != t ]]
scatter_datas(error_classify_datas)
# scatter_datas(datas)
plt.show()

