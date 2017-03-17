import numpy as np
import matplotlib.pyplot as plt 
import logging

from functools import reduce
from operator import add, mul
from ml.util import foreach, extend, model, logistic, debug
from . import SIGMOID, LINEAR

'''
使用反向传播算法训练前馈神经网络
'''

e = np.e
ln = np.log
classify = lambda score, threshold, c1, c2: c1 if score >= threshold else c2    # 分类函数

class Node:
    '''
    神经网络的结点
    包括权重向量，输出函数，导数，陨失函数及其导数
    '''
    def __init__(self, weight, index, tp=SIGMOID):
        self.weight = weight
        self.index = index
        self.tp = tp
        self.out = 0

        if tp == SIGMOID:
            # 输出函数
            self.output = lambda x: logistic(self.weight.dot(x))
            # 输出关于输入的导数
            self.derivative = lambda x: self.out * (1 - self.out) * self.weight
            # 输出关于权重向量的导数
            self.derivative_w = lambda x: self.out * (1 - self.out) * x
            # 陨失函数
            self.cost = lambda x,t: self.out if t == 1 else 1 - self.out
            # 损失关于输入的导数
            # self.cost_derivative = lambda o,t: t * (1 - o) - (1 - t) * o
            self.cost_derivative = lambda x,t: (t * (1 - self.out) - (1 - t) * self.out) * self.weight
            # 损失关于权重的导数
            self.cost_derivative_w = lambda x,t: (t * (1 - self.out) - (1 - t) * self.out) * x
        elif tp == LINEAR:
            # 输出函数
            self.output = lambda x: self.weight.dot(x)
            # 输出关于输入的导数
            self.derivative = lambda x: self.weight
            # 输出关于权重向量的导数
            self.derivative_w = lambda x: x
            # 陨失函数
            self.cost = lambda o,t: 0.5 * (self.out - t) ** 2
            # self.cost_derivative = lambda o,t: o - t
            # 损失关于输入的导数
            self.cost_derivative = lambda x,t: (self.out - t) * self.weight
            # 损失关于权重的导数
            self.cost_derivative_w = lambda x,t: (self.out - t) * x

    def update_weight(self, delta):
        self.weight += delta

def simple_back_propagation_ann(datas, num_output=1, num_hidden=5, output_type=LINEAR, iterations=100, step=1, step_out=1):
    '''
    一个使用反向传播算法训练的简单的三层前馈神经网络
    包含一个输出层，一个隐藏层和一个输入层
    num_output: 输出结点的数量
    num_hidden: 隐藏结点的数量
    隐藏结点为sigmoid结点
    输出结点为普通线性输出单元: out = wx
    误差函数为LMS函数: 0.5 * ∑ (o - t) ** 2
    '''
    # 输出结点
    outputs = [ Node(weight=np.random.rand(num_hidden + 1) - 0.5, index=i, tp=output_type) for i in range(num_output) ]
    # 隐藏结点
    hidden_type = SIGMOID
    hiddens = [ Node(weight=np.zeros(len(datas[0][0]) + 1), index=i, tp=hidden_type) for i in range(num_hidden) ]

    # 权重更新的方向
    # 如果输出结点是线性单元，取梯度向量相反的方向为误差下降最快方向
    # 如果输出单元是sigmoid单元，取梯度向量方向为目标值上升最快方向
    update_direct = 1 if output_type == SIGMOID else -1
    step_delta_out = step_out / (iterations + 1)    # 输出节点权重调整步长每轮迭代减少的量
    step_delta = step / (iterations + 1)            # 隐藏节点权重调整步长每轮迭代减少的量

    for i in range(iterations):
        step_out = step_out - step_delta_out
        step = step - step_delta
        for x,t in datas:
            if len(t) != num_output:
                raise RuntimeError('目标数量与输出节点数量不同')
            x = extend(x)
            logging.debug('x is: ' + str(x[:10]) + ', t is: ' + str(t))
            logging.debug('out weight is: ' + str([ node.weight[:10] for node in outputs ]))
            logging.debug('hidden weight is: ' + str([ node.weight[:10] for node in hiddens ]))

            # 计算隐藏结点的输出
            for node in hiddens: node.out = node.output(x)
            out_hiddens = extend([ node.output(x) for node in hiddens ])
            logging.debug('out_hiddens: ' + str(out_hiddens))

            # 计算输出结点的输出
            for node in outputs: node.out = node.output(out_hiddens)
            outs = [ node.output(out_hiddens) for node in outputs ]
            logging.debug('outs: ' + str(outs))

            # 计算输出单元损失的梯度向量
            gradient_outputs = [ node.cost_derivative_w(out_hiddens, target) for (node,target) in zip(outputs,t) ]
            logging.debug('gradient_outputs: ' + str(gradient_outputs[:10]))

            # 输出结点损失函数关于隐藏结点输出的导数
            d_cost_hidden = [ node.cost_derivative(out_hiddens, target)[1:] for node,target in zip(outputs,t) ]
            logging.debug('d_cost_hidden: ' + str(d_cost_hidden[:10]))

            # 隐藏结点关于输入权重的导数
            d_hidden_weight = [ node.derivative_w(x) for node in hiddens ]
            logging.debug('d_hidden_weight: ' + str([d[:10] for d in d_hidden_weight]))

            # 陨失函数(各输出结点关于某隐藏结点陨失之和)关于输入权重的导数(与该隐藏结点关于输入权重的导数相乘)
            gradient_hiddens = list(map(mul, reduce(add, d_cost_hidden), d_hidden_weight))
            logging.debug('gradient_hiddens: ' + str([g[:10] for g in gradient_hiddens]))

            # 更新权值向量
            # LMS误差度量方式时，方向取梯度向量的反方向即误差下降最快方向
            # 输出单元为sigmoid单元时，方向取梯度向量方向即可，为目标上升最快方向
            foreach(Node.update_weight, outputs, update_direct * step_out * np.array(gradient_outputs))
            foreach(Node.update_weight, hiddens, update_direct * step * np.array(gradient_hiddens))

            '''
            for gradient, node in zip(gradient_outputs, outputs):
                node.update_weight(step0 * -gradient)

            for gradient, node in zip(gradient_hiddens, hiddens):
                node.update_weight(step0 * -gradient)
            '''
            if logging.root.isEnabledFor(logging.DEBUG):
                debug()

        logging.info('step is: ' + str(step))
        logging.info('out weight is: ' + str([ node.weight[:10] for node in outputs ]))
        logging.info('hidden weight is: ' + str([ node.weight[:10] for node in hiddens ]))
        logging.info('=' * 50)

    # 神经网络的输出函数
    def output(x):
        x = extend(x)
        out_hiddens = extend([ node.output(x) for node in hiddens ])
        outs = [ node.output(out_hiddens) for node in outputs ]
        return outs
    output.hiddens = hiddens
    output.outputs = outputs

    return output
