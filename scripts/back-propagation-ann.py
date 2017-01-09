import numpy as np
import matplotlib.pyplot as plt 
from functools import reduce

'''
使用反向传播算法训练前馈神经网络
'''

e = np.e
ln = np.log
extend = lambda x: np.array([1] + x)        # 扩展样本数据，增加常数项
model = lambda x: np.sqrt(x.dot(x))         # 向量模长
normal = lambda x: x if model(x) == 0 else x / model(x)         # 单位向量
logistic = lambda x: 1 / (1 + e ** -x)      # 逻辑函数
classify = lambda x: 1 if x >= 0 else -1    # 分类函数

def simple_back_propagation_ann(datas, iterations=200, step=1):
    '''
    一个使用反向传播算法训练的简单的前馈神经网络
    网络中只包含一个输出结点，两个隐藏结点和一个输入结点
    隐藏结点为sigmoid结点
    输出结点为普通线性输出单元:out=wx
    '''
    num_hidden = 5
    wo = np.random.rand(num_hidden + 1)                                                 # 输出单元的初权重向量
    whs = np.array([ np.random.rand(len(datas[0]) + 1) for i in range(num_hidden) ])    # 隐藏结点的初始权重向量

    for i in range(iterations):
        step0 = step / (i * 2 + 1)
        for x,t in datas:
            x = extend(x)
            # 计算隐藏结点的输出
            ohs = [ logistic(wh.dot(x)) for wh in whs ]
            ohs = extend(ohs)
            # 计算输出结点的输出
            out = wo.dot(ohs)

            # 计算梯度向量
            gradient_wo = normal((out - t) * ohs)
            gradient_wh = [ normal(wo[i] * (out - t) * oh * (1 - oh) * x) for i,oh in enumerate(ohs) if i > 0 ] # 跳过第一个常数项

            # 更新权值向量
            wo += step0 * -gradient_wo
            whs = whs + step0 * -np.array(gradient_wh)    # 梯度向量为函数值上升最快的方向，反方向即为下降最快的方向

        print('step is: ' + str(step0) + 'wo is: ' + str(wo) + ', wh is: ' + str(whs))

    # 神经网络的输出函数
    def output(x):
        x = extend(x)
        ohs = [ logistic(wh.dot(x)) for wh in whs ]
        out = wo.dot(extend(ohs))
        return out

    return output


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

datas = np.array(datas2)      # 选择数据集
ann = simple_back_propagation_ann(datas)
outputs = [ (ann(x),t) for x,t in datas ]

# 计算误差和误差率
error = sum([ (o - t) ** 2 for o,t in outputs ]) / 2
print('error is: ' + str(error))        # error is: 0.172200922886

error_rate = sum([ 1 if classify(o) != t else 0 for o,t in outputs ]) / len(outputs)
print('error rate is: ' + str(error_rate))

error_classify_datas = datas[[ i for i,(o,t) in enumerate(outputs) if classify(o) != t ]]
scatter_datas(error_classify_datas)
# scatter_datas(datas)
plt.show()

