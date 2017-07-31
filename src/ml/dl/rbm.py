import numpy as np
from numpy import random
from ml.util import logistic


def round(p):
    t = np.random.rand(len(p))
    return np.array([int(b) for b in p > t])

#round = np.round

def cd_train(data, iterations=100, lr=0.01, hiddens=16):
    '''
    使用对比散度(cd)学习算法训练受限玻尔兹曼机
    具体算法请参见论文：《受限波尔兹曼机简介》（张春霞，姬楠楠，王冠伟）
    '''
    visions = len(data)   # 可见节点的数量
    # 由于array能进行logistic批量计算，所以w,a,b均使用array类型，需要的时候再转为matrix
    w = random.rand(visions, hiddens) - 0.5
    a = random.rand(visions) - 0.5
    b = random.rand(hiddens) - 0.5

    v1 = np.array(data)

    for t in range(iterations):
        # 计算隐藏层节点输出
        ph1 = logistic(v1.dot(w) - b)
        h1 = round(ph1)

        # 还原v2
        pv2 = logistic(w.dot(h1) - a)
        v2 = round(pv2)

        # 还原h2
        ph2 = logistic(v2.dot(w) - b)
        h2 = round(ph2)

        # 更新参数
        w = w + lr * np.array(np.mat(v1).transpose().dot(np.mat(ph1)) - np.mat(v2).transpose().dot(np.mat(ph2)))
        a = a + lr * (v1 - v2)
        b = b + lr * (ph1 - ph2)

    def out(data_):
        v1_ = np.array(data_)
        h1_ = round(logistic(v1_.dot(w) - b))
        v2_ = round(logistic(w.dot(h1_) - a))
        return v2_

    out.w = w
    out.a = a
    out.b = b

    return out

