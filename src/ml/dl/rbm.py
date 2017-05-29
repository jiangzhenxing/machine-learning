import numpy as np
from numpy import random
from ml.util import logistic

def cd_train(data, iterations=100, lr=0.01, hiddens=16):
    '''
    使用对比散度(cd)学习算法训练受限玻尔兹曼机
    '''
    v1 = np.array(data)
    visions = len(v1)   # 可见节点的数量
    # 由于array能进行logistic批量计算，所以w,a,b均使用array类型，需要的时候再转为matrix
    w = random.rand(visions, hiddens) - 0.5
    a = random.rand(visions) - 0.5
    b = random.rand(hiddens) - 0.5

    for t in range(iterations):
        # 计算隐藏层节点输出
        h1 = logistic(v1.dot(w) - b)
        # 还原v2, h2
        v2 = logistic(w.dot(h1) - a)
        h2 = logistic(v2.dot(w) - b)

        # 更新参数
        w = w + lr * np.array(np.mat(v1).transpose().dot(np.mat(h1)) - np.mat(v2).transpose().dot(np.mat(h2)))
        a = a + lr * (v1 - v2)
        b = b + lr * (h1 - h2)
        
#    print(v2)
    def out(data_):
        v1_ = np.array(data_)
        h1_ = logistic(v1_.dot(w) - b)
        v2_ = logistic(w.dot(h1_) - a)
        return v2_

    out.w = w
    out.a = a
    out.b = b

    return out
