#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt 
import logging

from functools import reduce
from operator import add, mul

from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

from ml.ann import simple_back_propagation_ann, SIGMOID, LINEAR
from ml.util import scatter_datas, extend

'''
使用反向传播算法训练前馈神经网络来对数据集进行分类
'''
logging.basicConfig(level=logging.INFO, 
                    format='[%(asctime)s line:%(lineno)d %(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
output_type = LINEAR
output_type = SIGMOID   # 输出节点的类型
# 类别值，线性输出节点为(-1,1), SIGMOID输入节点为(0,1)
c1 = 1 if output_type == SIGMOID else 1
c2 = 0 if output_type == SIGMOID else -1
threshold = 0 if output_type == LINEAR else 0.5
classify = lambda x, threshold, c1, c2: c1 if x >= threshold else c2    # 分类函数

t1 = [c1]
t2 = [c2]
# 一个以直线 y=x 为界的线性可分的数据集
datas1 = [ ([x,y],t1) if x>y else ([x,y],t2) for x in range(10) for y in range(10) if x!=y ]

# 一个线性不可分的数据集，大至以y=x分隔，在直线y=x附近有线性不可分的点
datas2 = [ ([x,y],t1) if x>y else ([x,y],t2) for x in range(10) for y in range(10) if np.abs(x - y) > 1 ]
datas2.extend([([9,8],t2), ([8,9],t1), ([8,7],t1), ([7,8],t2), ([7,6],t2), ([6,7],t1), ([6,5],t1), ([5,6],t2), \
               ([5,4],t2), ([4,5],t1), ([4,3],t1), ([3,4],t2), ([3,2],t2), ([2,3],t1), ([2,1],t1), ([1,2],t2), \
               ([1,0],t2), ([0,1],t1), ])
# 一个以半径为8的1/4圆弧为分界的数据集
datas3 = [ ([x,y],t1) if x ** 2 + y ** 2 < 64 else ([x,y],t2) for x in range(10) for y in range(10) ]

datas = np.array(datas2)                # 选择数据集
#scatter_datas(plt, datas, t1, t2)      # 展示数据
#plt.show()

# 以下参数可完全拟合数据集3
#ann = simple_back_propagation_ann(datas, num_output=1, num_hidden=15, output_type=SIGMOID, iterations=100, step=0.5)
#ann = simple_back_propagation_ann(datas, num_output=1, num_hidden=15, output_type=output_type, iterations=200, step=0.5)
'''
sigmoid和softmax是神经网络输出层使用的激活函数，分别用于两类判别和多类判别。
binary cross-entropy和categorical cross-entropy是相对应的损失函数。
'''
train_datas = [ d[0] for d in datas ]
targets = np.array([ d[1][0] for d in datas ])
model = Sequential()
model.add(Dense(30, activation='relu', input_dim=2))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
#model.add(Activation('sigmoid'))
sgd = SGD(lr=0.5, decay=0.01, momentum=0)
model.compile(loss='binary_crossentropy', optimizer=sgd)

model.fit(train_datas, targets, epochs=200, verbose=1)

model.save('model/class_with_keras.h5')

model = load_model('model/class_with_keras.h5')

#results = model.evaluate(datas, targets, verbose=1)

results = model.predict_classes(train_datas)
#logging.info('results is: ')
#logging.info(results)
'''
logging.info('weights is:')
for w in model.get_weights():
    print(w)
    print('-' * 50)
'''
errors = [ (d, t) for d,t,o in zip(train_datas, targets, results) if o != t ]
num_error = len(errors)
logging.info('错误分类数量：' + str(num_error))
logging.info('错误率：' + str(num_error / len(train_datas)))
'''
classifys = [ ([classify(o, threshold, c1, c2) for o in out ], t) for out,t in results ]
logging.info('分类结果为: ')
logging.info(classifys)

# 计算误差和误差率
#error = sum([ (o - t) ** 2 for o,t in outputs ]) / 2
#logging.debug('error is: ' + str(error))        # error is: 0.172200922886

# 计算错误率
error_rate = sum([ 1 if c != t else 0 for c,t in classifys ]) / len(classifys)
logging.info('error rate is: ' + str(error_rate))
'''
# 误分类的数据
#error_classify_datas = datas[[ i for i,(c,t) in enumerate(classifys) if c != t ]]
scatter_datas(plt, errors, c1, c2)

plt.show()

import gc
gc.collect()
