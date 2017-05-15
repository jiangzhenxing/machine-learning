#!/usr/bin/env python
import logging
import numpy as np

from ml.ann import simple_back_propagation_ann, SIGMOID
from ml.util import pgm_read
import os

'''
利用三层人工神经网络检测人脸的朝向
'''
level = logging.DEBUG
level = logging.INFO
logging.basicConfig(level=level,
                    format='[%(asctime)s line:%(lineno)d %(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
path = 'data/faces/'
train_user = ['an2i', 'at33', 'boland', 'bpm', 'ch4f', 'cheyer', 'choon', 'danieln', 'glickman', 
              'karyadi', 'kawamura', 'kk49', 'megak', 'mitchell', 'night', 'phoebe', 'saavik', 'steffi']
test_user1 = ['sz24', 'tammo']   # 使用独立测试集数据进行测试
test_user2 = ['an2i', 'at33']    # 使用训练集中的数据进行测试

# 人脸朝向
direction = lambda name: [ 0.9 if d in name else 0.1 for d in ['_left_', '_right_', '_straight_', '_up_']]
# 读取一个用户的人脸图像数据
def read_user_data(u):
    return [([x for x in pgm_read(path + u + '/' + p)], direction(p)) for p in os.listdir(path + u) if p.endswith('_4.pgm')]

train_data = [] # 训练数据集
for u in train_user:
    train_data += read_user_data(u)

output_type = SIGMOID
classify = lambda x: 0.9 if x >= 0.5 else 0.1    # 分类函数

ann = simple_back_propagation_ann(train_data, num_output=4, num_hidden=4, output_type=output_type, iterations=100, step=1e-5)

def test(test_data):
    print('-' * 20 + ' test begin ' + '-' * 20)
    results = [ (ann(x),t) for x,t in test_data ]
    logging.info('results is: ')
    logging.info(results)
    
    # 对结果进行分类
    classifys = [ ([classify(o) for o in out], t) for out,t in results ]
    #logging.info('classifys is: ')
    #logging.info(classifys)

    # 计算错误率
    error_rate = sum([ 1 if c != t else 0 for c,t in classifys ]) / len(classifys)
    logging.info('error rate is: ' + str(error_rate))

test_data1 = []  # 测试数据
test_data2 = []  # 测试数据

for u in test_user1:
    test_data1  += read_user_data(u)
for u in test_user2:
    test_data2  += read_user_data(u)

# 准确率接近90%
test(test_data1)
test(test_data2)
