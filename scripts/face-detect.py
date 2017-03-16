#!/usr/bin/env python
import logging

from ml.ann import simple_back_propagation_ann, SIGMOID
from ml.util import pgm_read
import os

'''
利用三层人工神经网络检测人脸的朝向
'''
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s line:%(lineno)d %(levelname)s] %(message)s',
                    datefmt='%H:%M:%S')
path = 'data/faces/'
train_user = ['an2i', 'at33', 'boland', 'bpm', 'ch4f', 'cheyer', 'choon', 'danieln', 'glickman', 'karyadi', 'kawamura', 'kk49', 'megak', 'mitchell', 'night', 'phoebe', 'saavik', 'steffi']
test_user = ['sz24', 'tammo']

# 人脸朝向
direction = lambda name: [ 1 if d in name else 0 for d in ['_left_', '_right_', '_straight_', '_up_']]
# 读取一个用户的人脸图像数据
def read_user_data(u):
    return [(pgm_read(path + u + '/' + p), direction(p)) for p in os.listdir(path + u) if p.endswith('_4.pgm')]

train_data = [] # 训练数据集
test_data = []  # 测试数据集

for u in train_user:
    train_data += read_user_data(u)

for u in test_user:
    test_data  += read_user_data(u)

output_type = SIGMOID
classify = lambda x: 1 if x >= 0.5 else 0    # 分类函数

ann = simple_back_propagation_ann(train_data, num_output=4, num_hidden=5, output_type=output_type, iterations=100, step=1)

results = [ (ann(x),t) for x,t in test_data ]
logging.info('results is: ')
logging.info(results)

# 对结果进行分类
classifys = [ ([classify(o) for o in out], t) for out,t in results ]
# 计算错误率
error_rate = sum([ 1 if c != t else 0 for c,t in classifys ]) / len(classifys)
logging.info('error rate is: ' + str(error_rate))
