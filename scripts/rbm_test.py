#!/usr/bin/env python3
import numpy as np
from  ml.dl import rbm

'''
使用手写数字图片数据对RBM进行测试
'''
def img2vector(filename):
    '''
    读取图片
    '''
    img = [] 
    for line in open(filename):
        img.extend([int(c) for c in line.strip()])
    return np.array(img)

def printImg(imgvector):
    '''
    打印图片，为使图片看起来方正一些，对所有数据都打印了两次
    '''
    for n in range(32):
        print(''.join([ str(int(n))*2 for n in imgvector[n * 32: n * 32 + 32] ]))

# 按结果概率进行还原
classify = lambda v: [1 if x >= 0.5 else 0 for x in v]

img = img2vector('data/digits/2_1.txt')
print('训练图像：')
printImg(img)

rbm_out = rbm.cd_train(img, iterations=200, hiddens=64)     # 隐藏结点数量不够多，还原效果不好

img2 = img2vector('data/digits/2_2.txt')
#img2 = np.round(np.random.rand(1024))     # 随机数据不能还原原始图像
#img2 = list(np.round(np.random.rand(32))) * 32
print('测试图像：')
printImg(img2)

n = 1
for i in range(n):
    img2 = rbm_out(img2)
    result = classify(img2)
    print('rbm还原结果：' + str(i + 1))
    printImg(result)

'''
print('### w: ###')
print(rbm_out.w)
print('### a: ###')
print(rbm_out.a)
print('### b: ###')
print(rbm_out.b)
'''
