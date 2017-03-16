import numpy as np

extend = lambda x: np.array([1] + x)    # 扩展样本数据，增加常数项，并转为numpy数组
model = lambda x: np.sqrt(x.dot(x))     # 向量模长
logistic = lambda x: 1 / (1 + np.e ** -x)  # 逻辑函数

def foreach(function, *iterators):
    '''使用function迭代执行参数'''
    for args in zip(*iterators):
        function(*args)

def scatter_datas(plt, datas, c1=1, c2=-1):
    '''
    绘制两类数据点
    '''
    data1 = [ d[0] for d in datas if d[1] == c1 ]
    data2 = [ d[0] for d in datas if d[1] == c2 ]

    if len(data1) > 0:
        plt.scatter(*zip(*data1), s=25, c='g', alpha=0.5)

    if len(data2) > 0:
        plt.scatter(*zip(*data2), s=25, c='r', alpha=0.5)

def debug():
    command = input('enter to continue, q to quit: ')
    if command == 'q': exit()

def pgm_read(filepath):
    '''
    pgm格式的图片读取
    '''
    f = open(filepath, 'rb')
    try:
        pgm_type = f.readline().decode().strip()
        if pgm_type != 'P5' and pgm_type != 'P2':
            raise RuntimeError('不支持的格式')
        size = f.readline().decode().strip()
        gray = int(f.readline().decode().strip())
        data = f.read()
        data = list(data) if pgm_type == 'P5' else list(map(int, data.decode().split()))

        if gray < 1 << 8:
            return data
        elif gray < 1 << 16:
            return list(map(lambda i: (data[i*2] << 8) | data[i*2+1], range(1, len(data) / 2)))
        else:
            raise RuntimeError('不支持的灰度值：' + str(gray))
    finally:
        f.close()
