
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
