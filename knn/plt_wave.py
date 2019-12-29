import matplotlib
import matplotlib.pyplot as plt
from numpy import *

def DisplayWave(arrays, type):
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)             #创建1个框图
    ax.scatter(arrays[:,1],arrays[:,2], 15.0*array(type), 15.0*array(type))         #绘图，使用array创建了一个储存分类的列表，在画点的时候对应1/2/3*15的颜色和大小
    ax = fig.add_subplot(2,1,2)
    ax.scatter(arrays[:,1], arrays[:,0], 15.0 * array(type), 15.0 * array(type))
    plt.show()

