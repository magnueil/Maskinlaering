import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def loadTableFromFile(fileName = 'length_weight.csv'):
    with open(fileName,'r') as f:
        f.readline()
        col = f.readline().split(',')
        for line in f:
            col.append(line.split(',')[0])
            col.append(line.split(',')[1])
    for x in range(0, len(col)):
        col[x] = float(col[x])
    return col

def make2dModel(x_train, y_train):
    fig, ax = plt.subplots()

    # Observed/training input and output
    #x_train = np.mat([[1], [1.5], [2], [3], [4], [5], [6]])
    #y_train = np.mat([[5], [3.5], [3], [4], [3], [1.5], [2]])

    ax.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    class LinearRegressionModel:
        def __init__(self, W, b):
            self.W = W
            self.b = b

        # Predictor
        def f(self, x):
            return x * self.W + self.b

        # Uses Mean Squared Error
        def loss(self, x, y):
            return np.mean(np.square(self.f(x) - y))

    model = LinearRegressionModel(np.mat([[-0.5074938]]), np.mat([[4.7599163]]))

    x = np.mat([[np.min(x_train)], [np.max(x_train)]])
    ax.plot(x, model.f(x), label='$y = f(x) = xW+b$')

    print('loss:', model.loss(x_train, y_train))

    ax.legend()
    plt.show()

#2d model
if True:
    t = loadTableFromFile(fileName='length_weight.csv')
    #separate columns
    t1, t2 = [], []
    for x in range(0,len(t), 2):
        t1.append(t[x])
    for x in range(1,len(t),2):
        t2.append(t[x])

    make2dModel(t1,t2)

#make2dModel()