import numpy as np
from FlexibleNN.flexible_nn.main import work

name_of_activation = input('Enter activation function\n')
n, epos, y_col = map(int, input('Enter number of layers, epochs, y column\n').split())
learning_rate = float(input('Enter learning rate for Binary and multi_class_classification\n'))

path = '/home/hx/PyCharmMiscProject/FlexibleNN/datasets/Score.csv'
for i in range(10):
    np.random.seed(i)
    work(n, epos, path,  name_of_activation,learning_rate, y_col)
