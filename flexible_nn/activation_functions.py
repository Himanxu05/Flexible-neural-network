import numpy as np

def Relu(x, alpha=0.01):
    return np.maximum(x, 0)

def Relu_dev(x, alpha=0.01):
    return x > 0

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def derived_sigmoid(x):
    y = sigmoid(x)
    return y * (1 - y)

def softmaxx(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def tanh(x):
    return np.tanh(x)

def tanh_dev(x):
    return 1 - np.tanh(x)**2

def activation(x, s):
    s = s.lower()
    if s == "relu":
        return Relu(x), Relu_dev(x)
    elif s == "sigmoid":
        return sigmoid(x), derived_sigmoid(x)
    elif s == "tanh":
        return tanh(x), tanh_dev(x)
    else:
        return Relu(x), Relu_dev