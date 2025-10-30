import numpy as np
import _math
import pandas as pd
from FlexibleNN.flexible_nn.activation_functions import softmaxx, sigmoid, activation

class Layer:
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.prev = None
        self.Weights = np.random.randn(in_dim, out_dim) *  np.sqrt(1.0 / in_dim)
        self.bias = np.zeros((1, out_dim))
        self.Z = 0
        self.A = 0

        #backprop
        self.dZ = 0
        self.dW = 0
        self.dA_prev = 0
        self.dB = 0

    def forward(self, A_prev, type_of_dataset, name_of_activation, is_last = False):
        self.Z = A_prev @ self.Weights + self.bias
        if is_last:
            if type_of_dataset == 'binary':
                self.A = sigmoid(self.Z)
            elif type_of_dataset == 'multi_class_classification':
                self.A = softmaxx(self.Z)
            elif type_of_dataset == 'regression':
                self.A = self.Z
        else:
            normal, derivative_function = activation(self.Z, name_of_activation)
            self.A = normal
        return self.A

    def backprop(self, A_prev, next_dZ, next_weight, m, is_start, is_last, X, Y, learning_rate, name_of_activation, type_of_dataset):
        if is_last:
            if type_of_dataset == 'multi_class_classification':
                num_class = self.out_dim
                self.dZ = (self.A - one_hot(Y, num_class)) / m
            else:
                self.dZ = (self.A - Y) / m
            self.dW = A_prev.T @ self.dZ
            self.dB = np.sum(self.dZ, axis=0,keepdims=True)
        else:
            normal, derivated = activation(self.Z, name_of_activation)
            if is_start:
                self.dZ = next_dZ @ next_weight.T * derivated
                self.dW = X.T @ self.dZ
                self.dB = np.sum(self.dZ, axis=0, keepdims=True)
            else:
                self.dZ = next_dZ @ next_weight.T * derivated
                self.dW = A_prev.T @ self.dZ
                self.dB = np.sum(self.dZ, axis=0, keepdims=True)

        self.Weights -= self.dW * learning_rate
        self.bias -= self.dB * learning_rate

def Normalize(x, MEAN, STD):
    return (x - MEAN)/(STD + 1e-6)

def Denormalize(x, MEAN, STD):
    # x; = x - mean / std
    # x; * std = x - mean
    # x = x; * std + mean
    return (x * STD) + MEAN

def split(X, Y, test_ratio=0.2, shuffle=True):
    n = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        np.random.shuffle(indices)

    test_size = max(1, int(n * test_ratio))
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]

    x_train, x_test = X[train_idx], X[test_idx]
    y_train, y_test = Y[train_idx], Y[test_idx]
    return x_train, x_test, y_train, y_test

def load_file(path, y_col=-1, normalize=True):
    data = pd.read_csv(path)
    data = data.dropna()
    D = data.values
    X = np.delete(D, y_col, axis=1)
    Y = D[:, [y_col]]
    x_train, x_test, y_train, y_test = split(X, Y, 0.2, True)
    MEANX = np.mean(x_train, axis=0)
    STDX = np.std(x_train, axis=0)
    MEANY = np.mean(y_train, axis=0)
    STDY = np.std(y_train, axis=0)
    STDX[STDX == 0] = 1e-8
    STDY[STDY == 0] = 1e-8
    if normalize == True:
        x_train = Normalize(x_train, MEANX, STDX)
        x_test = Normalize(x_test, MEANX, STDX)
        y_train = Normalize(y_train, MEANY, STDY)
        y_test = Normalize(y_test, MEANY, STDY)
    return X, Y, x_train, x_test, y_train, y_test, MEANX, STDX, MEANY, STDY


def check_dataset(x):
    x = x.flatten()
    unique_val = np.unique(x)
    print("Unique values:", unique_val, "dtype:", unique_val.dtype)  # DEBUG

    length = len(unique_val)
    if length == 2 and set(unique_val) == {0, 1}:
        return "binary"

    elif np.all(np.floor(unique_val) == unique_val) and length > 2:
        return "multi_class_classification"
    else:
        return "regression"

def one_hot(Y, num_classes=None):
    _Y = Y.astype(int).flatten()
    n = len(_Y)
    if num_classes is None:
        num_classes = _Y.max() + 1
    ok = np.zeros((n, num_classes))
    ok[np.arange(n), _Y] = 1
    return ok

def forward_pass(layers, X, type_of_dataset, name_of_activation):
    n = len(layers)

    A=X
    for i in range(n):
        this_lay = layers[i]
        A = this_lay.forward(A, is_last=(i == n - 1), type_of_dataset=type_of_dataset, name_of_activation=name_of_activation)
    return A

def back_pass(list, X, Y, type_of_dataset, name_of_activation, learning_rate):
    n = len(list)
    p, q = X.shape
    for i in range(n - 1, -1, -1):
        this_layer = list[i]
        last_layer = list[i - 1] if i - 1 >= 0 else None
        next_layer = list[i + 1] if i + 1 < len(list) else None
        # def backprop(self, A_prev, next_dZ, next_weight, m, is_start, is_last, X, Y, learning_rate):
        A_prev = last_layer.A if last_layer is not None else X
        next_dZ = next_layer.dZ if next_layer is not None else None
        next_weight = next_layer.Weights if next_layer is not None else None


        # if we are at n  - 1 next -> NULL
        this_layer.backprop(A_prev=A_prev, next_dZ=next_dZ, next_weight=next_weight, m=p, is_start=(i == 0),
                                     is_last=(i == n - 1), X=X, Y=Y, learning_rate=learning_rate, name_of_activation=name_of_activation, type_of_dataset=type_of_dataset)


def LIST(n, X, Y, org, type_of_dataset):
    p, q = X.shape
    list = []
    first = 50
    if type_of_dataset == 'regression':
        end = 1
    elif type_of_dataset == 'binary':
        end = 1
    elif type_of_dataset == 'multi_class_classification':
        end = len(np.unique(org))
    rate = _math.AP(first, end, n)
    prev = X
    loss = None
    in_d = q
    for i in range(n):
        if i != n - 1:
            out_d = first -rate
        else:
            out_d = end
        layer = Layer(int(in_d), int(out_d))
        list.append(layer)
        in_d = int(out_d)
        first = out_d
    return list

def work(n, epos, path, name_of_activation, learning_rate, y_col):
    X, Y, x_train, x_test, y_train, y_test, MEANX, STDX, MEANY, STDY = load_file(
        path, -1, True)

    type_of_dataset = check_dataset(Y)
    # print(type_of_dataset)

    layers = LIST(n, x_train, y_train, Y, type_of_dataset)

    for i in range(epos):
        forward_pass(layers, x_train, type_of_dataset, name_of_activation)
        back_pass(layers, x_train, y_train, type_of_dataset, name_of_activation, learning_rate)

    pr = forward_pass(layers, x_test, type_of_dataset, name_of_activation)
    if type_of_dataset == 'regression':
        pr = Denormalize(pr, MEANY, STDY)
        y_test_ok = Denormalize(y_test, MEANY, STDY)
    else:
        y_test_ok = y_test
    print(evalutaion_metrics(pr, y_test_ok, type_of_dataset))
    # print(type_of_dataset)


def evalutaion_metrics(pr, Y, type_of_dataset):
    if type_of_dataset == "regression":
        return f"Mean squared error is: {np.mean((pr - Y) ** 2)}"
    elif type_of_dataset == "binary":
        ok = (pr > 0.5).astype(int)
        acc = np.mean(ok == Y)
        return f"Accuracy for your binary mode is: {acc}"
    elif type_of_dataset == "multi_class_classification":
        ok = np.argmax(pr, axis=1)
        true_class = Y.flatten().astype(int)
        acc = np.mean(true_class == ok)
        return f"accuracy of this multi class model is {acc}"