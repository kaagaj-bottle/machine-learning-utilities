import numpy as np
import matplotlib.pyplot as plt


def load_data(path):
    data = np.loadtxt(path, delimiter=',')
    X = data[:, 0]
    y = data[:, 1]
    return X, y


def prediction(x, w, b):

    m = x.shape[0]
    y = np.zeros(m)

    for i in range(m):
        y[i] = w*x[i]+b

    return y


def cost_function(x, y, w, b):

    m = x.shape[0]
    cost = 0.

    for i in range(m):
        err = w*x[i]+b-y[i]
        cost += err**2

    cost /= 2*m
    return cost


def gradient_function(x, y, w, b):

    m = x.shape[0]
    dj_dw = 0.
    dj_db = 0.

    for i in range(m):
        err = w*x[i]+b-y[i]
        dj_dw += err*x[i]
        dj_db += err

    dj_dw /= 2*m
    dj_db /= 2*m

    return dj_dw, dj_db


def gradient_descent(x_train, y_train, w_init, b_init, alpha, num_iters):

    m = x_train.shape[0]
    w = w_init
    b = b_init
    cost_history = np.zeros(num_iters)

    for i in range(num_iters):
        cost_history[i] = cost_function(x_train, y_train, w, b)
        dj_dw, dj_db = gradient_function(x_train, y_train, w, b)
        w = w-alpha*dj_dw
        b = b-alpha*dj_db

    return w, b, cost_history
