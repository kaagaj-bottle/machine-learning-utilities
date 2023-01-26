import numpy as np


def prediction(x, w, b):

    m = x.shape[0]
    y = np.zeros(m)

    for i in range(m):
        y[i] = 1/(1+np.exp(-w*x[i]-b))

    return y


def sigmoid_function(x, w, b):
    m = x.shape[0]
    g = np.zeros(m)
    z = np.zeros(m)

    z = w*x+b
    g = 1/(1+np.exp(-z))

    return g, z


def gradient_function(x, y, w, b):

    m = x.shape[0]
    dj_dw = 0.
    dj_db = 0.

    for i in range(m):
        z = w*x[i]+b
        f_wb = 1/(1+np.exp(-z))
        err = y[i]-f_wb
        dj_dw += err*x[i]
        dj_db += err

    dj_dw /= 2*m
    dj_db /= 2*m

    return dj_dw, dj_db


def gradient_descent(x_train, y_train, w_init, b_init, alpha, num_iters):

    m = x_train.shape[0]
    w = w_init
    b = b_init

    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x_train, y_train, w, b)
        w = w+alpha*dj_dw
        b = b+alpha*dj_db

    return w, b
