# -*- coding: utf-8 -*-

import numpy as np

def update_parameters_with_gd_test_case():
    np.random.seed(1)
    learning_rate = 0.01
    w1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    w2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    dw1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dw2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    grads = {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}
    
    return parameters, grads, learning_rate

"""
def update_parameters_with_sgd_checker(function, inputs, outputs):
    if function(inputs) == outputs:
        print("Correct")
    else:
        print("Incorrect")
"""

def random_mini_batches_test_case():
    np.random.seed(1)
    mini_batch_size = 64
    X = np.random.randn(12288, 148)
    Y = np.random.randn(1, 148) < 0.5
    return X, Y, mini_batch_size

def initialize_velocity_test_case():
    np.random.seed(1)
    w1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    w2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    return parameters

def update_parameters_with_momentum_test_case():
    np.random.seed(1)
    w1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    w2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    dw1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dw2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    grads = {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}
    v = {'dw1': np.array([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.]]), 'dw2': np.array([[ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
        [ 0.]]), 'db2': np.array([[ 0.],
        [ 0.],
        [ 0.]])}
    return parameters, grads, v
    
def initialize_adam_test_case():
    np.random.seed(1)
    w1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    w2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    return parameters

def update_parameters_with_adam_test_case():
    np.random.seed(1)
    v, s = ({'dw1': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'dw2': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
         [ 0.]]), 'db2': np.array([[ 0.],
         [ 0.],
         [ 0.]])}, {'dw1': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'dw2': np.array([[ 0.,  0.,  0.],
         [ 0.,  0.,  0.],
         [ 0.,  0.,  0.]]), 'db1': np.array([[ 0.],
         [ 0.]]), 'db2': np.array([[ 0.],
         [ 0.],
         [ 0.]])})
    w1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    w2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    dw1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dw2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    
    parameters = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}
    grads = {"dw1": dw1, "db1": db1, "dw2": dw2, "db2": db2}
    
    return parameters, grads, v, s
