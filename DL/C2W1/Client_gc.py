import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import gc_utils     #第三部分，梯度校验


def forward_propogation(x, theta):
    j = np.dot(theta, x)
    return j


def backward_propagation(x, theta):
    dtheta = x
    return dtheta


def gradient_check(x, theta, epsilon=1e-7):
    theta_plus = theta + epsilon
    theta_minus = theta - epsilon
    j_plus = forward_propogation(x, theta_plus)
    j_minus = forward_propogation(x, theta_minus)
    grad_approx = (j_plus - j_minus) / 2 / epsilon

    grad = backward_propagation(x, theta)
    numerator = np.linalg.norm(grad - grad_approx)
    denominator = np.linalg.norm(grad_approx) + np.linalg.norm(grad)
    difference = numerator / denominator

    if difference < epsilon:
        print("Correct")
    else:
        print("Wrong")
    return difference


def forward_propagation_n(x, y, parameters):
    m = x.shape[1]
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    w3 = parameters['w3']
    b3 = parameters['b3']

    z1 = np.dot(w1, x) + b1
    a1 = gc_utils.relu(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = gc_utils.relu(z2)
    z3 = np.dot(w3, a2) + b3
    a3 = gc_utils.sigmoid(z3)

    cost = np.sum(np.multiply(-np.log(a3), y) + np.multiply(-np.log(1 - a3), (1 - y))) / m
    cache = [z1, a1, w1, b1, z2, a2, w2, b2, z3, a3, w3, b3]

    return cost, cache


def backward_propagation_n(x, y, cache):
    z1, a1, w1, b1, z2, a2, w2, b2, z3, a3, w3, b3 = cache
    m = x.shape[1]

    dz3 = a3 - y
    dw3 = np.dot(dz3, a2.T) / m
    db3 = np.sum(dz3, axis=1, keepdims=True) / m
    da2 = np.dot(w3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    da1 = np.dot(w2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dw1 = np.dot(dz1, x.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m

    gradients = {'dz1': dz1, 'dw1': dw1, 'db1': db1, 'da1': da1,
                 'dz2': dz2, 'dw2': dw2, 'db2': db2, 'da2': da2,
                 'dz3': dz3, 'dw3': dw3, 'db3': db3}
    return gradients

def gradient_check_n(parameters, gradients, x, y, epsilon=1e-7):
    parameters_values, keys = gc_utils.dictionary_to_vector(parameters)
    grad = gc_utils.gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    j_plus = np.zeros((num_parameters, 1))
    j_minus = np.zeros((num_parameters, 1))
    grad_approx = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        theta_plus = np.copy(parameters_values)
        theta_plus[i][0] += epsilon
        j_plus[i], cache = forward_propagation_n(x, y, gc_utils.vector_to_dictionary(theta_plus))

        theta_minus = np.copy(parameters_values)
        theta_minus[i][0] -= epsilon
        j_minus[i], cache = forward_propagation_n(x, y, gc_utils.vector_to_dictionary(theta_minus))

        grad_approx[i] = (j_plus[i] - j_minus[i]) / 2 / epsilon

    numerator = np.linalg.norm(grad - grad_approx)
    denominator = np.linalg.norm(grad_approx) + np.linalg.norm(grad)
    difference = numerator / denominator

    if difference < epsilon:
        print("Correct")
    else:
        print("Wrong")
    return difference

