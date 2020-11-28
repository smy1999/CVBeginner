import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

import opt_utils  # 参见数据包或者在本文底部copy
import testCase  # 参见数据包或者在本文底部copy

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def update_parameters_with_gd(parameters, gradients, learning_rate):
    length = len(parameters) // 2
    for i in range(length):
        parameters['w' + str(i + 1)] -= learning_rate * gradients['dw' + str(i + 1)]
        parameters['b' + str(i + 1)] -= learning_rate * gradients['db' + str(i + 1)]
    return parameters


def random_mini_batches(x, y, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = x.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))  # 返回长度为m, 包含0到m-1打乱顺序的数组
    shuffled_x = x[:, permutation]  # 将每一列数据按permutation重新排列
    shuffled_y = y[:, permutation].reshape((1, m))

    num_complete_mini_batches = math.floor(m / mini_batch_size)  # 地板除, 保留底
    for k in range(0, num_complete_mini_batches):
        mini_batch_x = shuffled_x[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch_y = shuffled_y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_x = shuffled_x[:, mini_batch_size * num_complete_mini_batches:]
        mini_batch_y = shuffled_y[:, mini_batch_size * num_complete_mini_batches:]
        mini_batch = (mini_batch_x, mini_batch_y)
        mini_batches.append(mini_batch)
    return mini_batches


def initialize_velocity(parameters):
    length = len(parameters) // 2
    velocity = {}
    for i in range(length):
        # v["dw" + str(i + 1)] = np.zeros_like(parameters["w" + str(i + 1)])  # 两种方法均可
        # v["db" + str(i + 1)] = np.zeros_like(parameters["b" + str(i + 1)])
        velocity['dw' + str(i + 1)] = np.zeros(parameters['w' + str(i + 1)].shape)
        velocity['db' + str(i + 1)] = np.zeros(parameters['b' + str(i + 1)].shape)
    return velocity


def update_parameters_with_momentum(parameters, gradients, velocity, beta, learning_rate):
    length = len(parameters) // 2
    for i in range(length):
        velocity['dw' + str(i + 1)] = beta * velocity['dw' + str(i + 1)] + (1 - beta) * gradients['dw' + str(i + 1)]
        velocity['db' + str(i + 1)] = beta * velocity['db' + str(i + 1)] + (1 - beta) * gradients['db' + str(i + 1)]

        parameters['w' + str(i + 1)] -= learning_rate * velocity['dw' + str(i + 1)]
        parameters['b' + str(i + 1)] -= learning_rate * velocity['db' + str(i + 1)]

    return parameters, velocity


def initialize_adam(parameters):
    length = len(parameters) // 2
    velocity = {}
    square = {}
    for i in range(length):
        square['dw' + str(i + 1)] = np.zeros(parameters['w' + str(i + 1)].shape)
        square['db' + str(i + 1)] = np.zeros(parameters['b' + str(i + 1)].shape)
        velocity['dw' + str(i + 1)] = np.zeros(parameters['w' + str(i + 1)].shape)
        velocity['db' + str(i + 1)] = np.zeros(parameters['b' + str(i + 1)].shape)

    return velocity, square


def update_parameters_with_adam(parameters, gradients, velocity, square, time, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    length = len(parameters) // 2
    square_corrected = {}
    velocity_corrected = {}
    for i in range(length):
        db = 'db' + str(i + 1)
        dw = 'dw' + str(i + 1)
        velocity[dw] = beta1 * velocity[dw] + (1 - beta1) * gradients[dw]
        velocity[db] = beta1 * velocity[db] + (1 - beta1) * gradients[db]

        square[dw] = beta2 * square[dw] + (1 - beta2) * np.square(gradients[dw])
        square[db] = beta2 * square[db] + (1 - beta2) * np.square(gradients[db])

        velocity_corrected[dw] = velocity[dw] / (1 - np.power(beta1, time))
        velocity_corrected[db] = velocity[db] / (1 - np.power(beta1, time))

        square_corrected[dw] = square[dw] / (1 - np.power(beta2, time))
        square_corrected[db] = square[db] / (1 - np.power(beta2, time))

        parameters['w' + str(i + 1)] -= learning_rate * velocity_corrected[dw] /\
                                        (np.sqrt(square_corrected[dw]) + epsilon)
        parameters['b' + str(i + 1)] -= learning_rate * velocity_corrected[db] /\
                                        (np.sqrt(square_corrected[db]) + epsilon)

    return parameters, velocity, square


def model(x, y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64,
          beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
          num_epochs=10000, print_cost=True, is_plot=True):
    """

    :param x:
    :param y:
    :param layers_dims:
    :param optimizer:
    :param learning_rate:
    :param mini_batch_size:
    :param beta:
    :param beta1:
    :param beta2:
    :param epsilon:
    :param num_epochs:
    :param print_cost:
    :param is_plot:
    :return:
    """
    length = len(layers_dims)
    costs = []
    time = 0
    seed = 10
    parameters = opt_utils.initialize_parameters(layers_dims)

    if optimizer == 'gd':  # 不使用优化器
        pass
    elif optimizer == 'momentum':  # 使用Momentum优化器
        velocity = initialize_velocity(parameters)
    elif optimizer == 'adam':  # 使用adam优化器
        velocity, square = initialize_adam(parameters)
    else:
        print('Optimizer Error!')
        exit(1)

    for i in range(num_epochs):
        seed = seed + 1  # 随机种子不同, 使得每次划分mini batch不同
        mini_batches = random_mini_batches(x, y, mini_batch_size, seed)

        for mini_batch in mini_batches:
            (mini_batch_x, mini_batch_y) = mini_batch
            a3, cache = opt_utils.forward_propagation(mini_batch_x, parameters)
            cost = opt_utils.compute_cost(a3, mini_batch_y)
            gradients = opt_utils.backward_propagation(mini_batch_x, mini_batch_y, cache)

            if optimizer == 'gd':  # 不使用优化器
                parameters = update_parameters_with_gd(parameters, gradients, learning_rate)
            elif optimizer == 'momentum':  # 使用Momentum优化器
                parameters, velocity = update_parameters_with_momentum(parameters, gradients, velocity, beta, learning_rate)
            elif optimizer == 'adam':  # 使用adam优化器
                time += 1
                parameters, velocity, square = update_parameters_with_adam(parameters, gradients, velocity, square,
                                                                           time, learning_rate, beta1, beta2, epsilon)
        if (i + 1) % 100 == 0:
            costs.append(cost)
            if print_cost and (i + 1) % 1000 == 0:
                print('Epoch ' + str(i + 1) + '/' + str(num_epochs) + '  cost = ' + str(cost))

    if is_plot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('costs')
        plt.xlabel('iterations (per tens')
        plt.title('learning rate =' + str(learning_rate))
        # plt.show()

    return parameters


train_x, train_y = opt_utils.load_dataset(is_plot=False)
layers_dims = [train_x.shape[0], 5, 2, 1]

plt.subplot(3, 2, 1)
param_gd = model(train_x, train_y, layers_dims, optimizer='gd', is_plot=True, print_cost=False)

plt.subplot(3, 2, 3)
param_momentum = model(train_x, train_y, layers_dims, beta=0.9, optimizer="momentum", is_plot=True, print_cost=False)

plt.subplot(3, 2, 5)
param_adam = model(train_x, train_y, layers_dims, optimizer="adam",is_plot=True, print_cost=False)

param_dict = {'gd': param_gd, 'momentum': param_momentum, 'adam': param_adam}
title_dict = {'gd': 'Gradient Descent', 'momentum': 'Momentum', 'adam': 'Adam'}
title_key = ['gd', 'momentum', 'adam']
# 显示准确率

for i in range(3):
    print('Optimizer type : ' + title_dict[title_key[i]])
    preditions = opt_utils.predict(train_x, train_y, param_dict[title_key[i]])

    # 绘制分类图
    plt.subplot(3, 2, 2 * i + 2)
    plt.title("Model with " + title_dict[title_key[i]] + " Optimization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 2.5])
    axes.set_ylim([-1, 1.5])
    opt_utils.plot_decision_boundary(lambda x: opt_utils.predict_dec(param_dict[title_key[i]], x.T), train_x, train_y)

plt.show()
