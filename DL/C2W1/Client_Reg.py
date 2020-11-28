import numpy as np
import matplotlib.pyplot as plt
import reg_utils    # 第二部分，正则化

train_x, train_y, test_x, test_y = reg_utils.load_2D_dataset(is_plot=True)


def compute_cost_with_regularization(al, y, parameters, lambd):
    m = y.shape[1]
    w1 = parameters['w1']
    w2 = parameters['w2']
    w3 = parameters['w3']
    cross_entropy_cost = reg_utils.compute_cost(al, y)
    l2_regression_cost = lambd * (np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3))) / 2 / m
    cost = cross_entropy_cost + l2_regression_cost
    return cost


def forward_propagation_with_dropout(x, parameters, keep_prob=0.5):
    np.random.seed(1)

    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    w3 = parameters["w3"]
    b3 = parameters["b3"]

    z1 = np.dot(w1, x) + b1
    a1 = reg_utils.relu(z1)
    d1 = np.random.rand(a1.shape[0], a1.shape[1])
    d1 = d1 < keep_prob
    a1 = a1 * d1
    a1 = a1 / keep_prob

    z2 = np.dot(w2, a1) + b2
    a2 = reg_utils.relu(z2)
    d2 = np.random.rand(a2.shape[0], a2.shape[1])
    d2 = d2 < keep_prob
    a2 = a2 * d2
    a2 = a2 / keep_prob

    z3 = np.dot(w3, a2) + b3
    a3 = reg_utils.sigmoid(z3)

    cache = (z1, d1, a1, w1, b1, z2, d2, a2, w2, b2, z3, a3, w3, b3)
    return a3, cache


def backward_propagation_with_regression(x, y, cache, lambd):
    m = x.shape[1]
    z1, a1, w1, b1, z2, a2, w2, b2, z3, a3, w3, b3 = cache
    dz3 = a3 - y
    dw3 = np.dot(dz3, a2.T) / m + lambd * w3 / m
    db3 = np.sum(dz3, axis=1, keepdims=True) / m
    da2 = np.dot(w3.T, dz3)
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dw2 = np.dot(dz2, a1.T) / m + lambd * w2 / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    da1 = np.dot(w2.T, dz2)
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dw1 = np.dot(dz1, x.T) / m + lambd * w1 / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
    gradients = {'dz1': dz1, 'dw1': dw1, 'db1': db1, 'da1': da1,
                 'dz2': dz2, 'dw2': dw2, 'db2': db2, 'da2': da2,
                 'dz3': dz3, 'dw3': dw3, 'db3': db3}
    return gradients


def backward_propagation_with_dropout(x, y, cache, keep_prob):
    m = x.shape[1]
    (z1, d1, a1, w1, b1, z2, d2, a2, w2, b2, z3, a3, w3, b3) = cache

    dz3 = a3 - y
    dw3 = np.dot(dz3, a2.T) / m
    db3 = 1. / m * np.sum(dz3, axis=1, keepdims=True)
    da2 = np.dot(w3.T, dz3)
    
    da2 = da2 * d2  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    da2 = da2 / keep_prob  # 步骤2：缩放未舍弃的节点(不为0)的值
    
    dz2 = np.multiply(da2, np.int64(a2 > 0))
    dw2 = 1. / m * np.dot(dz2, a1.T)
    db2 = 1. / m * np.sum(dz2, axis=1, keepdims=True)
    
    da1 = np.dot(w2.T, dz2)
    
    da1 = da1 * d1  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    da1 = da1 / keep_prob  # 步骤2：缩放未舍弃的节点(不为0)的值
    
    dz1 = np.multiply(da1, np.int64(a1 > 0))
    dw1 = 1. / m * np.dot(dz1, x.T)
    db1 = 1. / m * np.sum(dz1, axis=1, keepdims=True)

    gradients = {'dz1': dz1, 'dw1': dw1, 'db1': db1, 'da1': da1,
                 'dz2': dz2, 'dw2': dw2, 'db2': db2, 'da2': da2,
                 'dz3': dz3, 'dw3': dw3, 'db3': db3}
    return gradients


def model(x, y, learning_rate=0.3, num_iterations=30000, print_cost=True, is_plot=True, lambd=0, keep_prob=1):
    """
    Implement a neural network with 3 layers.
    Linear->Relu->Linear->Relu->Linear->Sigmoid.
    :param x:
    :param y:
    :param learning_rate:
    :param num_iterations:
    :param print_cost:
    :param is_plot:
    :param lambd:
    :param keep_prob:
    :return:
    """
    grads = {}
    costs = []
    m = x.shape[1]
    layers_dims = [x.shape[0], 20, 3, 1]
    parameters = reg_utils.initialize_parameters(layers_dims)

    for i in range(num_iterations):

        if keep_prob == 1:
            al, cache = reg_utils.forward_propagation(x, parameters)  # 不使用随机删除
        elif keep_prob < 1:
            al, cache = forward_propagation_with_dropout(x, parameters, keep_prob)  # 使用随机删除
        else:
            print("keep_prob参数错误！程序退出。")
            exit()

        if lambd == 0:
            cost = reg_utils.compute_cost(al, y)
        else:
            cost = compute_cost_with_regularization(al, y, parameters, lambd)

        assert(lambd == 0 or keep_prob == 1)

        if lambd == 0 and keep_prob == 1:
            grads = reg_utils.backward_propagation(x, y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regression(x, y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(x, y, cache, keep_prob)
        parameters = reg_utils.update_parameters(parameters, grads, learning_rate=learning_rate)

        if (i + 1) % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print('Epoch ' + str(i + 1) + '/' + str(num_iterations) + '  cost = ' + str(cost))

    if is_plot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('costs')
        plt.xlabel('iterations (per tens')
        plt.title('learning rate =' + str(learning_rate))
        # plt.show()

    return parameters


# 无正则化
plt.subplot(3, 2, 1)
parameters = model(train_x, train_y,is_plot=True, print_cost=False)
print("Train Set ", end='')
predictions_train = reg_utils.predict(train_x, train_y, parameters)
print("Test Set ", end='')
predictions_test = reg_utils.predict(test_x, test_y, parameters)

plt.subplot(3, 2, 2)
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_x, train_y)

# L2 regression
plt.subplot(3, 2, 3)
parameters = model(train_x, train_y,lambd = 0.7, is_plot=True, print_cost=False)
print("Train Set ", end='')
predictions_train = reg_utils.predict(train_x, train_y, parameters)
print("Test Set ", end='')
predictions_test = reg_utils.predict(test_x, test_y, parameters)

plt.subplot(3, 2, 4)
plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_x, train_y)

# dropout
plt.subplot(3, 2, 5)
parameters = model(train_x, train_y,keep_prob=0.86, learning_rate=0.3, is_plot=True, print_cost=False)
print("Train Set ", end='')
predictions_train = reg_utils.predict(train_x, train_y, parameters)
print("Test Set ", end='')
predictions_test = reg_utils.predict(test_x, test_y, parameters)

plt.subplot(3, 2, 6)
plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_x, train_y)

plt.show()
