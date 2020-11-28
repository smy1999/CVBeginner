import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
import lr_utils


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))  # b可以自动boardcast
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    param = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}
    return param


def initialize_parameters_deep(layers_dims):
    """

    :param layers_dims: 包含每个图层结点数量的列表
    :return:
    """
    np.random.seed(3)
    parameters = {}
    length = len(layers_dims)

    for i in range(1, length):
        parameters['w' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) / np.sqrt(layers_dims[i - 1])
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
        assert(parameters['w' + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
        assert(parameters['b' + str(i)].shape == (layers_dims[i], 1))

    return parameters


def linear_forward(a, w, b):
    z = np.dot(w, a) + b
    assert(z.shape == (w.shape[0], a.shape[1]))
    cache = (a, w, b)
    return z, cache


def linear_activation_forward(a_prev, w, b, activation):
    if activation == 'sigmoid':
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = sigmoid(z)
    elif activation == 'relu':
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = relu(z)
    assert(a.shape == (w.shape[0], a_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return a, cache


def l_model_forward(x, parameters):
    """
    实现L-1层linear + relu和一层linear+sigmoid
    (linear_forward + relu) ^ (L - 1) + linear_forward + relu
    :param x: 网络的输入x
    :param parameters: 初始化initialize_parameters_deep的输出
    :return:
    """
    caches = []
    a = x
    length = len(parameters) // 2  # 每层w和b共2个
    for i in range(1, length):
        a_prev = a
        a, cache = linear_activation_forward(a_prev, parameters['w' + str(i)], parameters['b' + str(i)], 'relu')
        caches.append(cache)
    al, cache = linear_activation_forward(a, parameters['w' + str(length)], parameters['b' + str(length)],
                                          'sigmoid')
    caches.append(cache)
    assert(al.shape == (1, x.shape[1]))
    return al, caches


def compute_cost(al, y):
    m = y.shape[1]
    cost = -np.sum(np.multiply(np.log(al), y) + np.multiply(np.log(1 - al), 1 - y)) / m
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    return cost


def linear_backward(dz, cache):
    a_prev, w, b = cache
    m = a_prev.shape[1]
    dw = np.dot(dz, a_prev.T) / m
    db = np.sum(dz, axis=1, keepdims=True) / m
    da_prev = np.dot(w.T, dz)
    assert(da_prev.shape == a_prev.shape)
    assert(dw.shape == w.shape)
    assert(db.shape == b.shape)
    return da_prev, dw, db


def linear_activation_backward(da, cache, activation='relu'):
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dz = relu_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)
    elif activation == 'sigmoid':
        dz = sigmoid_backward(da, activation_cache)
        da_prev, dw, db = linear_backward(dz, linear_cache)
    return da_prev, dw, db


def l_model_backward(al, y, caches):
    grads = {}
    length = len(caches)
    m = al.shape[1]
    y = y. reshape(al.shape)
    dal = -(np.divide(y, al) - np.divide(1 - y, 1 - al))
    current_cache = caches[length - 1]
    grads['da' + str(length)], grads['dw' + str(length)], grads['db' + str(length)] = \
        linear_activation_backward(dal, current_cache, 'sigmoid')
    for i in reversed(range(length - 1)):
        current_cache = caches[i]
        grads['da' + str(i + 1)], grads['dw' + str(i + 1)], grads['db' + str(i + 1)] = \
            linear_activation_backward(grads['da' + str(i + 2)], current_cache, 'relu')
    return grads


def update_parameters(parameters, grads, learning_rate):
    length = len(parameters) // 2
    for i in range(1, length + 1):
        parameters['w' + str(i)] = parameters['w' + str(i)] - learning_rate * grads['dw' + str(i)]
        parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * grads['db' + str(i)]
    return parameters


def model(x, y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, is_plot=True):
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)

    for i in range(num_iterations):
        al, caches = l_model_forward(x, parameters)
        cost = compute_cost(al, y)
        grads = l_model_backward(al, y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if (i + 1) % 200 == 0:
            costs.append(cost)
            if print_cost:
                print('Epoch ' + str(i + 1) + '/' + str(num_iterations) + '  cost = ' + str(cost))

    if is_plot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('costs')
        plt.xlabel('iterations (per tens')
        plt.title('learning rate =' + str(learning_rate))
        plt.show()

    return parameters


def predict(x, y, parameters):
    """

    :param x:
    :param y:
    :param parameters: 模型训练得到的参数
    :return:
    """
    m = x.shape[1]
    n = len(parameters) // 2
    p = np.zeros((1, m))

    probas, caches = l_model_forward(x, parameters)

    for i in range(probas.shape[1]):
        p[0, i] = 1 if probas[0, i] > 0.5 else 0

    print("准确度为: " + str(float(np.sum((p == y)) / m)))

    return p


train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

layers_dims = [12288, 20, 7, 5, 1]
parameters = model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True, is_plot=True)

predictions_train = predict(train_x, train_y, parameters)  # 训练集
predictions_test = predict(test_x, test_y, parameters)  # 测试集
