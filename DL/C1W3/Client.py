import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)  # 设置一个固定的随机种子，以保证接下来的步骤中我们的结果是一致的。

x, y = load_planar_dataset()

# plt.scatter(x[0, :], x[1, :], c=y, s=40, cmap=plt.cm.Spectral)  # scatter绘制散点图
# plt.show()

m = y.shape[1]

print('数据集 ' + str(m))
print('shape x = ' + str(x.shape))
print('shape y = ' + str(y.shape))

# logistic分类器测试
# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(x.T, y.T)  # 训练
#
# plot_decision_boundary(lambda x: clf.predict(x), x, y)  # 显示边界
# plt.title('logistic regression')
# plt.show()
# LR_predictions = clf.predict(x.T)
# print("逻辑回归的准确性： %d " % float((np.dot(y, LR_predictions) +
#                                np.dot(1 - y, 1 - LR_predictions)) / float(y.size) * 100) +
#       "% " + "(正确标记的数据点所占的百分比)")  # y为1或0


def layer_sizes(x, y):
    n_x = x.shape[0]
    n_h = 4
    n_y = y.shape[0]
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))  # b可以自动boardcast
    w2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    param = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}
    return param


def forward_propagation(x, param):
    w1 = param['w1']
    w2 = param['w2']
    b1 = param['b1']
    b2 = param['b2']
    z1 = np.dot(w1, x) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    assert(a2.shape == (1, x.shape[1]))
    cache = {'z1': z1, 'z2': z2, 'a1': a1, 'a2': a2}
    return a2, cache


def compute_cost(a2, y, param):
    m = y.shape[1]
    w1 = param['w1']
    w2 = param['w2']

    logprobs = np.multiply(np.log(a2), y) + np.multiply(1 - y, np.log(1 - a2))
    cost = -np.sum(logprobs) / m
    cost = float(np.squeeze(cost))

    assert(isinstance(cost, float))
    return cost


def backward_propagation(param, cache, x, y):
    m = x.shape[1]
    w1 = param['w1']
    w2 = param['w2']
    a1 = cache['a1']
    a2 = cache['a2']
    dz2 = a2 - y
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = (1 / m) * np.dot(dz1, x.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    grads = {'dw1': dw1, 'dw2': dw2, 'db1': db1, 'db2': db2}
    return grads


def update_parameters(param, grads, learning_rate=1.2):
    w1 = param['w1']
    w2 = param['w2']
    b1 = param['b1']
    b2 = param['b2']
    dw1 = grads['dw1']
    dw2 = grads['dw2']
    db1 = grads['db1']
    db2 = grads['db2']
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    param = {'w1': w1, 'w2': w2, 'b1': b1, 'b2': b2}
    return param


def model(x, y, n_h, num_iterations, learning_rate=0.5, print_cost=False):
    np.random.seed(3)
    n_x, temp, n_y = layer_sizes(x, y)
    param = initialize_parameters(n_x, n_h, n_y)
    for i in range(num_iterations):
        a2, cache = forward_propagation(x, param)
        cost = compute_cost(a2, y, param)
        grads = backward_propagation(param, cache, x, y)
        param = update_parameters(param, grads, learning_rate=learning_rate)
        if print_cost and (i + 1) % 1000 == 0:
            print('Epoch ' + str(i + 1) + '/' + str(num_iterations) + '  cost = ' + str(cost))
    return param


def predict(param, x):
    a2, cache = forward_propagation(x, param)
    predictions = np.round(a2)  # 四舍五入
    return predictions


# parameters = model(x, y, n_h=4, num_iterations=10000, learning_rate=0.5, print_cost=True)
# plot_decision_boundary(lambda x: predict(parameters, x.T), x, y)
# predictions = predict(parameters, x)
# print("rate %d" % float((np.dot(y, predictions.T) + np.dot(1 - y, 1 - predictions.T)) / float(y.size) * 100) + '%')

plt.figure(figsize=(16, 32))
layers = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(layers):
    plt.subplot(5, 2, i + 1)
    plt.title('hidden layer of size %d' % n_h)
    param = model(x, y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(param, x.T), x, y)
    predictions = predict(param, x)
    print("rate %d" % float((np.dot(y, predictions.T) + np.dot(1 - y, 1 - predictions.T)) / float(y.size) * 100) + '%')
plt.show()
