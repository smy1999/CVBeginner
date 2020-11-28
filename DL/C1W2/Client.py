import h5py
import numpy as np
import matplotlib.pyplot as plt
import lr_utils

train_set_x, train_set_y, test_set_x, test_set_y, classes = lr_utils.load_dataset()

m_train = train_set_x.shape[0]
m_test = test_set_x.shape[0]

print('train set shape : ' + str(train_set_x.shape))
print('test set shape : ' + str(test_set_x.shape))

train_set_x_flatten = train_set_x.reshape((m_train, -1)).T  # m_train行,列自动计算, .T转置
test_set_x_flatten = test_set_x.reshape((m_test, -1)).T

print('train set flatten shape' + str(train_set_x_flatten.shape))
print('test set flatten shape' + str(test_set_x_flatten.shape))

train_set_x_normalization = train_set_x_flatten / 255
test_set_x_normalization = test_set_x_flatten / 255

print()
print()


def sigmoid(z):
    """
    Sigmoid Activation.
    :param z: z = w.T * x + b
    :return:
    """
    s = 1 / (1 + np.exp(-z))
    return s


def initialize_with_zeros(dim):
    """
    Create a (dim, 1) vector for w which elements are all zeros and a zero for b.
    :param dim:
    :return:
    """
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


def propagate(x, y, w, b):
    """
    Calculate cost function.
    :param x: Input matrix.
    :param y: True label.
    :param w: Weight.
    :param b:
    :return: dw = dCost / dw, db = dz
    """
    m = x.shape[1]

    # forward propagation
    s = sigmoid(np.dot(w.T, x) + b)
    cost = -np.sum(y * np.log(s) + (1 - y) * np.log(1 - s)) / m

    # backward propagation
    dw = np.dot(x, (s - y).T) / m
    db = np.sum(s - y) / m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)  # 删除单维度
    assert(cost.shape == ())

    grads = {'dw': dw, 'db': db}
    return grads, cost


def optimize(x, y, w, b, num_iterations, learning_rate, print_cost=False):
    """
    Optimize w and b with gradient descent.
    :param x:
    :param y:
    :param w:
    :param b:
    :param num_iterations:
    :param learning_rate:
    :param print_cost:
    :return:
    """
    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(x, y, w, b)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and (i + 1) % 100 == 0:
            print('Epoch ' + str(i + 1) + '/' + str(num_iterations) + '  cost = ' + str(cost))

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}
    return params, grads, costs


def predict(w, b, x):
    m = x.shape[1]
    y_prediction = np.zeros((1, m))
    w = w.reshape(x.shape[0], 1)

    s = sigmoid(np.dot(w.T, x) + b)
    for i in range(s.shape[1]):
        y_prediction[0, i] = 1 if s[0, i] > 0.5 else 0
    assert(y_prediction.shape == (1, m))

    return y_prediction


def model(x_train, y_train, x_test, y_test, num_iterations=2000, learning_rate=90.5, print_cost=False):

    w, b = initialize_with_zeros(x_train.shape[0])
    params, grads, costs = optimize(x_train, y_train, w, b, num_iterations=num_iterations,
                                    learning_rate=learning_rate, print_cost=print_cost)
    w, b = params['w'], params['b']
    y_prediction_train = predict(w, b, x_train)
    y_prediction_test = predict(w, b, x_test)

    print('train set rate : ' + str(100 - 100 * np.mean(np.abs(y_prediction_train - y_train))) + "%")
    print(' test set rate : ' + str(100 - 100 * np.mean(np.abs(y_prediction_test - y_test))) + "%")

    d = {'costs': costs, 'w': w, 'b': b,
         'y_prediction_test': y_prediction_test, 'y_prediction_train': y_prediction_train,
         'learning_rate': learning_rate, 'num_iterations': num_iterations}
    return d


models = {}
learning_rates = [0.01, 0.001, 0.0001]
for i in learning_rates:
    print('learning rate : ' + str(i))
    models[str(i)] = model(train_set_x_normalization, train_set_y, test_set_x_normalization, test_set_y,
                           num_iterations=1000, learning_rate=i, print_cost=False)
    print()

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]['costs']), label=str(models[str(i)]['learning_rate']))
plt.ylabel('cost')
plt.xlabel('iterations(hundreds)')
plt.legend(loc='upper center')
plt.show()
