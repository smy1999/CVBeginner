import numpy as np
import matplotlib.pyplot as plt
import init_utils   # 第一部分，初始化

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_x, train_y, test_x, test_y = init_utils.load_dataset(is_plot=False)


def initialize_parameters_zeros(layers_dims):
    parameters = {}
    length = len(layers_dims)
    for i in range(1, length):
        parameters['w' + str(i)] = np.zeros((layers_dims[i], layers_dims[i - 1]))
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
        assert(parameters['w' + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
        assert(parameters['b' + str(i)].shape == (layers_dims[i], 1))
    return parameters


def initialize_parameters_random(layers_dims):
    np.random.seed(3)
    parameters = {}
    length = len(layers_dims)
    for i in range(1, length):
        parameters['w' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * 10
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
        assert (parameters['w' + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
        assert (parameters['b' + str(i)].shape == (layers_dims[i], 1))
    return parameters


def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    length = len(layers_dims)
    for i in range(1, length):
        parameters['w' + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * np.sqrt(2 / layers_dims[i - 1])
        parameters['b' + str(i)] = np.zeros((layers_dims[i], 1))
        assert (parameters['w' + str(i)].shape == (layers_dims[i], layers_dims[i - 1]))
        assert (parameters['b' + str(i)].shape == (layers_dims[i], 1))
    return parameters


def model(x, y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization='he', is_plot=True,\
          subplot=False):
    """
    Implement a neural network with 3 layers.
    Linear->Relu->Linear->Relu->Linear->Sigmoid.
    :param subplot:
    :param x:
    :param y:
    :param learning_rate:
    :param num_iterations:
    :param print_cost:
    :param initialization:
    :param is_plot:
    :return:
    """
    costs = []
    grads = {}
    m = x.shape[1]
    layers_dims = [x.shape[0], 10, 5, 1]
    if initialization == 'zeros':
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == 'random':
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == 'he':
        parameters = initialize_parameters_he(layers_dims)
    else:
        print('Initialization error!')
        exit

    for i in range(num_iterations):
        al, cache = init_utils.forward_propagation(x, parameters)
        cost = init_utils.compute_loss(al, y)
        grads = init_utils.backward_propagation(x, y, cache)
        parameters = init_utils.update_parameters(parameters, grads, learning_rate=learning_rate)
        if (i + 1) % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print('Epoch ' + str(i + 1) + '/' + str(num_iterations) + '  cost = ' + str(cost))

    if is_plot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('costs')
        plt.xlabel('iterations (per tens')
        plt.title('learning rate =' + str(learning_rate))
        if not subplot:
            plt.show()

    return parameters


initialization_type = ['zeros', 'random', 'he']

for i in range(3):
    print("-------Initialization Type : " + initialization_type[i] + "-----------")
    plt.subplot(3, 2, i * 2 + 1)

    parameters = model(train_x, train_y, initialization=initialization_type[i], print_cost=False, is_plot=True,
                       subplot=True)
    print("Train Set ", end='')
    predictions_train = init_utils.predict(train_x, train_y, parameters)
    print("Test Set ", end='')
    init_utils.predictions_test = init_utils.predict(test_x, test_y, parameters)

    plt.subplot(3, 2, i * 2 + 2)
    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_x, train_y)

plt.show()
