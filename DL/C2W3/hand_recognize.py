import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.framework import ops
import tf_utils
import time

x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes= tf_utils.load_dataset()

# index = 11
# plt.imshow(x_train_orig[index])
# plt.show()
# print('y = ' + str(np.squeeze(y_train_orig[:, index])))

x_train_flatten = x_train_orig.reshape(x_train_orig.shape[0], -1).T
x_test_flatten = x_test_orig.reshape(x_test_orig.shape[0], -1).T

x_train = x_train_flatten / 255
x_test = x_test_flatten / 255

y_train = tf_utils.convert_to_one_hot(y_train_orig, 6)
y_test = tf_utils.convert_to_one_hot(y_test_orig, 6)


def create_placeholders(n_x, n_y):
    """
    创建占位符, 使用None灵活处理样本数量
    :param n_x: 图片向量大小: 64 * 64 * 3 = 12288
    :param n_y: 分类数: 0 / 1 / 2 / 3 / 4 / 5
    :return:
    """
    x = tf.placeholder(tf.float32, [n_x, None], name='x')
    y = tf.placeholder(tf.float32, [n_y, None], name='y')
    return x, y


def initialize_parameters():
    """
    初始化神经网络的参数
    :return:
    """
    tf.set_random_seed(1)
    # tf.Variable()每次创建新对象, tf.get_variable()返回已创建的对象,若不存在则重新建立
    w1 = tf.get_variable('w1', [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable('b1', [25, 1], initializer=tf.zeros_initializer())
    w2 = tf.get_variable('w2', [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable('b2', [12, 1], initializer=tf.zeros_initializer())
    w3 = tf.get_variable('w3', [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable('b3', [6, 1], initializer=tf.zeros_initializer())

    parameters = {'w1': w1, 'w2': w2, 'w3': w3, 'b1': b1, 'b2': b2, 'b3': b3}
    return parameters


def forward_propagation(x, parameters):
    """
    z3停止, tf最后线性输出层的输出作为计算损失函数的输入, 故不需要a3
    :param x:
    :param parameters:
    :return:
    """
    w1, w2, w3 = parameters['w1'], parameters['w2'], parameters['w3']
    b1, b2, b3 = parameters['b1'], parameters['b2'], parameters['b3']
    z1 = tf.matmul(w1, x) + b1
    a1 = tf.nn.relu(z1)
    z2 = tf.matmul(w2, a1) + b2
    a2 = tf.nn.relu(z2)
    z3 = tf.matmul(w3, a2) + b3

    return z3


def compute_cost(z3, y):
    logits = tf.transpose(z3)  # 转置
    labels = tf.transpose(y)  # 转置
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def model(x_train, y_train, x_test, y_test, learning_rate=0.0001, num_epochs=1500,
          mini_batch_size=32, print_cost=True, is_plot=True):

    # 设定初始值
    ops.reset_default_graph()  # 重新运行模型而不覆盖tf变量
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = x_train.shape
    n_y = y_train.shape[0]
    costs = []

    # 设定过程
    x, y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    z3 = forward_propagation(x, parameters)
    cost = compute_cost(z3, y)
    # 设定优化函数, 与cost同时调用
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        # 循环
        for epoch in range(num_epochs):

            # 设定mini batch
            epoch_cost = 0
            num_mini_batches = int(m / mini_batch_size)
            seed += 1
            mini_batches = tf_utils.random_mini_batches(x_train, y_train, mini_batch_size, seed)

            # mini batch循环
            for mini_batch in mini_batches:
                (mini_batch_x, mini_batch_y) = mini_batch
                _, mini_batch_cost = sess.run([optimizer, cost], feed_dict={x: mini_batch_x, y:mini_batch_y})
                epoch_cost += mini_batch_cost / num_mini_batches

            # 每个batch计算cost, 而非mini batch
            if epoch % 5 == 0:
                costs.append(epoch_cost)
                if print_cost and epoch % 100 == 0:
                    print('Epoch ' + str(epoch + 1) + '/' + str(num_epochs) + '  cost = ' + str(epoch_cost))

        # 结束循环后
        if is_plot:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        parameters = sess.run(parameters)
        print("参数已经保存到session")

        correct_prediction = tf.equal(tf.argmax(z3), tf.argmax(y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print("训练集的准确率：", accuracy.eval({x: x_train, y: y_train}))
        print("测试集的准确率:", accuracy.eval({x: x_test, y: y_test}))
    return parameters


# 开始时间
start_time = time.clock()
# 开始训练
parameters = model(x_train, y_train, x_test, y_test)
# 结束时间
end_time = time.clock()
# 计算时差
print("CPU的执行时间 = " + str(end_time - start_time) + " 秒")
