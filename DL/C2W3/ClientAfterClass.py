import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import tf_utils
import time


def class_test():
    np.random.seed(1)

    y_hat = tf.constant(36, name='y_hat')  # 定义y_hat固定值36
    y = tf.constant(39, name='y')  # 定义y固定值39

    # 将该计算过程放入一个计算图中,但并没有计算
    loss = tf.Variable((y - y_hat) ** 2, name='loss')  # 为损失函数创建变量

    # 初始化
    init = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init)
        print(session.run(loss))

    # 创建了变量和计算函数, 但此时仅有计算图, 而没有进行计算, 没有结果
    a = tf.constant(2)
    b = tf.constant(10)
    c = tf.multiply(a, b)
    print(c)

    # 创建一个会话并运行才会计算
    session = tf.Session()
    print(session.run(c))
    print(session.run(c))

    # 占位符的应用
    x = tf.placeholder(tf.int64, name='x')
    print(session.run(2 * x, feed_dict={x : 3}))
    session.close()


def linear_function():
    """
    Linear function: y = w * x + b
    :return:
    """
    np.random.seed(1)
    x = np.random.randn(3, 1)
    w = np.random.randn(4, 3)
    b = np.random.randn(4, 1)

    y = tf.add(tf.matmul(w, x), b)
    session = tf.Session()
    result = session.run(y)
    session.close()
    return result


def sigmoid(z):
    x = tf.placeholder(tf.float32, name='x')
    sigmoid = tf.sigmoid(x)
    session = tf.Session()
    result = session.run(sigmoid, feed_dict={x: z})
    return result

# 计算成本
# tf.nn.sigmoid_cross_entropy_with_logits(logits= , labels= )


def one_hot_matrix(labels, c):
    """
    Create a matrix, in which ith-row is ith-cluster and jth-column is jth-sample.
    i行i类, j行j样本. 若样本j有标签i, 则(i, j)为1
    :param lables: 标签向量
    :param c: 分类树
    :return:
    """
    c = tf.constant(c, name='c')
    one_hot_matrix = tf.one_hot(indices=labels, depth=c, axis=0)
    session = tf.Session()
    one_hot = session.run(one_hot_matrix)
    session.close()
    return one_hot


def ones(shape):
    ones = tf.ones(shape)
    session = tf.Session()
    ones = session.run(ones)
    return ones


def zeros(shape):
    zeros = tf.zeros(shape)
    session = tf.Session()
    zeros = session.run(zeros)
    return zeros


print ("ones = " + str(ones([3])))