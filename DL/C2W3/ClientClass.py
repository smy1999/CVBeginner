import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# w = tf.Variable(0, dtype=tf.float32)
# # cost = tf.add(tf.add(w ** 2, tf.multiply(-10., w)), 25)
# cost = w ** 2 - 10 * w + 25
# train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
#
# # 初始化全局变量为0
# init = tf.global_variables_initializer()
# session = tf.Session()
# session.run(init)
#
# session.run(train)  # 执行一步梯度下降
# print(session.run(w))
#
# for i in range(1000):
#     session.run(train)
# print(session.run(w))

coefficient = np.array([[1.], [-10.], [25.]])
w = tf.Variable(0, dtype=tf.float32)
x = tf.placeholder(tf.float32, [3, 1])  # 建立[3,1]数组, 占位
cost = x[0][0] * w ** 2 + x[1][0] + x[2][0]
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w))

feed_dict = {x: coefficient}  # 将coefficient赋给x
session.run(train, feed_dict=feed_dict)  # 执行一步梯度下降
print(session.run(w))

for i in range(1000):
    session.run(train, feed_dict=feed_dict)
    print(session.run(w))
