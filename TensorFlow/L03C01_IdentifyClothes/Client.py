import tensorflow_datasets as tfds
import tensorflow as tf
import math
import numpy as np
from matplotlib import pyplot as plt
import logging

tfds.disable_progress_bar()
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# 得到训练集合
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

class_names = metadata.features['label'].names  # 标签集合, 共10
# print("Class names: {}".format(class_names))

num_train_examples = metadata.splits['train'].num_examples  # 训练集60000
num_test_examples = metadata.splits['test'].num_examples  # 测试集10000


def normalize(images, labels):
    """
    归一化函数 Normalization
    :param images:
    :param labels:
    :return:
    """
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


# 将归一化函数应用于集合中的每个元素中
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# 从硬盘加载,并保留在内存中
train_dataset = train_dataset.cache()
test_dataset = test_dataset.cache()
"""
# 显示第一个图像信息
for image, label in test_dataset.take(1):
    break
image = image.numpy().reshape((28, 28))

# Plot the image - voila a piece of fashion clothing
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# 显示25个图像信息
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(test_dataset.take(25)):
    image = image.numpy().reshape((28,28))
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(class_names[label])
plt.show()
"""

# 建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),  # 28*28 -> 784*1
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # 隐含层激活
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # 128神经元, 10个全连接层
])

# 模型参数设定(compile model)
model.compile(optimizer='adam',  # 最优化方法 梯度下降法中的一种 adam优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),  # 损失函数
              metrics=['accuracy'])  # 监视训练过程

BATCH_SIZE = 32  # 每32个样本更新一次参数
train_dataset = train_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

# 训练过程 所有样本完成一次反向传播成为一个epoch
model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

# 测试 test_accuracy为正确率
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)


for test_images, test_labels in test_dataset.take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()  # 实际index
    predictions = model.predict(test_images)  # 32*10 一个batch内共32个样本，每个样本的10维标签表示是10种标签其中一个的概率


def plot_image(i, predictions_array, true_labels, images):
    """
    输出图片
    :param i:
    :param predictions_array:
    :param true_labels:
    :param images:
    :return:
    """
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    """
    输出可能的概率信息
    :param i:
    :param predictions_array:
    :param true_label:
    :return:
    """
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)
plt.show()

img = test_images[0]
img = np.array([img])  # 为了预测,需要给img增加维度到1*28*28*1 最后的1是gray
predictions_single = model.predict(img)  # 预测矩阵

# 预测值柱状图
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

predictions_single_index = np.argmax(predictions_single[0])  # 最可能输出值

"""
1.Set training epochs set to 1
每个样本仅用一次, 结果差了
2.Number of neurons in the Dense layer following the Flatten one. For example, go really low (e.g. 10) in ranges
up to 512 and see how accuracy changes
调节隐含层神经元数量, 导致执行时间不一样, 多则慢效果好, 少则快效果差
3.Add additional Dense layers between the Flatten and the final Dense(10), experiment with different units
in these layers
效果不一样
4.Don't normalize the pixel values, and see the effect that has
结果有异常
"""