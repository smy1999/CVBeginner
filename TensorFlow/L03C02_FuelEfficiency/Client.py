from __future__ import absolute_import, division, print_function
import pathlib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset_path = keras.utils.get_file("auto.mpg.data",
                                    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
# print("Data Set Path : " + dataset_path)
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

# read data
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values='?', comment='\t',
                          sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()

# print(dataset.isna().sum())  # 用于检测空值, 统计NaN的数量
dataset = dataset.dropna()  # 删除空行

# 国家信息添加在内
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0

# 拆分测试集和训练集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 核密度估计Kernel Density Estimation KDE
# pairplot用于可视化数据特征之间的关系
sns.pairplot(train_dataset[['MPG', 'Cylinders', 'Displacement', 'Weight']], diag_kind='kde')
# plt.show()

pd.set_option('display.max_rows', None)  # 显示完整的行
pd.set_option('display.max_columns', None)  # 显示完整的列

# 描述性统计分析
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
# print(train_stats)

# MPG是模型预测的值, 将该值从标签中删除
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


def norm(x):
    """
    Normalization
    :param x:
    :return:
    """
    return (x - train_stats['mean']) / train_stats['std']


def build_model():
    """
    顺序模型，包括两个隐藏层，返回单个连续值的输出层
    :return:
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


# 归一化
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# model = build_model()  # 构造模型


# print(model.summary())  # 显示模型的特征


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 1000  # 周期为1000的训练

# history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
#                     validation_split=0.2, verbose=0, callbacks=[PrintDot()])
# print()


# 显示训练末期的几个训练结果
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail())


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'], label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'], label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


# plot_history(history)
# 结果发现在100epoch后, 结果反而恶化, 故更改fit

model = build_model()
# EarlyStopping测试每个epoch的训练条件, 如果经过一定熟练给定epoch后结果没有改进, 则停止训练
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)  # patience用于检查改进epochs的数量
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
print()
plot_history(history)

# 评估测试集效果
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# 使用测试集中的数据预测
test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# 误差分布
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
