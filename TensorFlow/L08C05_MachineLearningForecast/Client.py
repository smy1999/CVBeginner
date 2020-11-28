import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

keras = tf.keras


def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def window_dataset(series, window_size, batch_size=32, shuffle_buffer=1000):
    """
    Create a dataset of 30-step windows for training.
    :param series:
    :param window_size:
    :param batch_size:
    :param shuffle_buffer:
    :return:
    """
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


# -----Create Original Graph-----
time = np.arange(4 * 365 + 1)

slope = 0.05
baseline = 10
amplitude = 40
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

noise_level = 5
noise = white_noise(time, noise_level, seed=42)

series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)

# -----Split the graph into training dataset and validation dataset----
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

# -----Linear Model-----
# 1.自动调整下降率
# keras.backend.clear_session()  # Clear before cell data
# tf.random.set_seed(42)  # make code repeatable
# np.random.seed(42)
#
# window_size = 30
# train_set = window_dataset(x_train, window_size)
#
# model = keras.models.Sequential([keras.layers.Dense(1, input_shape=[window_size])])  # 单一dense层单一神经元
# lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10 ** (epoch / 30))  # 自动调整下降率
# optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
# model.compile(loss=keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
# history = model.fit(train_set, epochs=100, callbacks=[lr_schedule])
#
# # Plot 损失图像 发现在10**-4处loss开始震荡
# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-6, 1e-3, 0, 20])
# plt.show()

# 2.提前停止
# keras.backend.clear_session()
# tf.random.set_seed(42)
# np.random.seed(42)
#
# window_size = 30
# train_set = window_dataset(x_train, window_size)
# valid_set = window_dataset(x_valid, window_size)
#
# model = keras.models.Sequential([keras.layers.Dense(1, input_shape=[window_size])])
# optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
# model.compile(loss=keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
# # 如果patience周期内没有出现loss下降,则中断程序运行
# early_stopping = keras.callbacks.EarlyStopping(patience=10)
# model.fit(train_set, epochs=500, validation_data=valid_set, callbacks=[early_stopping])


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    forecast = model.predict(ds)
    return forecast


# lin_forecast = model_forecast(model, series[split_time - window_size:-1], window_size)[:, 0]
#
# plt.figure(figsize=(10, 6))
# plot_series(time_valid, x_valid)
# plot_series(time_valid, lin_forecast)
# plt.show()
#
# mae = keras.metrics.mean_absolute_error(x_valid, lin_forecast).numpy()
# print('MAE = ' + str(mae))

# 3.多层神经网络(early stopping, Learning Rate Scheduler同理
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

window_size = 30
train_set = window_dataset(x_train, window_size)
valid_set = window_dataset(x_valid, window_size)

model = keras.models.Sequential([
  keras.layers.Dense(10, activation="relu", input_shape=[window_size]),
  keras.layers.Dense(10, activation="relu"),
  keras.layers.Dense(1)
])

optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
model.compile(loss=keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
early_stopping = keras.callbacks.EarlyStopping(patience=10)
model.fit(train_set, epochs=500, validation_data=valid_set, callbacks=[early_stopping])

dense_forecast = model_forecast(model, series[split_time - window_size:-1], window_size)[:, 0]
plt.figure(figsize=(10, 6))
plot_series(time_valid, x_valid)
plot_series(time_valid, dense_forecast)
plt.show()

mae = keras.metrics.mean_absolute_error(x_valid, dense_forecast).numpy()
print('MAE = ' + str(mae))
