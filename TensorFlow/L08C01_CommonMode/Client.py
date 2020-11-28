import numpy as np
import matplotlib.pyplot as plt


def plot_series(time, series, format="-", start=0, end=None, label=None):
    """

    :param time: One dimension array containing time steps.
    :param series: One dimension array containing the values of time series at the given time steps.
    :param format: Line mode in the graph.
    :param start:
    :param end:
    :param label:
    :return:
    """
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    return np.where(season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """
    Repeats the same pattern at each period.
    :param time: One dimension array containing time steps.
    :param period: Define how often the seasonality repeats.
    :param amplitude: How high the pattern is.
    :param phase: Define how much the pattern is shifted relative to the the absolute time.
    :return:
    """
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def white_noise(time, noise_level=1, seed=None):
    """

    :param time:
    :param noise_level:
    :param seed:
    :return:
    """
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


# create a time series with only upward trend
time = np.arange(4 * 365 + 1)
baseline = 10
series = baseline + trend(time, 0.1)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.title('Upward Trend')
plt.show()

# create a time series with only seasonality
amplitude = 40
series = seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.title('Seasonality')
plt.show()

# create a time series with both upward trend and seasonality
slope = 0.05
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.title('Upward Trend and Seasonality')
plt.show()

# create a time series with only white noise
noise_level = 5
noise = white_noise(time, noise_level, seed=42)

plt.figure(figsize=(10, 6))
plot_series(time, noise)
plt.title('White Noise')
plt.show()

# create a time series with both white noise and seasonality
series += noise

plt.figure(figsize=(10, 6))
plot_series(time, series)
plt.title('White Noise, Upward Trend and Seasonality')
plt.show()
