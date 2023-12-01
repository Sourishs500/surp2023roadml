import pandas as pd
import numpy as np
import scipy
from sklearn import linear_model
from sklearn.preprocessing import normalize
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math

index = pd.date_range('1/1/2000', periods=9, freq='T')
series = pd.Series(range(9), index=index)

targetEnd = 10
t = [0, math.pi/4, math.pi/2, 3*math.pi/4, math.pi]
t1 = [(targetEnd / t[-1]) * x for x in t]
y = np.sin(t)

def fft_series(series, sr, time):
    N = abs(sr * (math.ceil(time[-1] - time[0])))
    T = 1/sr
    ffty = scipy.fft.fft(series)
    ffty = np.abs(ffty)
    fftx = scipy.fft.fftfreq(N, T)

    return fftx, ffty

def fft_plot(series, sr, time):
    N = sr * (math.ceil(time[-1] - time[0]))
    T = 1/sr
    ffty = scipy.fft.fft(series)
    ffty = np.abs(ffty)
    fftx = scipy.fft.fftfreq(N, T)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[1].stem(fftx, ffty)
    ax[0].plot(time, series)
    plt.grid()
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")
    return plt.show()

""" plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, y)
plt.subplot(2, 1, 2)
plt.plot(t1, y) """

#fft_plot(y, math.pi / 4.0, t)
frequency = 3
duration = 2
sampling_rate=2000 

#duration = curTimeInterval[-1] - curTimeInterval[0]
#sr = curTimeInterval[1] - curTimeInterval[0]

N = sampling_rate * duration

x = np.linspace(0, duration, N, endpoint=False)

y = np.sin((2 * np.pi) * x * frequency)

plt.plot(x, y)
plt.show()

ffty = scipy.fft.fft(y)
fftx = scipy.fft.fftfreq(N, 1 / sampling_rate)

plt.plot(fftx, np.abs(ffty))
plt.show()

fft_plot(y, sampling_rate, x)

print(type(fft_series(y, sampling_rate, x)[0]), type(fft_series(y, sampling_rate, x)[1]))
