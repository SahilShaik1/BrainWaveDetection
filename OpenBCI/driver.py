import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, lfilter_zi
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    zi = lfilter_zi(b, a) * data[0]
    y, _ = lfilter(b, a, data, zi=zi)
    return y

sf = 250

df = pd.read_csv('data.csv')

delta = butter_bandpass_filter(data=df.Fp2, lowcut=0.1, highcut=4, fs=sf, order = 3)

theta = butter_bandpass_filter(data=df.Fp2, lowcut=4, highcut=8, fs=sf, order = 3)

alpha = butter_bandpass_filter(data=df.Fp2, lowcut=8, highcut=13, fs=sf, order = 3)

beta = butter_bandpass_filter(data=df.Fp2, lowcut=13, highcut=32, fs=sf, order = 3)

gamma = butter_bandpass_filter(data=df.Fp2, lowcut=32, highcut=50, fs=sf, order = 3)


fig = plt.figure(1)

plt.subplot(6, 1, 1)
plt.plot(df.Fp2, linewidth=2)

plt.subplot(6, 1, 2)
plt.plot(delta, linewidth=2)

plt.subplot(6, 1, 3)
plt.plot(theta, linewidth=2)

plt.subplot(6, 1, 4)
plt.plot(alpha, linewidth=2)

plt.subplot(6, 1, 5)
plt.plot(beta, linewidth=2)

plt.subplot(6, 1, 6)
plt.plot(gamma, linewidth=2)

plt.show()
