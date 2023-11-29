import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import butter, lfilter, lfilter_zi
from brainflow import BoardShim, BrainFlowInputParams
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

def animate(i):
    data = board.get_board_data(250)  ## 250Hz at 1 sec ##

    eeg_channels = BoardShim.get_eeg_channels(id)
    eeg_names = BoardShim.get_eeg_names(id)

    df = pd.DataFrame(np.transpose(data[:, 1:]))

    df_eeg = df[eeg_channels]
    df_eeg.columns = eeg_names

    delta = butter_bandpass_filter(data=df.Fp2, lowcut=0.1, highcut=4, fs=sf, order=3)

    theta = butter_bandpass_filter(data=df.Fp2, lowcut=4, highcut=8, fs=sf, order=3)

    alpha = butter_bandpass_filter(data=df.Fp2, lowcut=8, highcut=13, fs=sf, order=3)

    beta = butter_bandpass_filter(data=df.Fp2, lowcut=13, highcut=32, fs=sf, order=3)

    gamma = butter_bandpass_filter(data=df.Fp2, lowcut=32, highcut=50, fs=sf, order=3)

    ax1 = plt.subplot(6, 1, 1)
    plt.plot(df.Fp2, linewidth=2)



    ax2 = plt.subplot(6, 1, 2)
    plt.plot(delta, linewidth=2)
    ax2.set_title('Delta')

    ax3 = plt.subplot(6, 1, 3)
    plt.plot(theta, linewidth=2)
    ax3.set_title('Theta')

    ax4 = plt.subplot(6, 1, 4)
    plt.plot(alpha, linewidth=2)
    ax4.set_title('Alpha')

    ax5 = plt.subplot(6, 1, 5)
    plt.plot(beta, linewidth=2)
    ax5.set_title('Beta')

    ax6 = plt.subplot(6, 1, 6)
    plt.plot(gamma, linewidth=2)
    ax6.set_title('Gamma')



sf = 250


id = 0
params = BrainFlowInputParams()

params.serial_port = '/dev/ttyUSB0'

board = BoardShim(id, params)

board.prepare_session()
board.start_stream()
time.sleep(1)

fig = plt.figure()

ani = animation.FuncAnimation(fig=fig, func=animate, frames=50, interval=1000)
