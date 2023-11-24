from pyOpenBCI import OpenBCICyton
import numpy as np
import pandas as pd
from scipy import signal
import time
fs = 250.0
EEG2UV_SCALE = (4500000)/24/(2**23-1)


start = time.time()


def print_freq(sample):
    #Add the samples that work
    sec = time.time() - start
    if sec == 0:
        sec = 1
    print(sec)
    sample = np.array(sample)
    uVolts = EEG2UV_SCALE * sample
    Volts = uVolts / 1000000
    win = sec * fs
    freq, psd = signal.welch(Volts, fs, nperseg = win)
    print(freq)

#/dev/ttyUSB1



print_freq([1, 2, 3, 4, 5, 6])
