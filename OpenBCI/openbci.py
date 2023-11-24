from pyOpenBCI import OpenBCICyton
import numpy as np
import pandas as pd

fs = 250.0


def print_freq(sample):
    #Add the samples that work
    uVolts = (4500000)/24/(2**23-1) * sample
    fft_vals = np.absolute(np.fft.rfft(uVolts))
    fft_freq = np.fft.rfftfreq(len(sample), 1.0/fs)
    eeg_bands = {
        'Delta': (0, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 45)
    }
    eeg_band_fft = dict()
    for band in eeg_bands:
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])
    df = pd.DataFrame(columns=['band', 'val'])
    df['band'] = eeg_bands.keys()
    df['val'] = [eeg_band_fft[band] for band in eeg_bands]
    ax = df.plot.bar(x='band', y='val', legend=False)
    ax.set_xlabel("EEG band")
    ax.set_ylabel("Mean band Amplitude")


board = OpenBCICyton(port='COM4', daisy=True)

board.start_stream(print_freq)
