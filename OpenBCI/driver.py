import argparse
import time
import numpy as np

import brainflow

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowOperations, DetrendOperations
import matplotlib.pyplot as plt

import matplotlib.animation as animation


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
params = BrainFlowInputParams()
params.serial_port = '/dev/ttyUSB0'
board = BoardShim(0, params)
sampling_rate = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
def update(frame):
    avg_bands = [0, 0, 0, 0, 0]
    bands = [0, 0, 0, 0, 0]
    time.sleep(10)
    plt.clf()
    data = board.get_board_data()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    active = [channel for channel in eeg_channels if channel > 0]
    psd_s = []

    for channel in active:
        DataFilter.detrend(data[channel], DetrendOperations.LINEAR.value)
        psd_data = DataFilter.get_psd_welch(data[channel], nfft, nfft // 2, sampling_rate, WindowOperations.BLACKMAN_HARRIS.value)
        plt.plot(psd_data[1][:60], psd_data[0][:60])
        psd_s.append(psd_data)
        bands[0] += DataFilter.get_band_power(psd_data, 2.0, 4.0)
        bands[1] += DataFilter.get_band_power(psd_data, 4.0, 8.0)
        bands[2] += DataFilter.get_band_power(psd_data, 8.0, 13.0)
        bands[3] += DataFilter.get_band_power(psd_data, 13.0, 30.0)
        bands[4] += DataFilter.get_band_power(psd_data, 30.0, 50.0)
    avg_bands = [band / len(active) for band in bands]
    print(f"DELTA: {avg_bands[0]}")
    print(f"THETA: {avg_bands[1]}")
    print(f"ALPHA: {avg_bands[2]}")
    print(f"BETA: {avg_bands[3]}")
    print(f"GAMMA: {avg_bands[4]}")

def main():

    BoardShim.enable_dev_board_logger()


    board.prepare_session()

    # board.start_stream () # use this for default options
    board.start_stream()

    time.sleep(1)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
      # get all data and remove it from internal buffer
    ani = animation.FuncAnimation(fig, update, frames=50, interval=1000)
    plt.show()



if __name__ == "__main__":
    main()
