import argparse
import time
import numpy as np
import threading
import brainflow

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowOperations, DetrendOperations
import tkinter as tk

params = BrainFlowInputParams()

board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)
sampling_rate = BoardShim.get_sampling_rate(BoardIds.SYNTHETIC_BOARD.value)
nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

window = tk.Tk()
window.geometry("300x300")

Tlabel = tk.Label(text="Focus on the Road.", foreground="grey", font=('Times New Roman', 15, 'bold'))
Tlabel.pack()
Tlabel.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

def update():
    print("STARTED")
    bands = [0, 0, 0, 0, 0]
    time.sleep(10)
    data = board.get_board_data()
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD.value)
    active = [channel for channel in eeg_channels if channel > 0]
    psd_s = []

    for channel in active:
        DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
        DataFilter.perform_bandpass(data[channel], sampling_rate, 3.0, 45.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(data[channel], sampling_rate, 48.0, 52.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.perform_bandstop(data[channel], sampling_rate, 58.0, 62.0, 2,
                                    FilterTypes.BUTTERWORTH_ZERO_PHASE, 0)
        DataFilter.detrend(data[channel], DetrendOperations.LINEAR.value)
        psd_data = DataFilter.get_psd_welch(data[channel], nfft, nfft // 2, sampling_rate,
                                            WindowOperations.BLACKMAN_HARRIS.value)
        # plt.plot(psd_data[1][:60], psd_data[0][:60])
        psd_s.append(psd_data)
        bands[0] += DataFilter.get_band_power(psd_data, 2.0, 4.0)
        bands[1] += DataFilter.get_band_power(psd_data, 4.0, 8.0)
        bands[2] += DataFilter.get_band_power(psd_data, 8.0, 13.0)
        bands[3] += DataFilter.get_band_power(psd_data, 13.0, 30.0)
        bands[4] += DataFilter.get_band_power(psd_data, 30.0, 50.0)
    avg_bands = [band / len(active) for band in bands]

    ind = np.argmax(avg_bands)
    phrase = "Focus on the Road."
    if ind == 3:
        print("------------------------")
        print("SLEEP PRONE, BETAMAX")
        print(f"DELTA: {avg_bands[0]}")
        print(f"THETA: {avg_bands[1]}")
        print(f"ALPHA: {avg_bands[2]}")
        print(f"BETA: {avg_bands[3]}")
        print(f"GAMMA: {avg_bands[4]}")
        active = True
        phrase = "STAY ALERT!"

    print(f"DELTA: {avg_bands[0]}")
    print(f"THETA: {avg_bands[1]}")
    print(f"ALPHA: {avg_bands[2]}")
    print(f"BETA: {avg_bands[3]}")
    print(f"GAMMA: {avg_bands[4]}")


    if active is False:
        Tlabel.configure(text="Focus on the Road.")
    if active:
        Tlabel.configure(text="STAY ALERT!")
        window.configure(bg="red")
        Tlabel['bg'] = "red"


    window.after(1000, update)


def main():






    BoardShim.enable_dev_board_logger()


    board.prepare_session()

# board.start_stream () # use this for default options
    board.start_stream()


# data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
# get all data and remove it from internal buffer
    window.after(1000, update)
    window.mainloop()


if __name__ == "__main__":
    main()
