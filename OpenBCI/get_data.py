import time
import numpy as np
import pandas as pd

from brainflow import BoardShim, BrainFlowInputParams
id = 0
board = BoardShim(id, '/dev/ttyUSB0')

board.prepare_session()
board.start_stream()
time.sleep(1)
data = board.get_board_data(250) ## 250Hz at 1 sec ##

board.stop_stream()
board.release_session()

eeg_channels = BoardShim.get_eeg_channels(id)
eeg_names = BoardShim.get_eeg_names(id)

df = pd.DataFrame(np.transpose(data[:, 1:]))

df_eeg = df[eeg_channels]
df_eeg.columns = eeg_names
df_eeg.to_csv('data.csv', sep = ',', index = False)

