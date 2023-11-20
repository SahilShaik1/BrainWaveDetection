from pyOpenBCI import OpenBCICyton
import numpy as np



def print_freq(sample):
    #Add the samples that work
    sample = [sample[0], ...]
    uVolts = (4500000)/24/(2**23-1) * sample
    freq = abs(np.fft.fft(uVolts))
    #Freq Set:
    #Delta: < 3
    #Theta: < 7
    #Alpha: < 14
    if freq < 7:
        print("Drowsiness detected")

board = OpenBCICyton(port='COM4', daisy=False)

board.start_stream(print_freq)
