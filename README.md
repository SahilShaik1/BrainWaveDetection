# Brain Wave Detector
- A Brain Wave Detector able to discern between Gamma, Beta, Alpha, Theta, and Delta Frequencies using electrodes connected to the OPENBCI Cyton Daisy Board

## Prerequisites
- An [OPENBCI Cyton Daisy Board](https://shop.openbci.com/products/cyton-daisy-biosensing-boards-16-channel)
- [Electrodes](https://shop.openbci.com/collections/frontpage?filter.p.m.my_fields.type=Electrodes&sort_by=manual) (Nearly any type of electrodes work, with each having their own advantages and disadvantages)
- (Optional) [Raspberry Pi](https://www.raspberrypi.com/products/) (Would recommend a modern model in order to ensure high performace)

## How to Run
- Download the 'ActualRun.py' file or copy its code onto a python file
- Replace the 'params.serial_port' variable's content to the associated serial port the board is connected to
- Download the brainflow, numpy, argparse, time, and matplotlib libraries
- Run the file
- The program will update every 10 seconds with new EEG data

## How to Run (without the board)
- Download the 'SyntheticRun.py' file or copy its code onto a python file
- Download the brainflow, numpy, argparse, time, and matplotlib libraries
- Run the file
- The program will update every 10 seconds with fake EEG data
