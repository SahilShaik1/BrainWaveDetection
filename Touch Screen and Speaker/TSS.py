import subprocess
from tkinter import *


def runAudio(name):
    # Assuming file is already downloaded
    subprocess.run(["omxplayer", "-o"], input=name)


def onClickYes():
    print(f"Yes was clicked!")


def onClickNo():
    print(f"No was clicked!")


def showPopup():
    root = Tk()
    root.geometry('100x100')
    btn = Button(root, text='Correct?', bd=5, command=onClickYes)
    btn2 = Button(root, text='Incorrect?', bd=5, command=onClickNo)
    btn.pack(side='top')
    btn2.pack(side='bottom')
    root.mainloop()
