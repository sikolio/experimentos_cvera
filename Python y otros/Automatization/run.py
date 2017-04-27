import pyperclip
import pyautogui
import time
import re
from random import choice
from string import ascii_uppercase
import argparse

parser = argparse.ArgumentParser(description="Generate 1 file with all the data")
parser.add_argument('-I', '--initial', default=1)
parser.add_argument('-F', '--frames', default=3000)

args = parser.parse_args()
print(args)
click1 = {'x':1038, 'y':172, 'button':'right'}
click2 = {'x':899, 'y':260}
click3 = {'x':679, 'y':288}

clickForward = {'x':191, 'y':699}
clickSave = {'x':565, 'y':469}

i=args.initial
time.sleep(15)
    
while i <= int(args.frames):
    pyautogui.click(**click1)
    pyautogui.click(**click2)
    pyautogui.click(**click3)
    time.sleep(0.1)
    pyautogui.typewrite(''.join(choice(ascii_uppercase) for j in range(4)) + str(i), interval=0.1)
    pyautogui.click(**clickSave)
    pyautogui.click(**clickForward)
    i = i+1
