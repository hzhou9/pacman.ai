import pyautogui
import time
import random

list1 = ["a","b","c","c"]
while True:
    #key1 = random.choice(list1)
    #pyautogui.press(key1)
    print("send command")
    pyautogui.hotkey('command', 'r')
    time.sleep(3)