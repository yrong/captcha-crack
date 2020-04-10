from PIL import Image
import requests
from io import BytesIO
import time
import os
from pathlib import Path
import re
import pyautogui

pwd = os.path.dirname(Path(__file__).resolve())
dataPath = pwd + '/data'

operatorDict = {
        "+": "add",
        "-": "sub",
        "*": "mul"
    }

def mkdir():
    if not os.path.exists(dataPath):
        os.mkdir(dataPath, 0o755)
    for num in range(10):
        path = dataPath + '/' + str(num)
        if not os.path.exists(path):
            os.mkdir(path, 0o755)
    for operator in ['add', 'sub', 'mul']:
        path = dataPath + '/' + str(operator)
        if not os.path.exists(path):
            os.mkdir(path, 0o755)


def label():
    # Opens a image in RGB mode
    response = requests.get('https://www.yooli.com/secure/verifyCode.jsp')
    im = Image.open(BytesIO(response.content))
    im.show()
    width, height = im.size

    # ask for input
    time.sleep(0.2)
    pyautogui.keyDown('alt')
    pyautogui.press('tab')
    pyautogui.keyUp('alt')
    val1 = input("first number from 0-9:")
    assert val1.isdigit()
    operator = input("operator from +/-/*:")
    assert re.match(r"\+|\-|\*", operator)
    operator = operatorDict[operator]
    val2 = input("second number from 0-9:")
    assert val2.isdigit()

    # Cropped image of above dimension
    im0 = im.crop((0, 0, 48, height))
    # Shows the image in image viewer
    # im0.show()
    t = str(int(round(time.time() * 1000)))
    im0.save(dataPath + '/' + val1 + '/' + t + '.png')

    im1 = im.crop((48, 0, 112, height))
    # Shows the image in image viewer
    # im1.show()
    t = str(int(round(time.time() * 1000)))
    im1.save(dataPath + '/' + operator + '/' + t + '.png')

    im2 = im.crop((112, 0, 150, height))
    # Shows the image in image viewer
    # im2.show()
    t = str(int(round(time.time() * 1000)))
    im2.save(dataPath + '/' + val2 + '/' + t + '.png')

if __name__ == "__main__":
    mkdir()
    for num in range(10000):
        label()