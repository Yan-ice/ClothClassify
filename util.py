import os
import random

import cv2
import numpy as np

TARGET_HEIGHT = 32
TARGET_WIDTH = 24
fonts = []


def getRandomFonts():
    global fonts
    if len(fonts) == 0:
        for root, dirs, files in os.walk('fonts'):
            fonts = files
    return fonts[random.randint(0, len(fonts)-1)]



class_names = ['1','2','3','4','5','6','7','8','9',
                   'A','B','D','E','F','G','H','I','J','K','L','M','N','P','R','T','W','Y',
               'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


def show(image):
    image = cv2.copyMakeBorder(image, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=0)
    cv2.imshow('c', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def trim(image):
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    sp = [np.sum(i) for i in image]

    range_begin = 0
    range_end = 0
    for a in range(1, len(sp) - 1):
        if sp[a] > 0:
            if range_begin == 0:
                range_begin = a
            range_end = a
    rm_blank_row = image[range_begin-2:range_end+2, :]

    height = range_end-range_begin

    sp = [np.sum(i) for i in np.transpose(rm_blank_row)]

    range_begin = 0
    range_end = 0
    for a in range(1, len(sp) - 1):
        if sp[a] > 0:
            if range_begin == 0:
                range_begin = a
            range_end = a

    width = range_end - range_begin

    rm_blank = rm_blank_row[:,range_begin-1:range_end+1]

    need_width = height * TARGET_WIDTH // TARGET_HEIGHT
    if width < need_width:
        rm_blank = cv2.copyMakeBorder(rm_blank, 0, 0, 0, need_width-width, cv2.BORDER_CONSTANT, value=0)

    need_height = width * TARGET_HEIGHT // TARGET_WIDTH
    if height < need_height:
        rm_blank = cv2.copyMakeBorder(rm_blank, need_height-height, 0, 0, 0, cv2.BORDER_CONSTANT, value=0)

    return cv2.resize(rm_blank, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)


