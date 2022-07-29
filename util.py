import cv2
import numpy as np

TARGET_HEIGHT = 32
TARGET_WIDTH = 24

class_names = ['0','1','2','3','4','5','6','7','8','9',
                   'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
               'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


def trim(image):
    sp = [np.sum(i) for i in image]

    range_begin = 0
    range_end = 0
    for a in range(1, len(sp) - 1):
        if sp[a] > 0:
            if range_begin == 0:
                range_begin = a
            range_end = a
    rm_blank_row = image[range_begin:range_end, :]

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

    need_width = height*TARGET_WIDTH//TARGET_HEIGHT
    rm_blank = rm_blank_row[:,range_begin:range_end]
    if width < need_width:
        rm_blank = cv2.copyMakeBorder(rm_blank, 0, 0, 0, need_width-width, cv2.BORDER_CONSTANT, value=0)

    return cv2.resize(rm_blank, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_CUBIC)


