import cv2
import numpy as np

import util

V_BLANK = 1
H_BLANK = 1


def h_split(image):
    sp = [np.sum(i) for i in np.transpose(image)]
    ch_list = []

    range_begin = 0
    for a in range(1, len(sp) - 1):
        if (sp[a] > H_BLANK * 255) and (range_begin == 0):
            range_begin = a
        if (sp[a] < H_BLANK * 255) and (range_begin != 0):
            if a - range_begin > util.TARGET_WIDTH/10:
                ch_list.append((range_begin-2, a+2))
            range_begin = 0
    return ch_list


def v_split(image):
    sp = [np.sum(i) for i in image]
    ch_list = []
    range_begin = 0
    for a in range(1, len(sp) - 1):
        if (sp[a] > V_BLANK * 255) and (range_begin == 0):
            range_begin = a
        if (sp[a] < V_BLANK * 255) and (range_begin != 0):
            if a - range_begin > util.TARGET_HEIGHT:
                ch_list.append((range_begin-2, a+2))
            range_begin = 0
    return ch_list


def extract_char_img(gray_img):
    ret, binary = cv2.threshold(gray_img, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    eroded = cv2.erode(binary, kernel)

    char_list = []

    ch_lines = v_split(eroded)
    for line_range in ch_lines:
        line = eroded[line_range[0]:line_range[1], :]
        ch_cols = h_split(line)
        for h_range in ch_cols:
            single = line[:, h_range[0]:h_range[1]]
            char_list.append(util.trim(single))
    return np.array(char_list)
