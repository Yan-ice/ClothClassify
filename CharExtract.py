import cv2
import numpy as np

import util

V_BLANK = 0.5
H_BLANK = 0.5


def h_split(image, gap):
    sp = [np.sum(i) for i in np.transpose(image)]
    ch_list = []
    range_begin = 0
    for a in range(1, len(sp) - 1):
        if (sp[a] > H_BLANK * 255) and (range_begin == 0):
            range_begin = a
            if len(ch_list) > 0:
                gap.append(a-ch_list[len(ch_list)-1][1])
        if (sp[a] < H_BLANK * 255) and (range_begin != 0):
            ch_list.append((range_begin-1, a+1))
            range_begin = 0
    gap.append(np.min(gap))
    return ch_list


def v_split(image):
    sp = [np.sum(i) for i in image]
    ch_list = []
    range_begin = 0
    for a in range(1, len(sp) - 1):
        if (sp[a] > V_BLANK * 255) and (range_begin == 0):
            range_begin = a
        if (sp[a] < V_BLANK * 255) and (range_begin != 0):
            if a - range_begin > util.TARGET_HEIGHT/2:
                ch_list.append((range_begin-1, a+1))
            range_begin = 0
    return ch_list


def extract_char_img(gray_img, enter, space):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    eroded = cv2.erode(gray_img, kernel)
    ret, binary = cv2.threshold(eroded, 140, 255, cv2.THRESH_BINARY_INV)

    util.show(binary)
    char_list = []

    ch_lines = v_split(binary)
    counter = 0
    for line_range in ch_lines:
        line = binary[line_range[0]:line_range[1], :]
        util.show(line)

        gap = []
        ch_cols = h_split(line, gap)
        space_hold = np.max(gap)/10+np.average(gap)
        space_counter = 0
        for h_range in ch_cols:
            single = line[:, h_range[0]:h_range[1]]
            char_list.append(util.trim(single))
            if gap[space_counter] > space_hold:
                space.append(counter)
            space_counter = space_counter + 1
            counter = counter + 1
        enter.append(counter-1)
    return np.array(char_list)
