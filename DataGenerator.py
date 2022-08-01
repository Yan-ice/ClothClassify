import random
import time

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import util

fontpath = './fonts/'

def draw_png(name, text_content, font_size = 64):
    font = ImageFont.truetype(fontpath + name, font_size)
    text_width, text_height = font.getsize(name[:-4])
    image = Image.new('L', (text_width*2, text_height*2), 255)
    draw_table = ImageDraw.Draw(im=image)
    draw_table.text(xy=(0, 0), text=text_content, fill='#000000', font=font) # text是要显示的文本
    img = np.array(image)
    return img

def draw_data_png(font_name, text_content, font_size = 56):
    font = ImageFont.truetype(fontpath + font_name, font_size)
    image = Image.new('L', (120, 150), 0)
    draw_table = ImageDraw.Draw(im=image)
    draw_table.text(xy=(30, 30), text=text_content, fill='#FFFFFF', font=font) # text是要显示的文本
    img = np.array(image)
    return img


def gen_dataset(size):
    seed_ = time.process_time_ns()
    print("seed: ", seed_)
    random.seed(seed_)
    dts = ([], [])
    for i in range(0, size):
        label = random.randint(0, len(util.class_names)-1)
        font = util.getRandomFonts()
        img = draw_data_png(font, util.class_names[label])
        dts[0].append(util.trim(img))
        dts[1].append(label)
    return dts


def gen_test(chars):
    img = draw_png('comic.ttf', chars)
    return np.array(img)
