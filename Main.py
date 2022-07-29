import cv2
import numpy as np
import CharExtract
import DataGenerator
import DigitClassify
import util

DigitClassify.train()

def get_max_predict(prediction_list):
    max_index = 0
    trust = 0
    for i in range(0,len(prediction_list)):
        if prediction_list[i] > trust:
            max_index = i
            trust = prediction_list[i]
    return max_index, trust

img = DataGenerator.gen_test('Hello I WORLD!')
imgs = CharExtract.extract_char_img(img)
prediction = DigitClassify.predict(imgs)

for i in range(0, len(imgs)):
    cv2.imshow('split', imgs[i])

    tup = get_max_predict(prediction[i])
    id = util.class_names[tup[0]]
    trust = tup[1]
    print(id, trust)
    cv2.waitKey(0)



