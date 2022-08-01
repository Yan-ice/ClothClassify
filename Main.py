import cv2
import numpy as np
import CharExtract
import DataGenerator
import DigitClassify
import util

print("generating datasets...")
train_images, train_labels = DataGenerator.gen_dataset(10)
print("training...")
DigitClassify.train(train_images, train_labels, 0)


def get_max_predict(prediction_list):
    max_index = 0
    trust = 0
    for i in range(0, len(prediction_list)):
        if prediction_list[i] > trust:
            max_index = i
            trust = prediction_list[i]
    return max_index, trust


# img = DataGenerator.gen_test('I am YJQ')
img = cv2.imread('testimage/test2.png', cv2.IMREAD_GRAYSCALE)

space = []
enter = []

imgs = CharExtract.extract_char_img(img, enter, space)
prediction = DigitClassify.predict(imgs)

str = ''
for i in range(0, len(prediction)):
    tup = get_max_predict(prediction[i])
    id = util.class_names[tup[0]]
    str = str+id
    if i in space:
        str = str + ' '
    if i in enter:
        str = str + '\n'

print(str)
