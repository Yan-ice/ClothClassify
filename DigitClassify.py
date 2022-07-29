# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

import os

import DataGenerator
import util

os.environ['TP_CPP_MIN_LOG_LEVEL'] = '2'


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(util.class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         util.class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(len(util.class_names)))
    plt.yticks([])
    thisplot = plt.bar(range(len(util.class_names)), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


probability_model = None


def train():
    # print(tf.__version__)

    train_images, train_labels = DataGenerator.gen_dataset(10000)
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    # (train_images, train_labels), (test_images, test_labels) = example

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(util.TARGET_HEIGHT, util.TARGET_WIDTH, 1)),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(300, activation='relu'),
        keras.layers.Dense(len(util.class_names))
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=15)

    test_loss, test_acc = model.evaluate(train_images, train_labels, verbose=2)

    print('\nTest accuracy:', test_acc)

    global probability_model
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])


def predict(imgs):
    global probability_model
    return probability_model.predict(imgs)


