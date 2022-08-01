# TensorFlow and tf.keras
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten, MaxPool2D, Dropout, Conv2D
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


def train(train_images, train_labels, epoch_num):
    # print(tf.__version__)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    train_images = train_images.reshape((train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))

    model = Sequential()

    # 卷积层
    model.add(
        Conv2D(
            filters=256,
            kernel_size=(5, 5),
            padding='same',  # 保证卷积核大小，不够补零
            input_shape=(util.TARGET_HEIGHT, util.TARGET_WIDTH, 1),
            activation='relu'))
    # 池化层
    model.add(MaxPool2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    # 卷积层
    model.add(
        Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(
        Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu'))

    model.add(MaxPool2D(pool_size=(3, 3)))
    model.add(Dropout(0.25))

    # 扁平层
    model.add(Flatten())
    # 全连接层激活函数relu
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.25))
    # 全连接层激活函数softmax
    model.add(Dense(len(util.class_names), activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    checkpoint_path = "training/cp.ckpt"
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    model.load_weights(checkpoint_path)
    if epoch_num > 0:
        model.fit(train_images, train_labels, epochs=epoch_num, callbacks=[cp_callback])
        test_loss, test_acc = model.evaluate(train_images, train_labels, verbose=2)
        print('\nTest accuracy:', test_acc)

    global probability_model
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])


def predict(imgs):
    imgs = np.array(imgs)
    imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
    global probability_model
    return probability_model.predict(imgs)


