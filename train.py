import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras import backend as K
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers.core import Activation
from keras import regularizers
import tensorflow as tf
import keras
import os
import random
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def load_images(directory):
    images = []
    labels = []
    outputs = 0
    uniq_labels = sorted(os.listdir(directory))

    for idx, label in enumerate(uniq_labels):
        print(label, " is ready to load")
        outputs = outputs + 1
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            # x=random.randint(96,2000);
            # y=random.randint(96,2000);
            image = cv2.resize(cv2.imread(filepath), (96, 96))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)

    images, labels = unison_shuffled_copies(images, labels)

    # images = images.reshape((len(images), 96, 96, 1))
    print(images.shape)
    images = images.astype('float32')/255.0
    labels = keras.utils.to_categorical(labels)
    return(images, labels, outputs)


images, labels, noutputs = load_images(directory="./data1")
print("Data has been loaded")

""" c = list(zip(images, labels))

random.shuffle(c)

images, labels = zip(*c)

x_train = images[0:500]
y_train = labels[0:500]
x_test = images[500:]
y_test = labels[500:] """
""" sample = images[1]
print(sample.shape, labels[1])
plt.imshow(sample)
plt.show() """

''' model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), input_shape=(
        96, 96, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(noutputs, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='adam')

history = model.fit(images, labels,
                    batch_size=64, epochs=5,
                    verbose=1)
 '''

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=5, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=128, kernel_size=5,
                 padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=5,
                 padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Dropout(0.5))
model.add(Conv2D(filters=256, kernel_size=5,
                 padding='same', activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(noutputs, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(images, labels, epochs=5, batch_size=64)
# saving the model
model.save("test1.h5")

""" plt.imshow(x_test[20])
plt.show()
print(y_test[20])
res = model.predict(x_test[20])
print(res)
 """
