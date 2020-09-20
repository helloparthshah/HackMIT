import keras
from keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

model = load_model("model.h5")


def load_images(directory):
    images = []
    labels = []
    uniq_labels = sorted(os.listdir(directory))

    for idx, label in enumerate(uniq_labels):
        print(label, " is ready to load")
        for file in os.listdir(directory + "/" + label):
            filepath = directory + "/" + label + "/" + file
            image = cv2.resize(cv2.imread(filepath), (96, 96))
            images.append(image)
            labels.append(idx)
    images = np.array(images)
    labels = np.array(labels)

    images = images.astype('float32')/255
    labels = keras.utils.to_categorical(labels)
    return(images, labels)


images, labels = load_images(directory="./data")
print("Data has been loaded")

img = images[700]
print(img.shape)
plt.imshow(img)

print(labels[20])
res = model.predict_classes(img.reshape(1,96,96,3))[0]
print(res)

plt.show()
