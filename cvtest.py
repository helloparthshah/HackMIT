import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
from PIL import Image, ImageOps

f = open("./converted_keras/l.txt", "r")
labels = f.read().splitlines()
f.close()

# shape = (224, 224)

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

model = load_model("./converted_keras/t.h5")
print(model.summary())
camera = cv2.VideoCapture(0)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

while True:
    (t, frame) = camera.read()
    frame = cv2.flip(frame, 1)

    roi = frame[0:250, 0:250]

    # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(roi, (7, 7), 0)
    # image = ImageOps.fit(roi, (224, 224), Image.ANTIALIAS)
    # image_array = np.asarray(image)
    gray = cv2.resize(roi, (224, 224))
    normalized_image_array = (gray.astype(np.float32) / 127.0) - 1
    # res = model.predict(gray.reshape(1, 224, 224, 3))
    data[0] = normalized_image_array
    # prediction = np.argmax(res, axis=-1)
    res = model.predict(data)
    p = np.argmax(res, axis=-1)
    # print(res)

    # char = prediction[0]+65
    char = labels[p[0]]
    """ char -= 1
    if char > 80:
        char += 1

    if char == 64:
        char = 'Space'
    else:
        char = chr(char) """

    cv2.rectangle(frame, (0, 0), (250, 250), (0, 255, 0), 2)
    cv2.putText(frame, char, (300, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (225, 0, 0), 2, cv2.LINE_AA)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    cv2.imshow('frame', frame)

    keypress = cv2.waitKey(1)

    if keypress == 27:
        break

camera.release()
cv2.destroyAllWindows()
