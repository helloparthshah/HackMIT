import cv2
import tensorflow as tf
from keras.models import load_model
import numpy as np

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

# model = load_model("../model.h5")
# model = load_model("../converted_keras/keras_model.h5")
model = load_model("../converted_keras/t.h5")


class VideoCamera(object):
    def __init__(self):
        # capturing video
        self.video = cv2.VideoCapture(0)
        self.letter = ''
        f = open("../converted_keras/l.txt", "r")
        self.labels = f.read().splitlines()
        f.close()
        self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    def __del__(self):
        # releasing camera
        self.video.release()

    def get_frame(self):
        # extracting frames
        (ret, frame) = self.video.read()

        frame = cv2.flip(frame, 1)

        roi = frame[0:250, 0:250]

        gray = cv2.resize(roi, (224, 224))
        normalized_image_array = (gray.astype(np.float32) / 127.0) - 1

        self.data[0] = normalized_image_array

        res = model.predict(self.data)
        p = np.argmax(res, axis=-1)

        char = self.labels[p[0]]
        ''' gray = cv2.GaussianBlur(roi, (7, 7), 0)

        gray = cv2.resize(gray, (96, 96))

        res = model.predict(gray.reshape(1, 96, 96, 3))

        prediction = np.argmax(res, axis=-1)
        # print(res[0][prediction[0]]*100)
        # print(res)

        char = prediction[0]+65
        char -= 1
        if char > 80:
            char += 1

        if char == 64:
            char = 'Space'
        else:
            char = chr(char) '''

        self.letter = char

        cv2.rectangle(frame, (0, 0), (250, 250), (0, 255, 0), 2)
        cv2.putText(frame, char, (600, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (225, 0, 0), 2, cv2.LINE_AA)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
