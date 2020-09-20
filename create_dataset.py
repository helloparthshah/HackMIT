import cv2
import numpy as np
import os

d = input('Alphabet: ')

dir0 = 'data1/'+d
print(dir0)

try:
    os.mkdir(dir0)
except:
    print('Already exists, using same directory')

camera = cv2.VideoCapture(0)
i = 0

while True:
    (t, frame) = camera.read()
    frame = cv2.flip(frame, 1)

    roi = frame[0:250, 0:250]

    # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(roi, (7, 7), 0)

    gray = cv2.resize(gray, (96, 96))
    # print(gray.shape)

    cv2.imwrite(dir0+'/' + str(i)+'.jpg', gray)
    i += 1
    print(dir0+'/' + str(i)+'.jpg')

    cv2.rectangle(frame, (0, 0), (250, 250), (0, 255, 0), 2)
    
    if i > 500:
        break

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

    cv2.imshow('frame', frame)

    keypress = cv2.waitKey(1)

    if keypress == 27:
        break

camera.release()
cv2.destroyAllWindows()
