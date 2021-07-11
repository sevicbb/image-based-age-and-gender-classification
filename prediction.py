import cv2

import numpy as np
import urllib.request
import matplotlib.pyplot as plt

from keras.models import load_model
from keras.preprocessing.image import img_to_array

from face_recognition.index import recognize

from PIL import Image

def show_frame(frame):
    cv2.imshow('Age and gender prediction', frame)
    if cv2.waitKey(50000) == 27:
        cv2.destroyAllWindows()

def main():
    age_model = load_model('age_model.h5')
    gender_model = load_model('gender_model.h5')

    age_model.summary()
    gender_model.summary()

    frame = cv2.imread("temp.jpg")
    (face_locations0,
     face_locations1,
     face_locations2,
     face_locations3) = recognize(frame)

    cv2.rectangle(frame, (face_locations3, face_locations0), (face_locations1, face_locations2), (255, 200, 0), 2)

    img = frame[face_locations0 - 30 : face_locations2 + 30, face_locations3 - 30 : face_locations1 + 30]

    img_predict = cv2.resize(img, (256, 256))
    img_predict = np.array([img_predict]).reshape((1, 256, 256, 3))

    print('>> predicting age results...')
    age_results = age_model.predict(img_predict)
    print('>> age results predicts...')
    print(age_results)
    
    print('>> predicting gender results...')
    gender_results = gender_model.predict(img_predict)
    print('>> gender results predicts...')
    print(gender_results)

if __name__ == "__main__":
    main()