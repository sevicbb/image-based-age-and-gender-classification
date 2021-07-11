import cv2

def recognize(img):
    face_cascade = cv2.CascadeClassifier('face_recognition/haarcascade_frontalface_default.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_locations0 = y
        face_locations1 = x + w
        face_locations2 = y + h
        face_locations3 = x

        return (
            face_locations0,
            face_locations1,
            face_locations2,
            face_locations3,
        )
        