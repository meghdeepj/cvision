import cv2

f_cascade = cv2.CascadeClassifier('/home/meghdeep/opencv/opencv/data/haarcascades/haarcascade_frontalface_default.xml')

img = cv2.imread('test.jpg')

cap= cv2.VideoCapture(0)

while True:

    _, img = cap.read()

    faces = f_cascade.detectMultiScale(img, 1.1, 7, minSize=(10,10))

    for (x, w, y, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
