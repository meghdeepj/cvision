# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
# cap = cv2.VideoCapture('./data/s2l1.webm')
cap = cv2.VideoCapture(0)
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while(cap.isOpened()):

    diff = cv2.absdiff(frame1, frame2)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=6)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # cv2.drawContours(frame1,contours, -1, (0,0,255),2)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        if(cv2.contourArea(contour) < 900):
            continue
        if(h/w < 1.4):
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 0, 255), 2)
            # cv2.putText(frame1, "Status: {}".format('Social Distancing Violated'),(10,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            continue

        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'),
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('vid', frame1)
    frame1 = frame2
    ret, frame2 = cap.read()
    time.sleep(0.05)
    if ret == False:    
        break

    k = cv2.waitKey(50)
    # while k not in [ord('q'), ord('k')]:
        # k = cv2.waitKey(0)
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
