#cs 763 lab00 part 3 q03 Video Input/Output

import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('col_frame', frame)
    cv2.moveWindow('col_frame', 0,0)
    cv2.imshow('gray_frame',gray)
    cv2.moveWindow('gray_frame', 640,0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#end of code
