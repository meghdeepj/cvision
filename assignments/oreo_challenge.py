import numpy as np
import cv2
import random, time
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

img_o= cv2.imread('../data/oreo.jpg')

hsv=cv2.cvtColor(img_o,cv2.COLOR_BGR2HSV)

low_blue=np.array([90, 50,50], np.float32)
up_blue=np.array([130,255,255], np.float32)

# mask=cv2.bitwise_not(cv2.inRange(hsv, low_blue, up_blue))

mask=cv2.inRange(hsv, low_blue, up_blue)

#res = cv2.bitwise_and(img_o, img_o, mask=mask)

im2, cont, hier = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


cont_sort = sorted(cont, key=cv2.contourArea, reverse=True)
# print(cv2.contourArea(cont_sort[0]))
# cont_sort.pop(0)
idx=[]
oreos=[]
for i, cont in enumerate(cont_sort):
    if cv2.contourArea(cont)>1000 and cv2.contourArea(cont) < 5000:
        idx.append(i)
        oreos.append(cont)


print('number of full oreos in the blue zone: ',len(idx))
cv2.drawContours(img_o, cont_sort, -1, (0,0,255), 2)
cv2.drawContours(img_o, oreos, -1, (0,255,0), 2)
cv2.imshow("img", img_o)
cv2.imshow("mask", mask)
cv2.waitKey(0)
