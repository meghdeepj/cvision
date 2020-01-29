import numpy as np, cv2, matplotlib.pyplot as plt
import argparse
parser= argparse.ArgumentParser()
parser.add_argument("-i", help="image")
args= parser.parse_args()
d_img= cv2.imread(args.i)

d_gray=cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
img=cv2.GaussianBlur(d_gray,(3,3), cv2.BORDER_DEFAULT)

kernel=np.ones((5,5), np.uint8)
edge=cv2.Canny(img,50,160)
edge=cv2.dilate(edge,kernel,iterations=1)


cv2.namedWindow('edge', cv2.WINDOW_NORMAL)
cv2.imshow('edge', edge)
cv2.waitKey(0)
contours, heirarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

img0 = img*0
for cnt in contours:
    cv2.drawContours(d_img, [cnt], 0, (0,255,0), 5)
    cv2.namedWindow('contour', cv2.WINDOW_NORMAL)
    cv2.imshow('contour', img0)
    cv2.waitKey(0)
#end of code
