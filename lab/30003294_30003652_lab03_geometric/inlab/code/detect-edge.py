#CS763 Lab02 30003294 30003652 Q01

import numpy as np, cv2, matplotlib.pyplot as plt
import argparse
parser= argparse.ArgumentParser()
parser.add_argument("-i", help="image")
args= parser.parse_args()
o_img= cv2.imread(args.i)
# o_img= cv2.imread('../data/butterfly.jpg')
g_img=cv2.cvtColor(o_img, cv2.COLOR_BGR2GRAY)
img=cv2.GaussianBlur(g_img,(3,3), cv2.BORDER_DEFAULT)
edge= cv2.Canny(img,80,170)

cv2.namedWindow('butter', cv2.WINDOW_NORMAL)
cv2.imshow('butter', img)
cv2.moveWindow('butter', 0,0)
cv2.namedWindow('edge', cv2.WINDOW_NORMAL)
cv2.imshow('edge', edge)
cv2.moveWindow('edge', 640,0)
cv2.waitKey(0)

cv2.imwrite('../data/butter-gray.jpg', img)
cv2.imwrite('../data/butter-edge.jpg', edge)

#end of code
