#CS763 Lab02 30003294 30003652 Q01

import numpy as np, cv2, matplotlib.pyplot as plt
import argparse
parser= argparse.ArgumentParser()
parser.add_argument("-i", help="image")
args= parser.parse_args()
d_img= cv2.imread(args.i)

# d_img= cv2.imread('../data/distorted.jpg')
cv2.namedWindow('distorted', cv2.WINDOW_NORMAL)
cv2.imshow('distorted', d_img)
cv2.waitKey(0)
#cv2.destroyAllWindows()
rows,cols=d_img.shape[:2]

d_pt=np.float32([[599,60],[659,659],[60,599]])
o_pt=np.float32([[600,0],[600,600],[0,600]])

s_fac=-60/599 #calculated using 1pt transform

H_a=np.array([[1, s_fac, 0], [s_fac, 1, 0]])

o_img=cv2.warpAffine(d_img, H_a, (rows,cols))
cv2.imwrite('../data/o_img_manual.jpg', o_img)
cv2.namedWindow('original', cv2.WINDOW_NORMAL)
cv2.imshow('original', o_img)
cv2.waitKey(0)
#plt.imshow(cv2.cvtColor(chess, cv2.COLOR_BGR2RGB))
#plt.show()
M=cv2.getAffineTransform(d_pt,o_pt)
o_img=cv2.warpAffine(d_img, H_a, (rows,cols))
cv2.imwrite('../data/o_img_getaff.jpg', o_img)

print(M)

#end of code
