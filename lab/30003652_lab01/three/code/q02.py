#cs 763 lab00 part 3 q02 Display Image

import numpy as np
import cv2
import matplotlib.pyplot as plt

img_1 = cv2.imread('../data/rgb.jpeg', 1)
img_2 = cv2.imread('../data/mona.jpeg', 1)
img_3 = cv2.imread('../data/insta.png', 1)
img_4 = cv2.imread('../data/lenna.png', 1)

img=np.array([img_1, img_2, img_3, img_4])
i=0
while(True):
	cv2.imshow('img', img[i])
	k=cv2.waitKey(0)
	if k==27:
		break
	elif k & 0xFF==ord('n'):
		i=(i+1)%4
	elif k & 0xFF==ord('p'):
		i=(i-1)%4
	else:
		continue
cv2.destroyAllWindows()
#end of code