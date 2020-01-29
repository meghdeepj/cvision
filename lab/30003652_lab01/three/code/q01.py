#cs 763 lab00 part 3 q01 Image Conversion

import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('../data/rgb.jpeg', 1)

plt_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img.shape
n_img=np.zeros(img.shape)
n_img=cv2.normalize(img, n_img, 0, 1, cv2.NORM_MINMAX)
n_plt_img=np.zeros(img.shape)
n_plt_img=cv2.normalize(plt_img, n_plt_img, 0, 1, cv2.NORM_MINMAX)

f, axarr = plt.subplots(1,2)
axarr[0].imshow(plt_img)
axarr[1].imshow(n_plt_img)
plt.show()

cv2.imshow('image_og',img)
cv2.imshow('image_norm',n_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#end of code