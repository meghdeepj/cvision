import numpy as np
import cv2
# import glob
import time
import pandas as pd


def get_data():
	intx = pd.read_csv('../data/augment_book/intrinsic.txt', sep=" ", header=None)
	intx=intx.to_numpy()
	front=cv2.imread('../data/augment_book/front.png')
	side=cv2.imread('../data/augment_book/side.png')

	return intx, front, side


if __name__ == '__main__':
	intx, front,side = get_data()

# cv2.namedWindow('ima_true', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('ima_true', 600,600)
# cv2.imshow("ima_true", img2_true)
# cv2.waitKey(0)
