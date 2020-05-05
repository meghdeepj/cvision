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

def draw_book_on_img(img,corners,imgpts):
	imgpts = np.int32(imgpts).reshape(-1,2)
	# draw ground floor in green
	img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
	# draw pillars in blue color
	for i,j in zip(range(4),range(4,8)):
		img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
	# draw top layer in red color
	img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
	return img

if __name__ == '__main__':
	intx, front,side = get_data()
	gray=cv2.imread('../data/augment_book/input_1.jpg',0)
	img=cv2.imread('../data/augment_book/input_1.jpg',1)

	obj_pts=[]
	img_pts=[]

	img_pts_2d = np.array([[1343,3052],[1448,3014], [1553, 2983], [1339,3181], [1449,3145], [1547,3109]], np.float32)
	pts_3d= np.array([[0,0,0],[3,0,0], [6,0,0], [0,3,0], [3,3,0], [6,3,0]], np.float32)

	obj_pts.append(pts_3d)
	img_pts.append(img_pts_2d)

	ret,rvecs,tvecs =cv2.solvePnP(pts_3d,img_pts_2d, intx, None)

	axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

	imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, intx, None)

	out=draw_book_on_img(img,img_pts,imgpts)

	cv2.namedWindow('board1', cv2.WINDOW_NORMAL)
	cv2.resizeWindow('board1', 600,600)
	# cv2.imshow('board', img)
	cv2.imshow('board1', out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()




# cv2.namedWindow('ima_true', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('ima_true', 600,600)
# cv2.imshow("ima_true", img2_true)
# cv2.waitKey(0)
