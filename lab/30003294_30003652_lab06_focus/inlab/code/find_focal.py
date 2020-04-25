import numpy as np
import cv2
import time

img1_c = cv2.imread('../data/calib_images/calib_1.jpg')
img2_c= cv2.imread('../data/calib_images/calib_2.jpg')

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 600,600)
cv2.imshow("image", img1_c)
cv2.waitKey(0)

cv2.namedWindow('image2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image2', 600,600)
cv2.imshow("image2", img2_c)
cv2.waitKey(0)

img1_pts_2d = np.array([[619,3108],[471,3110], [320, 3114], [619,3250], [471,3253], [319,3257]], np.float32)
img2_pts_2d = np.array([[1040,3117],[886,3157], [721, 3202], [1037,3274], [883,3319], [716,3368]], np.float32)

pts_3d= np.array([[6,3,0],[3,3,0], [0,3,0], [6,0,0], [3,0,0], [0,0,0]], np.float32)

img_pts=[]
obj_pts=[]
obj_pts.append(pts_3d)
obj_pts.append(pts_3d)
img_pts.append(img1_pts_2d)
img_pts.append(img2_pts_2d)


for pts in img_pts[1]:
	img2_true=cv2.circle(img2_c,(pts[0],pts[1]),7, (0,255,0), 7)

for pts in img_pts[0]:
	img1_true=cv2.circle(img1_c,(pts[0],pts[1]),7, (0,255,0), 7)

cv2.namedWindow('ima_true', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ima_true', 600,600)
cv2.imshow("ima_true", img2_true)
cv2.waitKey(0)

cv2.imwrite('../data/true_1.jpg', img1_true)
cv2.imwrite('../data/true_2.jpg', img2_true)

img1=cv2.cvtColor(img1_c, cv2.COLOR_BGR2GRAY)
img2=cv2.cvtColor(img2_c, cv2.COLOR_BGR2GRAY)
ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img1.shape[::-1], None, None)
print('int_mat: ', mtx)
print('dist_mat: ', dst)
print('rvecs: ', rvecs)
print('tvecs: ', tvecs)

#From image
h=img1.shape[0]
w=img1.shape[1]

#From intrinsic matrix
f_x=mtx[0,0]
f_y=mtx[1,1]

#Manufacturer data, width and height of sensor
W=6.4
H=4.8
print('From manufacturer\'s data, sensor is of type 1/2" with W=',W,' mm and H=', H,' mm')

F_x=f_x*(W/w)
F_y=f_y*(H/h)

print('F_x=', F_x, ' mm', '\nF_y=', F_y, ' mm')
