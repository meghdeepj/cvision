import numpy as np
import cv2
# import glob
import time
import pandas as pd

data = pd.read_csv('../data/points.txt', sep=" ", header=None)

points = data.to_numpy()
#print(points)
img1_pts_2d=np.array([points[:6]],np.float32)
img2_pts_2d=np.array([points[-6:]],np.float32)
pts_3d= np.array([[6,3,0],[3,3,0], [0,3,0], [6,0,0], [3,0,0], [0,0,0]], np.float32)

img_pts=[]
obj_pts=[]
obj_pts.append(pts_3d)
obj_pts.append(pts_3d)
img_pts.append(img1_pts_2d)
img_pts.append(img2_pts_2d)
#print(img1_pts2d, img2_pts2d)
img1_c=np.zeros((1200,1200,3), np.uint8)
img2_c=np.zeros((1200,1200,3), np.uint8)
i=1
j=1
for pts in img_pts[1][0]:
	img2_true=cv2.circle(img2_c,(pts[0],pts[1]),11*i, (0,255,0), 5)
	i+=1
for pts in img_pts[0][0]:
	img1_true=cv2.circle(img1_c,(pts[0],pts[1]),11*j, (0,255,0), 5)
	j+=1
cv2.namedWindow('ima_true', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ima_true', 600,600)
cv2.imshow("ima_true", img1_true)
cv2.waitKey(0)

cv2.namedWindow('ima_true', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ima_true', 600,600)
cv2.imshow("ima_true", img2_true)
cv2.waitKey(0)

# cv2.imwrite('../data/true_1.jpg', img1_true)
# cv2.imwrite('../data/true_2.jpg', img2_true)

### REORDER THE POINTS
pts2d_1=img1_pts_2d
pts2d_2=img2_pts_2d.copy()
print(pts2d_2)
pts2d_2[0][0], pts2d_2[0][1], pts2d_2[0][2]=img2_pts_2d[0][3], img2_pts_2d[0][2], img2_pts_2d[0][4]
pts2d_2[0][3], pts2d_2[0][4], pts2d_2[0][5]=img2_pts_2d[0][5], img2_pts_2d[0][1], img2_pts_2d[0][0]
pts_3d_2= np.array([[3,3,0],[0,3,0], [0,0,0], [6,3,0], [3,0,0], [6,0,0]], np.float32)
print(pts2d_2)
img_pts=[]
obj_pts=[]
obj_pts.append(pts_3d_2)
obj_pts.append(pts_3d_2)
img_pts.append(pts2d_1)
img_pts.append(pts2d_2)

i=1
j=1
for pts in img_pts[1][0]:
	img2_true=cv2.circle(img2_c,(pts[0],pts[1]),9*i, (0,0,255), 5)
	i+=1
for pts in img_pts[0][0]:
	img1_true=cv2.circle(img1_c,(pts[0],pts[1]),9*j, (0,0,255), 5)
	j+=1

cv2.namedWindow('ima_true', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ima_true', 600,600)
cv2.imshow("ima_true", img1_true)
cv2.waitKey(0)

cv2.namedWindow('ima_true', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ima_true', 600,600)
cv2.imshow("ima_true", img2_true)
cv2.waitKey(0)
ret, mtx, dst, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, (1200,1200), None, None)
print('int_mat: ', mtx)
print('dist_mat: ', dst)
print('rvecs: ', rvecs)
print('tvecs: ', tvecs)

k,j=1,1
proj_pts1=[]
proj_pts2=[]
proj_error=[]
for i in range(len(obj_pts)):

	img_pts2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dst)
	#print(img_pts2)
	if i==1:
		for pts in img_pts2:
			#print('pts: ',pts)
			img2_true=cv2.circle(img2_c,(pts[0,0],pts[0,1]),11*k, (255,0,0), 5)
			k+=1
	if i==0:
		for pts in img_pts2:
			img1_true=cv2.circle(img1_c,(pts[0,0],pts[0,1]),11*j, (255,0,0), 5)
			j+=1
	img_pts2=img_pts2.reshape(-1,2)
	error = cv2.norm(img_pts[i].reshape(-1,2),img_pts2, cv2.NORM_L2)/len(img_pts2)
	proj_error.append(error)


tot_error=np.sum(proj_error)
img1_error=np.sum(proj_error[0])
img2_error=np.sum(proj_error[1])
print('image1 mean error', img1_error)
print('image2 mean error', img2_error)
print("total mean error: ", tot_error/len(proj_error))

cv2.namedWindow('ima_true', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ima_true', 600,600)
cv2.imshow("ima_true", img1_true)
cv2.waitKey(0)

cv2.namedWindow('ima_true', cv2.WINDOW_NORMAL)
cv2.resizeWindow('ima_true', 600,600)
cv2.imshow("ima_true", img2_true)
cv2.waitKey(0)
