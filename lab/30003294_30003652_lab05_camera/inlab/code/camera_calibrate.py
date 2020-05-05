import numpy as np
import cv2
# import glob
import time
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-i", help="p values for p norm")
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--featureDisplay", help="p values for p norm")
#
# args = parser.parse_args()
img1_c = cv2.imread('../data/calib_images/calib_1.jpg')
img2_c= cv2.imread('../data/calib_images/calib_2.jpg')
# img1 = cv2.imread('../query_image_set/%s.bmp'%args.i,0)
# img1 = cv2.resize(img1,None,fx=0.8,fy=0.8,interpolation = cv2.INTER_CUBIC)
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

proj_pts1=[]
proj_pts2=[]
proj_error=[]
for i in range(len(obj_pts)):
	img_pts2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dst)
	print(obj_pts[i].shape,img_pts2)
	for pts in img_pts2:
		if i==0:
			proj_pts1.append(pts)
			img1_proj=cv2.circle(img1_c,(pts[0,0],pts[0,1]),7, (0,255,0), 7)
		if i==1:
			proj_pts2.append(pts)
			img2_proj=cv2.circle(img2_c,(pts[0,0],pts[0,1]),7, (0,255,0), 7)


	img_pts2, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], mtx, dst)
	img_pts2=img_pts2.reshape(-1,2)
	error = cv2.norm(img_pts[i],img_pts2, cv2.NORM_L2)/len(img_pts2)
	proj_error.append(error)

cv2.namedWindow('p_1', cv2.WINDOW_NORMAL)
cv2.resizeWindow('p_1', 1000,1000)
cv2.imshow("p_1", img1_proj)
cv2.waitKey(0)
cv2.imwrite('../data/projected_1.jpg', img1_proj)

cv2.namedWindow('p_2', cv2.WINDOW_NORMAL)
cv2.resizeWindow('p_2', 1000,1000)
cv2.imshow("p_2", img2_proj)
cv2.waitKey(0)
cv2.imwrite('../data/projected_2.jpg', img2_proj)

tot_error=np.sum(proj_error)
img1_error=np.sum(proj_error[0])
img2_error=np.sum(proj_error[1])
print('image1 mean error', img1_error)
print('image2 mean error', img2_error)

print("total mean error: ", tot_error/len(proj_error))
