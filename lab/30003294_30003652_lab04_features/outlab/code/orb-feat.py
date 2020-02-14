import numpy as np
import cv2
import glob
import time
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("-i", help="p values for p norm")
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--featureDisplay", help="p values for p norm")
#
# args = parser.parse_args()
#
#
img1 = cv2.imread('../query_image_set/barbara.bmp',0)
# img1 = cv2.imread('../query_image_set/%s.bmp'%args.i,0)
img1 = cv2.resize(img1,None,fx=0.8,fy=0.8,interpolation = cv2.INTER_CUBIC)

cv2.imshow("image",img1)
cv2.waitKey(0)

image_path = glob.glob("../repo_image_set/*")
scores_list = []

for path in image_path:

	start = time.time()

	img2 = cv2.imread(path,0)
	# img2 = cv2.imread("../repo_image_set/barbara.bmp",0)
	# img2 = cv2.imread("../repo_image_set/tiffany_ROTSCALE_1.bmp",0)
	img2 = cv2.resize(img2,None,fx=0.8,fy=0.8,interpolation = cv2.INTER_CUBIC)

	# Initiate SIFT detector
	orb = cv2.ORB_create(edgeThreshold=10,patchSize=14,nlevels=5)

	# Find the  keypoints and descriptors with SIFT
	kp1,des1 = orb.detectAndCompute(img1,None)
	kp2,des2 = orb.detectAndCompute(img2,None)

	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

	# Match descriptors
	matches = bf.match(des1,des2)

	# print("time : ",time.time() - start)


	# sort them in the order of their distance
	matches = sorted(matches, key=lambda x:x.distance)
	minimum_score = 1
	maximum_score = 0
	if matches:
		distances = [x.distance for x in matches]
		min1 = min(distances)
		max1 = max(distances)
		score = (np.sum([(x.distance) for x in matches]))/(len(distances))
		scores_list.append(score)


		img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:],None,flags=2)

		list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
		list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

	else:
		scores_list.append(1)

# print(scores_list)
indx = np.argmin(scores_list)
path = image_path[indx]
print(path)
scores_list.sort()

with open('scores.txt', 'w') as f:
    for item in scores_list[:5]:
        f.write("%s\n" % item)

img2 = cv2.imread(path,0)
# img2 = cv2.imread("../repo_image_set/barbara.bmp",0)
# img2 = cv2.imread("../repo_image_set/tiffany_ROTSCALE_1.bmp",0)
img2 = cv2.resize(img2,None,fx=0.8,fy=0.8,interpolation = cv2.INTER_CUBIC)

# Initiate SIFT detector
orb = cv2.ORB_create(edgeThreshold=10,patchSize=14,nlevels=5)

# Find the  keypoints and descriptors with SIFT
kp1,des1 = orb.detectAndCompute(img1,None)
kp2,des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

# Match descriptors
matches = bf.match(des1,des2)

# print("time : ",time.time() - start)


# sort them in the order of their distance
matches = sorted(matches, key=lambda x:x.distance)

if matches:
	img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:],None,flags=2)
	cv2.imshow("matches",img3)
	cv2.waitKey(0)
