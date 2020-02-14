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
img1 = cv2.imread('../retrieval/query_image_set/wall.png',0)
# img1 = cv2.imread('../query_image_set/%s.bmp'%args.i,0)
# img1 = cv2.resize(img1,None,fx=0.8,fy=0.8,interpolation = cv2.INTER_CUBIC)

cv2.imshow("image",img1)
cv2.waitKey(0)

image_path = glob.glob("../retrieval/repo_image_set/**")
scores_list = []

for path in image_path:

    start = time.time()

    img2 = cv2.imread(path,0)
	# img2 = cv2.imread("../repo_image_set/barbara.bmp",0)
	# img2 = cv2.imread("../repo_image_set/tiffany_ROTSCALE_1.bmp",0)
	#img2 = cv2.resize(img2,None,fx=0.8,fy=0.8,interpolation = cv2.INTER_CUBIC)

	# Initiate ORB detector
    sift = cv2.xfeatures2d.SIFT_create()

	# Find the  keypoints and descriptors with SIFT
    kp1,des1 = sift.detectAndCompute(img1,None)
    kp2,des2 = sift.detectAndCompute(img2,None)

	# create FLannMatcher object
    FLANN_INDEX_KDTREE=0
    index_params= dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    flann = cv2.FlannBasedMatcher(index_params,search_params)

    matches = flann.knnMatch(des1,des2,k=2)

	# print("time : ",time.time() - start)
    print(len(matches))
    goodmatch=[]
    for m in matches:
        if len(m)>1 and m[0].distance<0.7*(m[1].distance):
            goodmatch.append(m[0])

    # for m, n in matches:
    #     if m.distance < 0.7*n.distance:
    #         goodmatch.append(m)

    # img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,goodmatch,None,flags=2)
    # cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('matches', 600,600)
    # cv2.imshow("matches",img3)
    # cv2.waitKey(0)


	# sort them in the order of their distance
    goodmatch = sorted(goodmatch, key=lambda x:x.distance)
    minimum_score = 1
    maximum_score = 0
    if matches:
        distances = [x.distance for x in goodmatch]
        min1 = min(distances)
        max1 = max(distances)
		# score = (np.sum([(x.distance) for x in matches]))/((max1)*len(distances))
        score = np.sum([(x.distance) for x in goodmatch])/len(distances)
        scores_list.append(score)
        print(score)

        # img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:],None,flags=2)

        # list_kp1 = [kp1[mat.queryIdx].pt for mat in matches]
        # list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

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

print(path)
img2 = cv2.imread(path,0)
# img2 = cv2.imread("../repo_image_set/barbara.bmp",0)
# img2 = cv2.imread("../repo_image_set/tiffany_ROTSCALE_1.bmp",0)
#img2 = cv2.resize(img2,None,fx=0.8,fy=0.8,interpolation = cv2.INTER_CUBIC)

# Initiate ORB detector
sift = cv2.xfeatures2d.SIFT_create()

# Find the  keypoints and descriptors with SIFT
kp1,des1 = sift.detectAndCompute(img1,None)
kp2,des2 = sift.detectAndCompute(img2,None)

# create FLannMatcher object
FLANN_INDEX_KDTREE=0
index_params= dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=100)

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# print("time : ",time.time() - start)
print(len(matches))
goodmatch=[]
for m in matches:
    if len(m)>1 and m[0].distance<0.7*(m[1].distance):
        goodmatch.append(m[0])

# for m, n in matches:
#     if m.distance < 0.7*n.distance:
#         goodmatch.append(m)

# img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,goodmatch,None,flags=2)
# cv2.namedWindow('matches', cv2.WINDOW_NORMAL)
# cv2.resizeWindow('matches', 600,600)
# cv2.imshow("matches",img3)
# cv2.waitKey(0)
