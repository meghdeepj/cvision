# ORB detector
import numpy as np, cv2, matplotlib.pyplot as plt, time, glob


# q_img = cv2.imread('../query_image_set/barbara.bmp', 0)
q_img = cv2.imread('../query_image_set/gosh.bmp', 0)

orb = cv2.ORB_create(scaleFactor=1.2, nlevels=4, edgeThreshold=12, patchSize=12)

img_path= glob.glob("../repo_image_set/**")

for path in img_path:
    m_img=cv2.imread(path,0)
    t0= time.time()
    kp1, des1 = orb.detectAndCompute(q_img, None)
    kp2, des2 = orb.detectAndCompute(m_img, None)
    bf= cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    k_img=cv2.drawKeypoints(q_img, kp1, None ,color=(0,255,0), flags=0)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    if matches:
        distances=[x.distance for x in matches]
        score=np.sum([x.distance for x in matches])/(max(distances)*len(distances))
        print('time taken for detect, compute, match and score: ', time.time()-t0, 's')
        print('score: ', score, 'with image', path)
        result = cv2.drawMatches(q_img, kp1, m_img, kp2, matches[0:15], None, flags = 2)
        # plt.imshow(k_img)
        # plt.show()
    cv2.namedWindow('compare', cv2.WINDOW_NORMAL)
    cv2.imshow('compare', result)
    cv2.waitKey(0)
