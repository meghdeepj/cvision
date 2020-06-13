import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

def img2edges(img):
    gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # fig=plt.figure(figsize=(9,12))
    # plt.imshow(gray,cmap='gray')
    # plt.show()

    gray=cv2.GaussianBlur(gray,(3,3),0)

    # fig=plt.figure(figsize=(9,12))
    # plt.imshow(gray, cmap='gray')
    # plt.show()

    edges=cv2.Canny(gray,50,180)

    return edges

def reg_o_interest(img,vertices):
    mask=np.zeros_like(img)
    match_mask_color=255

    cv2.fillPoly(mask, np.array([vertices],np.int32), match_mask_color)
    masked_img=cv2.bitwise_and(img,mask)

    return masked_img

def edge2lane(img,cropped):
    lines = cv2.HoughLinesP(cropped, rho=3, theta=np.pi/90, threshold=150,
                            lines=np.array([]), minLineLength=5, maxLineGap=20)


    for line in lines:
        (x1, y1, x2, y2) = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=5)

    return img

# road=cv2.imread('./data/road1.png', 1)

# img=cv2.cvtColor(road, cv2.COLOR_BGR2RGB)

def laneDetect(img):

    H,W,C = img.shape
    # print(H,W,C)

    edges=img2edges(img)

    # fig = plt.figure(figsize=(9, 12))
    # plt.imshow(edges, cmap='gray')
    # plt.show()

    ROI = [(0, H*0.9), (W, H*0.9), (2*W/3, 9*H/20), (W/3, 9*H/20)]
    cropped=reg_o_interest(edges,ROI)

    # fig=plt.figure(figsize=(9,12))
    # plt.imshow(cropped, cmap='gray')
    # plt.show()

    out=edge2lane(img,cropped)

    return out

if __name__=='__main__':
    cap=cv2.VideoCapture('./data/lane_det.mp4')

    while(cap.isOpened()):
        ret,frame=cap.read()
        if ret==False:
            break
        frame=laneDetect(frame)
        cv2.imshow('lane',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):
            time.sleep(3)
    cap.release()
    cv2.destroyAllWindows()
