import cv2
import numpy as np
import time
import math
import matplotlib.pyplot as plt

def project(R,T,K,pts3d):
    # pts=np.zeros((pts3d.shape[0],3))
    # print(R)
    # rvec,_=cv2.Rodrigues(R)
    # tvec=-T[:,3].reshape(-1,1)
    # print(rvec, tvec)
    # print(pts3d[2])
    # pts2d,_=cv2.projectPoints(pts3d, rvec,tvec,K, None)
    # print(pts2d[0][0])
    RT = np.matmul(R,T)
    P=np.matmul(K,RT)
    pts2d=np.matmul(P,pts3d)
    u=pts2d[0,:]/(pts2d[2,:]+1e-10)
    v=pts2d[1,:]/(pts2d[2,:]+1e-10)

    return pts2d, np.array([u,v]).T
    # return pts2d

def getMap(pts2d, img_pts, img):
    map_x=np.zeros(img.shape, dtype=np.float32)
    map_y=np.zeros(img.shape, dtype=np.float32)
    img2d=img_pts.reshape(img.shape[0],img.shape[1],2)
    for i in range(map_x.shape[0]):
        for j in range(map_x.shape[1]):
            map_x[i,j]=int(img2d[i,j][0])
    for i in range(map_y.shape[0]):
        for j in range(map_x.shape[1]):
            map_y[i,j]=int(img2d[i,j][1])
    return map_x, map_y

def intr(focus, sx,sy,ox=0,oy=0,sh=0):
    K = np.array([[focus/sx,sh,ox],[0,focus/sy,oy],[0,0,1]], dtype=np.float32)
    return K

def extr(x=0,y=0,z=0,alpha=0,beta=0,gamma=0):
    alpha=alpha*np.pi/180
    beta=beta*np.pi/180
    gamma=gamma*np.pi/180
    T=np.array([[1, 0, 0, -x],[0, 1, 0, -y], [0, 0, 1, -z]],dtype=np.float32)
    Rx = np.array([[1, 0, 0], [0, math.cos(alpha), -math.sin(alpha)], [0, math.sin(alpha), math.cos(alpha)]], dtype=np.float32)

    Ry = np.array([[math.cos(beta), 0, -math.sin(beta)],[0, 1, 0],[math.sin(beta),0,math.cos(beta)]],dtype=np.float32)

    Rz = np.array([[math.cos(gamma), -math.sin(gamma), 0],[math.sin(gamma),math.cos(gamma), 0],[0, 0, 1]], dtype=np.float32)

    R= np.matmul(Rx, np.matmul(Ry, Rz))
    RT = np.matmul(R,T)
    return RT, R,T

def nothing(x):
    pass

if __name__=='__main__':
    img = cv2.imread('./data/check.jpg', 0)

    img_p = np.pad(img,50,'constant', constant_values=0)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    intx=np.array([[100, 0, 0], [0, 100, 0], [0, 0, 1]])

    H,W=img.shape[:2]

    x=np.linspace(-W/2,W/2,W)
    y=np.linspace(-H/2,H/2,H)

    xx,yy=np.meshgrid(x,y)

    X=xx.reshape(-1,1)
    Y=yy.reshape(-1,1)
    Z=X*0+1
    Z+=20*np.exp(-0.5*((X*1.0/W)/0.1)**2)/(0.1*np.sqrt(2*np.pi))

    pts3d=np.concatenate((X,Y,Z,X*0+1),axis=1).T
    # pts3d=np.concatenate((X,Y,Z),axis=1)
    print("3d Point in space: ",pts3d[:,-1])

    RT,R,T=extr(0,0,-W,0,0,0) #enter in pixels and degrees
    K=intr(W,1,1,W/2,H/2,0)

    # P=np.matmul(K,RT)

    pts2d, img_pts=project(R,T,K,pts3d)
    # pts2d=project(R,T,K,pts3d)

    print("Corresponding 2d point [u, v] : ",img_pts[1])
    print("shape: ",img_pts.shape)
    # pts2d=cv2.projectPoints(pts3d, R,T,K, None)
    map_x,map_y=getMap(pts2d,img_pts, img)

    outp=cv2.remap(img,map_x,map_y, cv2.INTER_LINEAR)
    # outp=cv2.remap(img,map_x, interpolation=cv2.INTER_LINEAR)

    cv2.imshow('out', outp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    WIN_NAME="gui-out"

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 700,700)
    cv2.createTrackbar("X",WIN_NAME, W,2*W,nothing)
    cv2.createTrackbar("Y",WIN_NAME, W,2*W,nothing)
    cv2.createTrackbar("Z",WIN_NAME, W,2*W,nothing)
    cv2.createTrackbar("al",WIN_NAME, 180,360,nothing)
    cv2.createTrackbar("bet",WIN_NAME, 180,360,nothing)
    cv2.createTrackbar("gamma",WIN_NAME,180,360,nothing)

    while(1):
        img = cv2.imread('./data/check.jpg', 0)

        X=-cv2.getTrackbarPos('X', WIN_NAME)+W
        Y=-cv2.getTrackbarPos('Y', WIN_NAME)+W
        Z=-cv2.getTrackbarPos('Z', WIN_NAME)
        alpha=cv2.getTrackbarPos('al', WIN_NAME)-180
        beta=cv2.getTrackbarPos('bet', WIN_NAME)-180
        gamma=cv2.getTrackbarPos('gamma', WIN_NAME)-180

        RT,R,T=extr(X,Y,Z,alpha,beta,gamma)
        K=intr(W,1,1,W/2,H/2,0)

        pts2d, img_pts=project(R,T,K,pts3d)

        map_x,map_y=getMap(pts2d,img_pts, img)

        outp=cv2.remap(img,map_x,map_y, cv2.INTER_LINEAR)

        cv2.imshow(WIN_NAME, outp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


#end of code
