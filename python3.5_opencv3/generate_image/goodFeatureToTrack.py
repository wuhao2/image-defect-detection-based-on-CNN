#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
作者：
功能：跟踪温度高的区域。
"""

import numpy as np
import cv2
# import cv2.cv as cv

cap=cv2.VideoCapture("output.avi")

feasize=1
max=200
qua=0.05
mindis=7
blocksize=10
usehaar=True
k=0.04

paras=dict(maxCorners=200,
           qualityLevel=0.05,
           minDistance=7,
           blockSize=10,
           useHarrisDetector=True,
           k=0.04)

keypoints=list()
mask=None
marker=None


def getkpoints(imag,input1):
    mask1=np.zeros_like(input1)
    x=0
    y=0
    w1,h1=input1.shape
    #print 666
    #print input1.shape
    input1=input1[0:w1,200:h1]
    #print input1.shape
    try:
        w,h=imag.shape

        #w=w/2
        #h=h/2
        #print w,h
    except:
        return None

    mask1[y:y+h,x:x+w]=255

    keypoints=list()

    #kp=cv2.goodFeaturesToTrack(input1,
    #mask1,
    #**paras)
    #input1=input1.fromarray
    kp=cv2.goodFeaturesToTrack(input1,200,0.04,7)

    #cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)
    if kp is not None and len(kp)>0:
        for x,y in np.float32(kp).reshape(-1,2):
            keypoints.append((x,y))
    return keypoints


def process(image):
    grey1=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    grey=cv2.equalizeHist(grey1)
    keypoints=getkpoints(grey,grey1)
    print (keypoints)

    print (image.shape)
    if keypoints is not None and len(keypoints)>0:

        for x,y in keypoints:

            cv2.circle(image, (int(x+200),y), 3, (255,255,0))
    return image



p=cv2.imread('face.png')
p2=process(p)

cv2.imshow('my',p2)

cv2.waitKey(0)
cv2.destroyAllWindows()

# while (cap.isOpened()):
#     ret,frame=cap.read()
#     frame=process(frame)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1)&0xFF==ord('q'):
#         break
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()