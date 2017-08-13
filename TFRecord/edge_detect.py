# _*_ coding: utf-8 _*_
__author__ = 'wuhao'
__date__ = '2017/8/13 10:17'

import cv2
img = cv2.imread('21.jpg')

while True:

    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    edges = cv2.Canny(img, 50, 50)

    cv2.imshow('orginal', img)
    cv2.imshow('laplacian', laplacian)
    cv2.imshow('Canny', edges)
    cv2.imshow('sobelx', sobelx)
    cv2.imshow('sobely', sobely)