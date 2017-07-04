# _*_ coding=utf-8 _*_
# import numpy as np
# import cv2
# cap = cv2.VideoCapture(0)
"""
Capture Video from Camera
"""
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# # When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()
###############################################################################################
# import numpy as np
# import cv2
# """
# Playing Video from file
# """
# #this two kinds of pathname is ok
# # cap = cv2.VideoCapture('C:\\Users\\wuhao\\Desktop\\deepLearning\\神经网络-Tensorflow\\Tensorflow 1 why .mp4')
# cap = cv2.VideoCapture('C:/Users/wuhao/Desktop/deepLearning/神经网络-Tensorflow/Tensorflow 1 why .mp4')
#
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

###########################################################################################

import numpy as np
import cv2
"""
saveing a capture video
"""
cap = cv2.VideoCapture(0)
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)#1 means postive  
        # write the flipped frame
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()