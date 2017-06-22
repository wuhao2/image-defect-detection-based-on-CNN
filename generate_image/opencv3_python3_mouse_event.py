# _*_ coding=utf-8  _*_
# import cv2
# import numpy as np
#
# # mouse callback function
# def draw_circle(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)
#
# # Create a black image, a window and bind the function to window
# img = np.zeros((512,512,3), np.uint8)
# cv2.namedWindow('image')
# cv2.setMouseCallback('image',draw_circle)
#
# while(1):
#     cv2.imshow('image',img)
#     if cv2.waitKey(20) & 0xFF == 27:
#         break
# cv2.destroyAllWindows()
#######################################################################################

import cv2
import numpy as np

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True  #mouse is pressed
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE: #mouse is moved
        if drawing == True:
            if mode == True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
                # cv2.line(img, (ix,iy),(x,y), (255,0,0), 1)

            else:
                cv2.circle(img,(x,y),5,(0,0,255),-1)

    elif event == cv2.EVENT_LBUTTONUP: #mouse is released
        drawing = False
        if mode == True:
            cv2.rectangle(img,(ix,iy),(x,y),(255,255,255),-1)
            # cv2.line(img, (ix,iy),(x,y), (255,0,0), 1) #blue
        else:
            cv2.circle(img,(x,y),5,(0,0,255),-1)


img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF #return ascii encode
    if k == ord('m'): #pres m change mode
        mode = not mode
    elif k == 27: # press esc exit
        break

cv2.destroyAllWindows()

"""
waitKey(x);
第一个参数： 等待x ms，如果在此期间有按键按下，则立即结束并返回按下按键的
ASCII码，否则返回-1
如果x=0，那么无限等待下去，直到有按键按下
"""