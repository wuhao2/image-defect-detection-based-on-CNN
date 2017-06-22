# _*_ coding=utf-8 _*_
import cv2
import numpy as np

drawing = False # true if mouse is pressed
ix,iy = -1,-1
count = 0

# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,count

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True  #mouse is pressed
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE: #mouse is moved
        if drawing == True:
            # cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            cv2.line(img, (ix,iy),(x,y), (255,255,255), 1)

    elif event == cv2.EVENT_LBUTTONUP and count<=100: #mouse is released
        drawing = False
        # cv2.rectangle(img,(ix,iy),(x,y),(255,255,255),-1)
        cv2.line(img, (ix,iy),(x,y), (255,255,255), 1) #blue
        count += 1

            # cv2.imwrite('./black/dot%d.jpg'%int(count/3), img)
        cv2.imwrite('./black/dot%d.jpg'%count, img)
        cv2.destroyWindow('image')



while(1):
    img = np.zeros((512,512,3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF #return ascii encode
    if k == 27: # press esc exit
        break
cv2.destroyAllWindows()
