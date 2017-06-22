import numpy as np
import cv2

img = cv2.imread('faces.jpg',0)
cv2.imshow('image',img)
k = cv2.waitKey(0) #鼠标停留在图片的任意位置，按任意键退出

if k == 27:# wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.imwrite('save.jpg',img)
    cv2.destroyAllWindows()



