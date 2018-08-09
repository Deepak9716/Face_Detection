import cv2
import numpy

cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

while(True):
    res,img =  cam.read();
    img1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(img1,1.1,5)
    for (x,y,w,z) in face:
        cv2.rectangle(img,(x,y),(x+w,y+z),(0,255,0),2)
    cv2.imshow('face',img)
    if(cv2.waitKey(1) == ord('q')):
        break
cam.release()
cv2.destroyAllWindows()