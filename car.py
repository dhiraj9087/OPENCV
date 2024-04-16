import cv2
import numpy as np

car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')
ved = cv2.VideoCapture('/Users/dhirajmarathe/Desktop/iit kgp /car moving.mp4')
while ved.isOpened():
    ret,frame = ved.read()
    grey = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # bodies = body_classifier.detectMultiScale(grey,1.2,3)
    bodies = car_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,h,w) in bodies:
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,255,255),2)
        cv2.imshow('car',frame)
    if cv2.waitKey(1)==13:
        break
ved.release()
cv2.destroyAllWindows()