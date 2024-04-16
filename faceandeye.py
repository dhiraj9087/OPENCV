import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier  = cv2.CascadeClassifier('haarcascade_eye.xml')
image = cv2.imread('/Users/dhirajmarathe/Desktop/photo.png')
grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(grey,1.3,5)
# print(faces)
if faces is ():
    print("NO face found")
for (x,y,h,w) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow("img",image)
    cv2.waitKey(0)
    roi_gray = grey[y:y+h,x:x+w]
    roi_color = image[y:y+h,x:x+w]
    eyes = eye_classifier.detectMultiScale(roi_gray)
    for (ex,ey,eh,ew) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,255,0),2)
        cv2.imshow("img2",image)
        cv2.waitKey(0)

cv2.destroyAllWindows()