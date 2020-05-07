import cv2
import facex
import numpy as np

clasifier = facex
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
def crop_center(img, x, y, w, h):    
    return img[y:y+h,x:x+w]
def detectFaces(img):
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(img, 1.1, 4)
    return faces

cap = cv2.VideoCapture(0)
ai = 'anger'
img = np.zeros((200, 200, 3))
ct = 0

while(True):
    # Capture frame-by-frame
    
    ret, frame = cap.read()
    height , width , layers =  frame.shape
    new_h=int(height/2)
    new_w=int(width/2)
    frame = cv2.resize(frame, (new_w, new_h)) 
    ct+=1
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = detectFaces(gray)   
    ano = ''    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, ai, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        if ct > 3:
            img = crop_center(gray, x, y , w , h)
            ai = clasifier.predict(img)[1]
            ct = 0

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()