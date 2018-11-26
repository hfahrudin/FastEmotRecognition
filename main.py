import cv2
import sys
from keras.models import model_from_json
import tensorflow as tf
from PIL import Image
import numpy as np
from scipy.misc import imread
from skimage.transform import resize
from keras import optimizers


def crop_center(img, x, y, w, h):    
    return img[y:y+h,x:x+w]


cascPath = "haarcascade_frontalface_default.xml"

json_file = open('model/model_vgg16.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model/model_vgg16.h5')
print("Loaded model from disk")
loaded_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9)
)

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

cap = cv2.VideoCapture(0)

img = np.zeros((200, 200, 3))
ct = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    ct+=1
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(150, 150)
        #flags = cv2.CV_HAAR_SCALE_IMAGE
    )
    
    ano = ''    
    for (x, y, w, h) in faces:
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        img = crop_center(gray, x, y , w , h)
        img = resize(img,(200,200, 3))
        img = np.expand_dims(img,axis=0)
        if(np.max(img)>1):
            img = img/255.0
        
        res = loaded_model.predict(img)
        
        classes = np.argmax(res,axis=1)
        
        if ct > 50:
            if classes == 0:
                ano = 'anger'
            elif classes == 1:
                ano = 'disgust'
            elif classes == 2:
                ano = 'fear'
            elif classes == 3:
                ano = "happy"
            elif classes == 4:
                ano = "neutral"
            elif classes == 5:
                ano = 'sadness'
            else :
                ano = 'surprised'
            ct = 0
            print(ano) 

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
