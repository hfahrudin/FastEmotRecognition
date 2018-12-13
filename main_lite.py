import cv2
import sys
import tensorflow as tf
from PIL import Image
import numpy as np
from scipy.misc import imread
from skimage.transform import resize

def crop_center(img, x, y, w, h):    
    return img[y:y+h,x:x+w]

def preprocess_img(raw):
    img = resize(raw,(200,200, 3))
    img = np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img = img/255.0
    return img

def brain(raw):
    ano = ''
    img = crop_center(raw, x, y , w , h)
    img = preprocess_img(img)
    f.set_tensor(i['index'], img.astype(np.float32))
    f.invoke()
    res = f.get_tensor(o['index'])
    classes = np.argmax(res,axis=1)
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
    return ano
    
    
print('Loading ..')

f = tf.contrib.lite.Interpreter("model.tflite")
f.allocate_tensors()
i = f.get_input_details()[0]
o = f.get_output_details()[0]

print('Load Success')

cascPath = "haarcascade_frontalface_default.xml"

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
        if ct >10:
            print(brain(gray))
            ct = 0

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


