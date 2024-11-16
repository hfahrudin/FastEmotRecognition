from skimage.transform import resize
import numpy as np 

def _preprocess_img(raw):
    img = resize(raw,(200,200, 3))
    img = np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img = img/255.0
    return img