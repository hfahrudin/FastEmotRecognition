from skimage.transform import resize
import numpy as np 
import tensorflow as tf


def input_resize(raw):
    img = resize(raw,(200,200, 3))
    img = np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img = img/255.0
    return img


def parse_result(res):
    cat = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprised']
    return dict(zip(cat, res[0]))


def create_interpreter(model_path):
    """
    Creates and allocates a new TFLite interpreter instance.
    """
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter