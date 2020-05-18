
from skimage.transform import resize
import numpy as np
import pathlib
import os

weight = os.path.join(pathlib.Path(__file__).parent.absolute(), "model_optimized.tflite")

import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=weight)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()



def _preprocess_img(raw):
    img = resize(raw,(200,200, 3))
    img = np.expand_dims(img,axis=0)
    if(np.max(img)>1):
        img = img/255.0
    return img

def _parse_result(res):
    cat = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprised']
    return dict(zip(cat, res[0]))]
    
def predict(img):
    
    input_data = np.array(_preprocess_img(img), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    return _parse_result(output_data)
