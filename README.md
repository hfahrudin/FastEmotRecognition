## Facex : Facial Expression Classifier API powered by Tensorflow

Before use, you need to install :
- Tensorflow >= 2.0.0
- Python >= 3.6

Usage example :
```python
import cv2
import facex
clasifier = facex

im = cv2.imread("try.jpeg")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
res = clasifier.predict(gray)
```
Result example:
```python
({'anger': 0.998833, 'disgust': 0.00052476715, 'fear': 2.7101655e-06, 'happy': 1.9320699e-15, 'neutral': 2.370802e-06, 'sadness': 0.00063713803, 'surprised': 1.0851344e-09}, 'anger')
```

The result will a tuple of probability of each category and the highest probability category.