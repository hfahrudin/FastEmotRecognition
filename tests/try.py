
import cv2
import facex
clasifier = facex

im = cv2.imread("try.jpeg")
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(clasifier.predict(gray))
