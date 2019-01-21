<p align="center">
  <img  src="https://user-images.githubusercontent.com/25025173/51177457-37460a00-18f2-11e9-8858-9c51f6c987a1.gif">
</p>

this repo contain emotion recognition model that light weight, fast, and accurate.

### How i made it ..

This is model use mobilenetv2 as architecture. I work on how this architecture can run in low spec devices. I tried this model with modification on Raspberry pi 3. 

Dataset that i used is affectnet, jaffe, ck+, and picked an image from google search. I train 3 architecture which are inception, vgg16, and mobilenetv2. If i run trained model in pc (i5 gen 7th), there are no noticeable difference between those models. But, when i run them on raspberry pi3, there are huge difference in fps. 

Inception and VGG16 averaging around 0.05 fps (i should have make documentation of it, mybad). Mobilenetv2 around 0.5 fps which is very good improvement. {erformance wise, there are no significant error but, i notice mobilenetv2 have slightly better performance compare to other models. 

To fit my implementation, i use tensorflow-lite to boost my fps. I got 3 fps after converting my tensorflow model to tf-lite model. It's tricky to install tensorflow + tflite in raspi 3 so feel free to ask me.

Dependencies:
1. python 3.5
2. tensorflow + tflite for raspberry pi
3. opencv 3.2
