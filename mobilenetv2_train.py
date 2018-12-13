# -*- coding: utf-8 -*-
#{'anger': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'neutral': 4, 'sadness': 5, 'surprised': 6}

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras import optimizers
import matplotlib.pyplot as plt

## other
img_width, img_height = 200, 200
nb_train_samples = 4000
nb_validation_samples = 750
top_epochs = 50
fit_epochs = 50
batch_size = 8
nb_classes = 5
nb_epoch = 15

#build CNN

model = MobileNetV2(weights='imagenet', include_top=False)

input = Input(shape=(img_width, img_height, 3),name = 'image_input')

output = model(input)

model.summary()

x = Flatten(name='flatten')(output)
#x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(7, activation='softmax', name='predictis')(x)

mobilenetv2_model = Model(inputs=input, outputs=x)

mobilenetv2_model.summary()


#Image preprocessing and image augmentation with keras
mobilenetv2_model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy']
)

# Setting learning data
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/train',
                                                 target_size = (img_width, img_height),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/val',
                                            target_size = (img_width, img_height),
                                            batch_size = 16,
                                            class_mode = 'categorical')

y_true_labels = training_set.class_indices

history = mobilenetv2_model.fit_generator(
        training_set,
        steps_per_epoch=nb_train_samples,
        epochs=nb_epoch,
        validation_data=test_set,
        validation_steps=nb_validation_samples
)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("acc.png")
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss.png")
plt.show()


#Save the model
# serialize model to JSON
my_model_json = mobilenetv2_model.to_json()
with open("mobilenetv2_model.json", "w") as json_file:
    json_file.write(my_model_json)
# serialize weights to HDF5
mobilenetv2_model.save_weights("mobilenetv2_model.h5")
print("Saved model to disk")
