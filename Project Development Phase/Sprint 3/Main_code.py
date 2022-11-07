


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator( rescale= 1./255, shear_range=0.1,zoom_range=0.1, horizontal_flip=True)

val_datagen = ImageDataGenerator( rescale = 1./255)

"""FOR BODY DAMAGE"""

train_path = "C:\Users\krish\OneDrive\Desktop\Dataset\body\training"
test_path = "C:\Users\krish\OneDrive\Desktop\Dataset\body\validation"

training_set = train_datagen.flow_from_directory (trainPath, target_size= (224,224), batch_size = 10, class_mode= 'categorical')

test_set = test_datagen.flow_from_directory(testPath, target_size = (224,224), batch_size=10, class_mode= 'categorical')

"""FOR THE LEVEL OF DAMAGE"""
test_set = 'C:\Users\krish\OneDrive\Desktop\Dataset\level\validation'
train_path = "C:\Users\krish\OneDrive\Desktop\Dataset\level\training"

training_set = train_datagen.flow_from_directory (trainPath, target_size= (224,224), batch_size = 10, class_mode= 'categorical')

test_set = test_datagen.flow_from_directory(testPath, target_size = (224,224), batch_size=10, class_mode= 'categorical')

"""Libraries"""

from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

vgg=VGG16 ( input_shape=imageSize + [3], weigts = 'imageset', include_top=False)

for layer in vgg.layers:
  layer.trainable = False

X = Flatten()(vgg.output)

prediction = Dense(3, activation = 'softmax')(x)

model = Model( inputs=vgg.input, outputs=prediction)

model.compile(loss = 'categorial_crossentropy', optimixer = 'adam', metrics = ['acc'])

import sys
r = model.fit_generator(
    training_set,
    validation_data = test_set,
    epochs=25,
    steps_per_epoch=979//10,
    validation_steps=171//10)


r = model1.fit_generator(
    training_set,
    validation_data = test_set,
    epochs=25,
    steps_per_epoch=979//10,
    validation_steps=171//10)


model.save('body.h5')

model.save('level.h5')

from tensorflow.keras.models import import_model
import cv2
from skimage.transform import resize

model = load_model('level.h5')

def detect(frame):
    img = cv2.resize(frame,(64,64))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    if(np.max(img)>1):
      img = img/255.0
    img = np.array([img])
    prediction = model.predict(img)
    label = ["front","rear","side"]
    preds = label[np.argmax(prediction)]
    return preds

def detect(frame):
    img = cv2.resize(frame,(64,64))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    if(np.max(img)>1):
      img = img/255.0
    img = np.array([img])
    prediction = model.predict(img)
    print(prediction)
    label = ["minor","moderate","severe"]
    preds = label[np.argmax(prediction)]
    return preds

import numpy as np

data = "C:\Users\krish\OneDrive\Desktop\Dataset\body\validation\00-front\0004.JPEG"
image = cv2.imread(data)
print(detect(image))

import numpy as np
data = "C:\Users\krish\OneDrive\Desktop\Dataset\level\validation\02-moderate_damage\0001.JPEG"
image = cv2.imread(data)
print(detect(image))













