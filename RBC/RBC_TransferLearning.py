# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:14:03 2018

@author: krish
"""

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D , Input
from keras import backend as K
K.set_image_data_format('channels_last')
from keras.applications import resnet50, vgg16, inception_v3 
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras import optimizers
from keras.models import Model

import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import time

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

path = os.getcwd()
data_path = path+ '/data'
classes = os.listdir(data_path)
classes.remove('.DS_Store')


def image_array_transform(img_path,target_size=(224, 224)):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis =0)
    x = preprocess_input(x)
    
    
    x = x/255
    return x


def images_prep(data_path,classes):
    img_data_list=[]
    for dataset in classes:
        img_list=glob.glob(os.path.join(data_path+'/'+ dataset,'*jpg'))
        print ('Loaded the {0} images of dataset-{1} '.format(len(img_list),dataset))
        for img in img_list:
            image_array = image_array_transform(img)
            img_data_list.append(image_array)
    img_data = np.array(img_data_list)
    print("Transformed {0} images in to numpy array".format(int(img_data.shape[0])))
    return img_data


image_array = images_prep(data_path,classes)
image_array=np.rollaxis(image_array,1,0)
print ("Rolled axis of the image ",image_array.shape)
image_array=image_array[0]
print ("Numpy array shape ",image_array.shape)


# Define the number of classes
num_classes = len(classes)
num_of_samples = image_array.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

len_eat = len(glob.glob(os.path.join(data_path+'/'+'eat','*.jpg')))
len_hang = len(glob.glob(os.path.join(data_path+'/'+'hang','*.jpg')))
len_rear = len(glob.glob(os.path.join(data_path+'/'+'rear','*.jpg')))
len_drink = len(glob.glob(os.path.join(data_path+'/'+'drink','*.jpg')))

labels[0:len_drink]=0
labels[len_drink:len_eat+len_drink]=1
labels[len_eat+len_drink:len_eat+len_hang+len_drink]=2
labels[len_eat+len_hang+len_drink:len_eat+len_hang+len_drink+len_rear]=3

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(image_array,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

image_input = Input(shape=(224, 224, 3))
rbc_model = vgg16.VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

rbc_model.summary()

last_layer = rbc_model.get_layer('fc2').output
#x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model = Model(image_input, out)
custom_vgg_model.summary()


for layer in custom_vgg_model.layers[:-1]:
	layer.trainable = False
    
custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


t=time.time()
#	t = now()
hist = custom_vgg_model.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

#[INFO] loss=0.9857, accuracy: 68.2474%
###############################################################################################################################
rbc_model.summary()
last_layer = rbc_model.get_layer('block5_pool').output
x= Flatten(name='flatten')(last_layer)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
out = Dense(num_classes, activation='softmax', name='output')(x)
custom_vgg_model2 = Model(image_input, out)
custom_vgg_model2.summary()

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-3]:
	layer.trainable = False

custom_vgg_model2.summary()

custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

t=time.time()
#	t = now()
hist = custom_vgg_model2.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


#[INFO] loss=0.4920, accuracy: 78.1443%
###############################################################################################################################

for layers in range(len(custom_vgg_model2.layers)):
    for layer in custom_vgg_model2.layers[:layers+1]:
        layer.trainable = True
    print(custom_vgg_model2.summary())
    print("Number of trainable layers",layers+1)
    custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    t = time.time()
    hist = custom_vgg_model2.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
    print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
    
#################################################################################################################################
#  

for layers in range(len(custom_vgg_model2.layers)-1):
    for layer in custom_vgg_model2.layers[:-1]:
        layer.trainable = True
    for layer in custom_vgg_model2.layers[:-layers-1]:
        layer.trainable = False
    custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])
    print(custom_vgg_model2.summary())
    print("Number of trainable layers",layers+1)
    t = time.time()
    hist = custom_vgg_model2.fit(X_train, y_train, batch_size=32, epochs=12, verbose=1, validation_data=(X_test, y_test))
    print('Training time: %s' % (t - time.time()))
    (loss, accuracy) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
    
#1 layer 48%
#2 layer     
#################################################################################################################################        
import json
from keras.utils.data_utils import get_file    

CLASS_INDEX = None
CLASS_INDEX_PATH = './rbc_custom_class.json'


def decode_predictions_v1(preds, top=4):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 4:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 4)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        CLASS_INDEX = json.load(open(CLASS_INDEX_PATH))
    results = []
    for pred in preds:
        top_indices = preds.argsort()[::-1][0]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        results.append(result)
        return results

    
#model eval
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from keras.applications.imagenet_utils import decode_predictions  


def image_preprocess(path):
    x = image.load_img(path,target_size=(224,224))
    x = image.img_to_array(x)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    x = x/255
    return(x)
    
test_image_path = './test_data/hang.JPG'
image.load_img(test_image_path,target_size=(224,224))
test_image = image_preprocess(test_image_path)
y_prob = custom_vgg_model2.predict(test_image)
decode_predictions_v1(y_prob)
