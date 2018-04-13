#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:28:26 2018

@author: krish
"""

import os,cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split

from keras import backend as K
import keras
#keras.__version__
K.set_image_data_format('channels_last')

from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.models import *
from keras.layers import *
from keras.applications import resnet50, inception_v3, vgg16, inception_resnet_v2
from keras import optimizers

#define project directory
path = os.getcwd()

#define data path
data_path = path + '/data'
data_dir_list = ['eat', 'hang', 'rear', 'drink']
img_data_list=[]

for dataset in data_dir_list:
	img_list=glob.glob(os.path.join(data_path+'/'+ dataset,'*jpg'))
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = img
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
		x = x/255
		print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
print(img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)


# Define the number of classes
num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')



len_eat = len(glob.glob(os.path.join(data_path+'/'+'eat','*.jpg')))
len_hang = len(glob.glob(os.path.join(data_path+'/'+'hang','*.jpg')))
len_rear = len(glob.glob(os.path.join(data_path+'/'+'rear','*.jpg')))
len_drink = len(glob.glob(os.path.join(data_path+'/'+'drink','*.jpg')))

labels[0:len_eat]=0
labels[len_eat:len_eat+len_hang]=1
labels[len_eat+len_hang:len_eat+len_hang+len_rear]=2
labels[len_eat+len_hang+len_rear:len_eat+len_hang+len_rear+len_drink]=3


names = ['eat', 'hang', 'rear', 'drink']

# convert class labels to on-hot encoding
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
x,y = shuffle(img_data,Y, random_state=2)
# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)


image_input = Input(shape=(224, 224, 3))
rbc_model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')
#rbc_model.summary()
last_layer = rbc_model.get_layer('fc2').output
#x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
rbc_custom_vgg_model = Model(image_input, out)

#Compile custom model
rbc_custom_vgg_model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])



hist = rbc_custom_vgg_model.fit(X_train, y_train, batch_size=10, epochs=20, verbose=1, validation_data=(X_test, y_test))
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = rbc_custom_vgg_model.evaluate(X_test, y_test, batch_size=10, verbose=1)
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))


#training with callbacks

from keras import callbacks

filename = 'rbc_custom_vgg_model.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

early_stopping = callbacks.EarlyStopping(monitor='val_loss',min_delta = 0 ,patience = 0, verbose =0, mode='min')

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [csv_log,early_stopping,checkpoint]

hist = rbc_custom_vgg_model.fit(X_train, y_train, batch_size=16, nb_epoch=num_epoch, verbose=1, validation_data=(X_test, y_test),callbacks=callbacks_list)








