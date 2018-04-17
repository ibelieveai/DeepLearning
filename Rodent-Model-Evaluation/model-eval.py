#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 10:59:58 2018

@author: krish
"""

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from keras.applications.imagenet_utils import decode_predictions

rac_model = load_model('custom_vgg_model.h5')

rac_model.summary()

classes = ['eat', 'hang', 'rear', 'drink']



def image_preprocess(path):
    x = image.load_img(path,target_size=(224,224))
    x = image.img_to_array(x)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)
    x = x/255
    return(x)
    
test_image_path = './test_images/4.jpg'
print(image.load_img(test_image_path,target_size=(224,224)))
test_image = image_preprocess(test_image_path)
y_prob = rac_model.predict(test_image)
decode_predictions_v1(y_prob)[0]


