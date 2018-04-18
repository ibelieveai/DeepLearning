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



import numpy as np
import json

from keras.utils.data_utils import get_file
from keras import backend as K

CLASS_INDEX = None
CLASS_INDEX_PATH = '/Users/krish/OneDrive/OneDrive-CharlesRiverLaboratories/Learning/deeplearning/Rodent-Model-Evaluation/rbc_custom_class_index.json'


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


