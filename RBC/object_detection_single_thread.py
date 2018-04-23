# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 11:48:00 2018

@author: cr201692
"""

from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from  keras.applications.vgg16 import VGG16
from keras.models import load_model
from keras.preprocessing import image 
import numpy as np
from helper import decode_predictions_custom, image_preprocess
import argparse
import cv2
import numpy as np
import os
import random
import glob
import sys
from sklearn.utils import shuffle
import time

file = 'video_file_3.mpg'


print("[INFO] loading network...")
custom_model = 'rbc_custom_model.h5'
model = load_model(custom_model)

cap = cv2.VideoCapture(file)
time.sleep(2)
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  - 1

frames = 1
while frames < video_length:
    
    
    ret,original = cap.read()
    # Load the image using Keras helper ultility
    print("[INFO] loading and preprocessing image...")
    frame = cv2.resize(original, (224, 224)) 
    frame = image_utils.img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    preds = model.predict(frame)
    (inID, label, prob) = decode_predictions_custom(preds)[0][0]
    # Display the predictions
    print("RBC ID: {}, Label: {}, Prob: {}".format(inID, label, prob))
    cv2.putText(original, "Label: {}, Prob: {}".format(label, prob), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", original)
    cv2.waitKey(1)
    frames += 1
    if cv2.waitKey(1)&0xFF == ord('q'):
            break
cv2.destroyAllWindows()
sys.exit()