# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:33:43 2018

@author: krish
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

from sklearn.utils import shuffle

path = os.getcwd()
data_path = path+ '/test_data'
files = glob.glob(os.path.join(data_path+'/','*jpg'))
s_files = shuffle(files, random_state=2)

# Load the VGG16 network
print("[INFO] loading network...")
custom_model = 'rbc_custom_model.h5'
#loading custom model
model = load_model(custom_model)
#file = files[0]
for file in s_files:
    # Load the image using OpenCV
    orig = cv2.imread(file)

    # Load the image using Keras helper ultility
    print("[INFO] loading and preprocessing image...")
    image = image_utils.load_img(file, target_size=(224, 224))
    image = image_utils.img_to_array(image)

    # Convert (3, 224, 224) to (1, 3, 224, 224)
    # Here "1" is the number of images passed to network
    # We need it for passing batch containing serveral images in real project
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)


    # Classify the image
    print("[INFO] classifying image...")
    preds = model.predict(image)
    (inID, label, prob) = decode_predictions_custom(preds)[0][0]

    # Display the predictions
    print("RBC ID: {}, Label: {}, Prob: {}".format(inID, label, prob))
    cv2.putText(orig, "Label: {}, Prob: {}".format(label, prob), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow("Classification", orig)
    cv2.waitKey(0)
cv2.destroyAllWindows()
sys.exit()