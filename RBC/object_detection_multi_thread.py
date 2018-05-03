# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 13:17:03 2018

@author: cr201692
"""
import keras
from keras.preprocessing import image as image_utils
from helper import decode_predictions_custom, image_preprocess
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
import argparse
import cv2
import numpy as np
import os
import random
import sys
import threading

label = ''
frame = None
custom_model = 'rbc_custom_model.h5'
file = 'video_file_3.mpg'
model = load_model(custom_model)

class MyThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		global label
		custom_model = 'rbc_custom_model.h5'
		# Load the VGG16 network
		print("[INFO] loading network...")
		self.model = load_model(custom_model)

		while (~(frame is None)):
			(inID, label, prob) = self.predict(frame)

	def predict(self, frame):
		#frame = cv2.resize(frame, (224, 224)) 
		frame = image_utils.img_to_array(frame)
		frame = np.expand_dims(frame, axis=0)
		frame = preprocess_input(frame)
		preds = self.model.predict(frame)
		return decode_predictions_custom(preds)[0][0]


cap = cv2.VideoCapture(file)
if (cap.isOpened()):
	print("Camera OK")
else:
	cap.open()


keras_thread = MyThread()
keras_thread.start()

while (True):
	ret, original = cap.read()

	frame = cv2.resize(original, (224, 224))

	# Display the predictions
	# print("ImageNet ID: {}, Label: {}".format(inID, label))
	cv2.putText(original, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
	cv2.imshow("Classification", original)

	if (cv2.waitKey(1) & 0xFF == ord('q')):
		break;

cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()