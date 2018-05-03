import keras
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import preprocess_input
from helper import decode_predictions_custom, image_preprocess
import argparse
import cv2
import numpy as np
import os
import random
import sys
from keras.models import load_model
import threading
import time
from datetime import datetime

label = ''
inID = ''
prob = ''
frame = None
file = 'video_3.mpg'
print("[INFO] loading network...")
custom_model = 'rbc_custom_model.h5'
behavior_list = []

def currtime():
    return str(datetime.now())

class MyThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)

	def run(self):
		global label,inID,prob
		# Load the VGG16 network
		print("[INFO] loading network...")
		self.model = load_model(custom_model)

		while (~(frame is None)):
			(inID, label, prob) = self.predict(frame)

	def predict(self, frame):
		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
		#image = image.transpose((2, 0, 1))
		image = image.reshape((1,) + image.shape)

		image = preprocess_input(image)
		preds = self.model.predict(image)
		return decode_predictions_custom(preds)[0][0]

cap = cv2.VideoCapture(file)
#time.sleep(1)
if (cap.isOpened()):
	print("Camera OK")
else:
	cap.open()

keras_thread = MyThread()
keras_thread.start()

while (True):
	time.sleep(0.1)
	ret, original = cap.read()

	frame = cv2.resize(original, (224, 224))
	action = (inID, label, prob,currtime())
	behavior_list.append(action)
    

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