
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image 
from helper import decode_predictions_custom, image_preprocess
import numpy as np
import argparse
import cv2
import numpy as np
import os
import random
import sys
import threading
from PIL import ImageGrab

label = ''
frame = None

custom_model = 'rbc_custom_model.h5'
test_image_path = './test_data/h1.jpg'


label = ''
frame = None

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global label
        # Load the VGG16 network
        print("[INFO] loading network...")
        self.model = load_model(custom_model)

        while (~(frame is None)):
            (inID, label, prob) = self.predict(frame)[0][0]

    def predict(self, frame):
        imag = image.img_to_array(frame)
        imag = cv2.resize(imag,(224,224))
        imag = np.expand_dims(imag,axis=0)
        imag = preprocess_input(imag)

        imag = imag/255
        preds = self.model.predict(imag)
        return decode_predictions_custom(preds)[0]

#cap = cv2.VideoCapture(0)
#if (cap.isOpened()):
#print("Camera OK")
#else:
#    cap.open()

keras_thread = MyThread()
keras_thread.start()

while (True):
    frame = ImageGrab.grab(bbox=(0,0,590,500))
    

    # Display the predictions
    # print("ImageNet ID: {}, Label: {}".format(inID, label))
    #cv2.putText(frame, "Label: {}".format(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    #cv2.imshow("Classification", frame)

    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;

frame = None
cv2.destroyAllWindows()
sys.exit()