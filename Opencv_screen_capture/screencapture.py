# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:11:21 2018

@author: krish
"""

import numpy as np
import cv2
from PIL import ImageGrab
from keras.preprocessing import image

while(True):
    img = ImageGrab.grab(bbox=(0,0,590,500))
    img_np = np.array(img)
    img_ml = cv2.resize(img_np,(224,224))
    cv2.imshow("frame",img_np)
    key= cv2.waitKey(1)
    if key == 27:
        break
#vid.release()
cv2.destroyAllWindows()

img_np.shape
img_ml.shape