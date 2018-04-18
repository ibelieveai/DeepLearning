# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:11:21 2018

@author: krish
"""

import numpy as np
import cv2
from PIL import ImageGrab

while(True):
    img = ImageGrab.grab(bbox=(100,10,600,500))
    img_np = np.array(img)
    cv2.imshow("frame",img_np)
    key= cv2.waitKey(1)
    if key == 'q':
        break
#vid.release()
cv2.destroyAllWindows()