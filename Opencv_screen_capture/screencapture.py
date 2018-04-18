# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:11:21 2018

@author: krish
"""

import numpy as np
import cv2
from PIL import ImageGrab

img = ImageGrab.grab(bbox=(100,100,400,400))
img_np = np.array(img)
frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
cv2.imshow("frame",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()