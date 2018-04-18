# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 10:41:41 2018
@adpated from https://github.com/ChunML/DeepLearning/blob/master/camera_test.py
@author: krish
"""

from keras.preprocessing import image as image_utils
from imagenet_utils import decode_predictions
from imagenet_utils import preprocess_input
from vgg16 import VGG16
import argparse
import cv2
import numpy as np
import os
import random
import sys
import threading