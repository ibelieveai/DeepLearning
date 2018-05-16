#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 15:41:09 2018

@author: krish
"""

# Importing required packages for object detection.
import torch # deeplearning framework as dynamic graphs helps us in faster computations
from torch.autograd import Variable #adagrad module responsible for gradient desent..variable class helps in converting tensors to torch variable 
import cv2 #Used to draw rectangles on top of the image
from data import BaseTransform , VOC_CLASSES as labelmap #Image transformation and class labels
from ssd import build_ssd #build helps in construct the single shot multibox detection model
import imageio #process the images

