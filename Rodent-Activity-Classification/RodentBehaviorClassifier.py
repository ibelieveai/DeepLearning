#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 11:40:27 2018

@author: krish
"""

from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
from sklearn import preprocessing
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image


MODEL_PATH = './models/custom_vgg_model_json.h5'
model = load_model(MODEL_PATH)

