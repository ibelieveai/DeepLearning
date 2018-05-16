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


#defining the detect function
def detect(frame, net, transform):
    height, width = frame.shape[:2]
    #Transformation of the frame which results a numpy array
    frame_t = transform(frame)[0]
    #Convert numpy array to torch tensor (Advanced matirx ) and convert RGB to GRB
    x = torch.from_numpy(frame_t).permute(2,0,1)
    #adding fake dimension for batch size as neural network only accepts the images in batches.
    x = Variable(x.unsqueeze(0))
    y = net(x) #passing the torch variable to neural network
    detections = y.data #extracting data
    scale = torch.Tensor([width,height,width,height])
    #detections = [batches, number of classes, number of occurances,tuple of {score,x0,y0,x1,y1}]
    for i in range(detections.size(1)):
        j = 0 #occurance of the class
        while detection[0,i,j,0]>0.6:#score for occurance j of class i
            pt = (detections[0,i,j,1:] * scale).numpy()
            # imagge, co-ordinates of upper left , co-ordinates of lower right, color of rectangle , thickness of text to display
            cv2.rectangle(frame,(int(pt[0]),int(pt[1])),(int(pt[2]),int(pt[3])),(255,0,0),2) 
            # image, name to print , where to print, font , color, thickness of text, continues text
            cv2.putText(frame,labelmap[i - 1], (int(pt[0]),int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255),2,cv2.LINE_AA)
            
            
            
            
            
        
    
    
    
    
    
    
    
