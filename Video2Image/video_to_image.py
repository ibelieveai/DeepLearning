#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:06:16 2018

@author: krish
"""

import os
import glob
import cv2
from subprocess import call
from os import  getcwd
import time
import ffmpeg

path = '/Users/krish/OneDrive/OneDrive-CharlesRiverLaboratories/Research/clipped_database/video/'
extracted_images ='/Users/krish/OneDrive/OneDrive-CharlesRiverLaboratories/Learning/deeplearning/ImageExtractor/extracted_images'

def video_path_list(path):
    path_list = glob.glob(os.path.join(path,'*.mpg'))
    return path_list


def video_class(video_path):
    action = video_path.split('_')[-2]
    return action

def count_images(extracted_images):
    path_list = len(glob.glob(os.path.join(extracted_images,'*.jpg')))
    return path_list
    
    
    

def extract_image(new_path):
    cap = cv2.VideoCapture(new_path)
    #time_start = time.time()
    try:
        if not os.path.exists('extracted_images'):
            os.makedirs('extracted_images')
    except OSError:
        print ('Error: Creating directory of extracted_images')
      
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  - 1
    print ("Number of frames: ", video_length)
    print ("Converting video..\n")
    # Start converting the video
    frame_number = 1
    while frame_number< video_length:
        cap.set(1,frame_number)
        ret,frame = cap.read()
    # Saves image of the current frame in jpg file
        name = './extracted_images/'+video_class(new_path)+'/'+'frame_' +video_class(new_path)+'_'+ str(count_images('./extracted_images/'+video_class(new_path))) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
        frame_number += 15

def main():
    """
    Extract images from videos and build a new file that we
    can use as our data input file. It can have format:
    [train|test], class, filename, nb frames
    """
    video_files = video_path_list(path)
    for file_path in video_files:
        extract_image(file_path)
        
    

if __name__ == '__main__':
    main()
    




    
    
    
    