#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:34:04 2017

@author: Rober
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

import lines
import thresholding
import pipeline
import toolbox
import pickle

# Read camera calibration coefficients
with open('calibrate_camera.p', 'rb') as f:
    save_dict = pickle.load(f)
mtx = save_dict['mtx']
dist = save_dict['dist']

fname = './test_images/test1.jpg'
#Read RGB image
img = mpimg.imread(fname)
undist = cv2.undistort(img, mtx, dist, None, mtx)
top_down, poly, _,_ = lines.warper(undist, mtx, draw_poly=True)
toolbox.plot2img(poly, top_down, ['Original Image','Undistorted and Warped Image'])
  

