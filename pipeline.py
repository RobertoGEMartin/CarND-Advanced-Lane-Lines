#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:36:32 2017

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

# Edit this function to create your own pipeline.
def pipeline(img, mtx, dist):
    
    img = np.copy(img)
    
    ########################################################################################
    ##STEP: DISTORSION CORRECTION
    ########################################################################################
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    ########################################################################################
    #STEP:  LANE FINDING
    ########################################################################################
       
    combined_binary, sxbinary, s_binary = thresholding.process_image_grad_color(img)
    
    ########################################################################################
    ########################################################################################

    
    ########################################################################################
    ##STEP: WARPER IMAGE
    #top_down, poly, _, _ = warper(combined_binary, mtx, dist, draw_poly=False)
    ########################################################################################
    
    #toolbox.plot3img(s_binary,sxbinary,color_binary)
    return undist, combined_binary