#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:54:55 2017

@author: Rober
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob



    
def plot3img(img1, img2,img3,titles=['Image1','Image2','Image3']):
    # Plot the result
    f, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20, 9))
    f.tight_layout()
    ax1.imshow(img1, cmap='gray')
    ax1.set_title(titles[0], fontsize=32)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(titles[1], fontsize=32)
    ax3.imshow(img3, cmap='gray')
    ax3.set_title(titles[2], fontsize=32)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show
    
    
def plot2img(img1, img2,titles=['Image1','Image2']):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1,2, figsize=(20, 9))
    f.tight_layout()
    ax1.imshow(img1, cmap='gray')
    ax1.set_title(titles[0], fontsize=32)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(titles[1], fontsize=32)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show    

