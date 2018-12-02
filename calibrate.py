#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:44:20 2017

@author: Rober
"""
import numpy as np
import cv2
import glob

ret, mtx, dist, rvecs, tvecs = 0

def calibrate():
    #prepare object points
    nx = 9#TODO: enter the number of inside corners in x
    ny = 6#TODO: enter the number of inside corners in y
    
    #read all images in folder
    images = glob.glob('./camera_cal/calibration*.jpg')
    
    objpoints = [] # 3D points in real world space
    imgpoints = [] # 2D points in image plane
    
    objp = np.zeros([ny*nx,3], np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x, y coordinates
    
    
    # If cornes are found, add object points, add image points and draw corners
    for name in images:
        img = cv2.imread(name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)
            
    #Undistorting a test image:
    fname = './camera_cal/calibration1.jpg'
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
    #Camera calibration, given object points, image points, and the shape of the grayscale image:
    ret, mtx, dist, rvecs,tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return  
                                                             