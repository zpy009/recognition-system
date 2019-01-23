# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:21:01 2018

@author: ZHANG, Peiyi
"""


from skimage import io,img_as_ubyte
import numpy as np
import cv2
from skimage.transform import probabilistic_hough_line
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float
'''
task 1         houghcircle

function [y0detect,x0detect,Accumulator] = houghcircle(Imbinary,r,thresh)
%HOUGHCIRCLE - detects circles with specific radius in a binary image.
%
%Comments:
%       Function uses Standard Hough Transform to detect circles in a binary image.
%       According to the Hough Transform for circles, each pixel in image space
%       corresponds to a circle in Hough space and vice versa. 
%       upper left corner of image is the origin of coordinate system.
%
%Usage: [y0detect,x0detect,Accumulator] = houghcircle(Imbinary,r,thresh)
%
%Arguments:
%       Imbinary - a binary image. image pixels that have value equal to 1 are
%                  interested pixels for HOUGHLINE function.
%       r        - radius of circles.
%       thresh   - a threshold value that determines the minimum number of
%                  pixels that belong to a circle in image space. threshold must be
%                  bigger than or equal to 4(default).
%
%Returns:
%       y0detect    - row coordinates of detected circles.
%       x0detect    - column coordinates of detected circles. 
%       Accumulator - the accumulator array in Hough space.

if nargin == 2
    thresh = 4;
elseif thresh < 4
    error('treshold value must be bigger or equal to 4');
    return
end

%Voting

'''
def myprewittedge(image,r,thresh):
    if thresh==None:
        thresh=4
    elif thresh<4:
      print('treshold value must be bigger or equal to 4')  
    height=image.shape[1]#图片高度 行
    width=image.shape[0]#图片宽度 列
    image=np.array(image)
    Accumulator=np.zeros(image.shape)###建立一个空的  同图片大小的np.array
    rows,cols=image.nonzero()##非零元素的坐标
    r2=np.square(r)
    for i in range(len(rows)):
        low=rows[i]-r
        high=rows[i]+r
        if low<1:
            low=1
        if high>width:
            high=width
        for x_0 in range(low,high+1):
            y_0ffset=np.sqrt(r2-(rows[i]-x_0)**2)
            y_01=np.round(cols[i]-y_0ffset)
            y_02=np.round(cols[i]+y_0ffset)
            if (y_01<height and y_01>=1):
                Accumulator[x_0,y_01]=Accumulator[x_0,y_01]+1
            if (y_02<height and y_02>=1):
                Accumulator[x_0,y_02]=Accumulator[x_0,y_02]+1
    y0detect = []
    x0detect = []
    Accumulatormax = peak_local_max(Accumulator, min_distance=1,indices=True)
    Accumulatortemp=Accumulator-thresh
    for x in range(len(Accumulatormax)):
        if Accumulatortemp[Accumulatormax[x]]>=0:
            y0detect=y0detect.append(Accumulatormax[x][1])
            x0detect=x0detect.append(Accumulatormax[x][0])
            
        
        
    
