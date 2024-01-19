# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 17:31:18 2024

see https://stackoverflow.com/questions/58821130/how-to-calculate-the-contrast-of-an-image

note that converting to greyscale is not necessary for these stimuli.

@author: jomisiba
"""

import cv2
import numpy as np

#%% Michelson contrast

# output contrast. Scale: 0 (low) to 1 (high)

img = cv2.imread("gaborStatic.jpg")

Y = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0]

# compute min and max of Y
min = np.min(Y)
max = np.max(Y)

# compute contrast
contrast = (max-min)/(max+min)
print(min,max,contrast)

#%% RMS contrast
img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
contrast = img_grey.std()