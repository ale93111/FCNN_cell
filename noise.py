# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 00:18:49 2016

@author: alessandro
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

import noise 
from noise import pnoise2, snoise2

path = '/home/User/Documents/folder/'

#%%

imgnoise = np.ones((950,950))

octaves = 8
freq = 16.0 * octaves

#base changes noise seed
width, height = imgnoise.shape
for y in range(width):
	for x in range(height):
		imgnoise[x,y] = int(snoise2( x/freq, y/freq, octaves, base = 9) * 100.0)

#generated noise contains negative values
imgnoise[:,:] +=  np.abs(np.min(imgnoise))

plt.imshow(imgnoise, cmap='gray')

print (imgnoise.shape, 'max =',np.max(imgnoise), '   min=',np.min(imgnoise))
#%%
#overwrite of an image adding simplex noise
img = cv2.imread(path+'a4.png',0)
truth = cv2.imread(path+'b4.png',0)

truth = 255 - truth #negative
truth = 0.5*truth * imgnoise/255 #addnoise only on the background, noise scaling 
imgtot = img + truth
imgtotblur = cv2.GaussianBlur(imgtot, (5,5), 0)

print (img.shape, 'max =',np.max(img))
plt.figure(figsize=(7,7))
plt.imshow(img, cmap='gray')

plt.figure(figsize=(7,7))
plt.imshow(truth, cmap='gray')

plt.figure(figsize=(12,12))
plt.imshow(imgtotblur, cmap='gray')

cv2.imwrite( path+'a4.png',imgtotblur)
#%%
#cv2.imwrite( path+'testnoise.png',imgtot)

#imgtotblur = cv2.GaussianBlur(imgtot, (5,5), 0)

#plt.figure(figsize=(12,12))
#plt.imshow(imgtot, cmap='gray')

#cv2.imwrite( path+'testnoiseblur.png',imgtotblur)
