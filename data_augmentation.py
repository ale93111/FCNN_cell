# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 17:38:56 2016

@author: alessandro
"""
import cv2
import h5py 
import numpy as np
import matplotlib.pyplot as plt

path = '/home/User/Documents/folder/'
pathout = '/home/User/Documents/folder/'

pathhdf = '/home/User/Documents/folder/' #path to hdf5 files
#%% Number of files
numberOfFiles = 55
inputFileList = []
labelFileList = []

# Image dimensions
width = 128
height = 128
channel = 1

path1 = path+'a'
path2 = path+'b'

for i in range(numberOfFiles):
    inputFileList.append("%s%d.tif" % (path1,i+1)) 
    labelFileList.append("%s%d.tif" % (path2,i+1))


#%%
# data augmentation # data augmentation # data augmentation # data augmentation 

counter = 1

for i in range(numberOfFiles):
    #read image
    img_data  = cv2.imread(inputFileList[i],-1) 
    img_label = cv2.imread(labelFileList[i],-1)
    
    #data augmentation
    for theta in [0,90,180,270]:
        img_data  = np.rot90(img_data)
        img_label = np.rot90(img_label)
        
        img_dataflip  = img_data[:,::-1]
        img_labelflip = img_label[:,::-1]
            
        #save images
        cv2.imwrite( "%s%d.tif" % (pathout+'a',counter) ,img_data)
        cv2.imwrite( "%s%d.tif" % (pathout+'b',counter) ,img_label)

        cv2.imwrite( "%s%d.tif" % (pathout+'a',counter+1) ,img_dataflip)
        cv2.imwrite( "%s%d.tif" % (pathout+'b',counter+1) ,img_labelflip)
        
        counter += 2