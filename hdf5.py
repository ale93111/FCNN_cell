# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:49:32 2016

@author: alessandro
"""

import cv2
import h5py 
import numpy as np
import matplotlib.pyplot as plt

path = '/home/User/Documents/folder/'
pathout = '/home/User/Documents/folder/'

pathhdf = '/home/User/Documents/folder/' #path to hdf5 files

#%%

# Number of files
numberOfFiles = 440
inputFileList = []
labelFileList = []

# Image dimensions
width = 128
height = 128
channel = 1

path1 = pathout+'a'
path2 = pathout+'b'

for i in range(numberOfFiles):
    inputFileList.append("%s%d.tif" % (path1,i+1)) 
    labelFileList.append("%s%d.tif" % (path2,i+1))

#shuffle dataset
temp = [inputFileList, labelFileList]
temp = np.transpose(temp)
temp = np.random.permutation(temp)
temp = np.transpose(temp)
inputFileList = temp[0]
labelFileList = temp[1]

#%%
train_set = {}
test_set  = {}

train_split = 0.75

size_train = int(train_split*numberOfFiles)
size_test = numberOfFiles - size_train

train_set['data']  = inputFileList[:size_train]
train_set['label'] = labelFileList[:size_train]

test_set['data']   = inputFileList[size_train::]
test_set['label']  = labelFileList[size_train::]
 
#%% Train dataset
 
todo = 'train'

f = h5py.File(pathhdf+todo+'.h5', 'w')

f.create_dataset('data',  (size_train,channel,width,height), dtype=np.float32)
f.create_dataset('label', (size_train,width,height), dtype=np.float32)#ATTENZIONE

for i in range(size_train):
    img_data  = cv2.imread(train_set['data'][i] ,-1) 
    img_label = cv2.imread(train_set['label'][i],-1)
    
    #normalize values for data pre-processing
    img_data_norm  = cv2.normalize(img_data,img_data, alpha=0, beta=1, 
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_label_norm = cv2.normalize(img_label,img_label, alpha=0, beta=1, 
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    f['data' ][i] = [img_data_norm] 
    f['label'][i] = img_label_norm.astype(int)
    
f.close()

f_txt = pathhdf + todo+'.txt'
with open(f_txt, 'w') as f:
   print(pathhdf+todo+'.h5', file = f)
   
#%% Test dataset
   
todo = 'test'

f = h5py.File(pathhdf+todo+'.h5', 'w')

f.create_dataset('data',  (size_test,channel,width,height), dtype=np.float32)
f.create_dataset('label', (size_test,width,height), dtype=np.float32)#ATTENZIONE

for i in range(size_test):
    img_data  = cv2.imread(test_set['data'][i] ,-1) 
    img_label = cv2.imread(test_set['label'][i],-1)
    
    #normalize values for data pre-processing
    img_data_norm  = cv2.normalize(img_data,img_data, alpha=0, beta=1, 
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_label_norm = cv2.normalize(img_label,img_label, alpha=0, beta=1, 
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    f['data' ][i] = [img_data_norm] 
    f['label'][i] = img_label_norm.astype(int)
    
f.close()

f_txt = pathhdf + todo+'.txt'
with open(f_txt, 'w') as f:
   print(pathhdf+todo+'.h5', file = f)