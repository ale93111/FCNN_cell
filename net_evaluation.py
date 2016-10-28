# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 23:56:59 2016

@author: alessandro
"""

import h5py 
import numpy as np
import matplotlib.pyplot as plt

import caffe
from caffe import layers as L, params as P

path = '/home/User/Documents/folder/'

#%%

#load net
model = path + 'real_deploy.prototxt'
weights = path + 'snap_iter_1000.caffemodel'

caffe.set_mode_cpu()
#caffe.set_device(0) #per gpu

net = caffe.Classifier(model, weights,
                       raw_scale=1,
                       image_dims=(128,128)) 
#%%
                       
#load test dataset from hdf5 file
with h5py.File(path+'test.h5','r') as hf:
    dataset  = hf.get('data')
    labelset = hf.get('label')
    
    data  = np.array(dataset)
    label = np.array(labelset)

#compute predictions for every image in the test dataset
predictions = np.zeros(label.shape)

for i in range(data.shape[0]):
    input_image = np.expand_dims(data[i,0], axis=3)
    res = net.predict( [input_image] , oversample=False)
    predictions[i] = res[0,1,:,:]
    
#%% Performance parameters
    
TPR = []
FPR = []
acc = [] #accuracy
fsc = [] #fscore
mcc = [] #Matthews correlation coefficient

tresholds = np.linspace(0,1,100)
for tresh in tresholds:
    tp = np.sum( np.logical_and(predictions>tresh, label==1))
    fp = np.sum( np.logical_and(predictions>tresh, label==0))
    tn = np.sum( np.logical_and(predictions<tresh, label==0))
    fn = np.sum( np.logical_and(predictions<tresh, label==1))


    specificity = tn/(fp + tn) 
    precision   = tp/(tp + fp)
    recall      = tp/(tp + fn) #TPR
    fall_out    = fp/(fp + tn) #FPR
    accuracy    = (tp + tn)/(tp+tn+fp+fn)
    fscore      = 2*tp/(2*tp + fp + fn)
    
    mccden      = np.sqrt(tp+fp)*np.sqrt(tp+fn)*np.sqrt(tn+fp)*np.sqrt(tn+fn)
    if((tp+fp)<0.01 or (tp+fn)<0.01 or (tn+fp)<0.01 or (tn+fn)<0.01):
        mccden = 1
    mccoeff     = (tp*tn - fp*fn)/(mccden)
    
    TPR.append(recall)
    FPR.append(fall_out)
    acc.append(accuracy)
    fsc.append(fscore)
    mcc.append(mccoeff)

#%%

plt.figure(figsize=(7,7))
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.plot(FPR,TPR)

print('AUC: ', np.trapz(TPR[::-1],FPR[::-1]))
#%%

tresh = tresholds[67] #best mcc 

tp = np.sum( np.logical_and(predictions>tresh, label==1))
fp = np.sum( np.logical_and(predictions>tresh, label==0))
tn = np.sum( np.logical_and(predictions<tresh, label==0))
fn = np.sum( np.logical_and(predictions<tresh, label==1))
print('\n',
      'true_positive: ', tp, '\n',
      'true_negative: ', tn, '\n',
      'false_positive: ', fp,'\n', 
      'false_negative: ', fn)

specificity = tn/(fp + tn) 
precision   = tp/(tp + fp)
recall      = tp/(tp + fn) #TPR
fall_out    = fp/(fp + tn) #FPR
accuracy    = (tp + tn)/(tp+tn+fp+fn)
fscore      = 2*tp/(2*tp + fp + fn)
mccoeff     = (tp*tn - fp*fn)/(np.sqrt(tp+fp)*np.sqrt(tp+fn)*np.sqrt(tn+fp)*np.sqrt(tn+fn))

print('\n',
      'specificity: ',specificity, '\n',
      'precision:   ',precision,   '\n',
      'recall:      ',recall,      '\n',
      'fall out:    ',fall_out,    '\n',
      'accuracy:    ',accuracy,    '\n',      
      'fscore:      ',fscore,      '\n',      
      'mccoeff:     ',mccoeff,     '\n',
      'AUC:         ',np.trapz(TPR[::-1],FPR[::-1]))