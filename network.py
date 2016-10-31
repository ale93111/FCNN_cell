# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:09:20 2016

@author: alessandro
"""


import cv2
import h5py as hdf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import caffe
from caffe import layers as L, params as P

path = '/Users/name/folder/' #path to your folder

#%% Define the layers


def conv(bottom, nout, ks=3, stride=1, pad=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad,bias_term=False, weight_filler=dict(type='xavier'))
        
    return conv
    
def deconv(bottom, nout, ks=4, stride=2, pad=1):
    return L.Deconvolution(bottom, 
                           convolution_param=dict(num_output=nout, 
                                                  kernel_size=ks, 
                                                  stride=stride, pad=pad,
                                                  bias_term=False, weight_filler=dict(type='xavier')))
                                                  
def relu_conv(bottom, nout, ks=1, stride=1, pad=0):
    relu = L.ReLU(bottom, in_place=True)
    return relu, L.Convolution(relu, kernel_size=ks, stride=stride,
                               num_output=nout, pad=pad,bias_term=False, weight_filler=dict(type='xavier'))
    
def conv_relu(bottom, nout, ks=3, stride=1, pad=1,group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
        num_output=nout, pad=pad, group=group,bias_term=False, weight_filler=dict(type='xavier'))
      
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)
#%% Define the training net

def lenethdf5(hdf5 , batch_size):
    # Net: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5,ntop=2)
                                
    
    # the base net
    n.conv1, n.relu1 = conv_relu(n.data, 32)
    n.pool1 = max_pool(n.relu1)
   
    n.conv2, n.relu2 = conv_relu(n.pool1, 64)
    n.pool2 = max_pool(n.relu2)    
    n.conv3, n.relu3 = conv_relu(n.pool2, 128)
    n.pool3 = max_pool(n.relu3)      
    
    # fully convolutional
    n.fc1, n.rlfc1  = conv_relu(n.pool3, 512, ks=3, pad=1)
    
                               
    n.decov5 = deconv(n.rlfc1, 128, pad=1)
    n.relu5, n.conv5 = relu_conv(n.decov5, 128, pad=0)
    n.decov6 = deconv(n.conv5, 64, pad=1)
    n.relu6, n.conv6 = relu_conv(n.decov6, 64, pad=0)
    n.decov7 = deconv(n.conv6, 32, pad=1)
    n.relu7, n.conv7 = relu_conv(n.decov7, 32, pad=0)
    
    n.relu8, n.conv8 = relu_conv(n.conv7, 2, pad=0)
    
    n.accuracy= L.Accuracy(n.conv8, n.label)        
    n.loss =  L.SoftmaxWithLoss(n.conv8, n.label)
    
    
    
    
    return n.to_proto()

# create file prototxt for training and validation
with open(path+'real_train.prototxt', 'w') as f:
    f.write(str(lenethdf5('./train.txt', 10)))
    
with open(path+'real_test.prototxt', 'w') as f:
    f.write(str(lenethdf5('./test.txt', 10)))

#%%

train_net_path     = path+'real_train.prototxt'
test_net_path      = path+'real_test.prototxt'
solver_config_path = path+'real_solver.prototxt'

### define solver
from caffe.proto import caffe_pb2
s = caffe_pb2.SolverParameter()

# Set a seed for reproducible experiments:
# this controls for randomization in training.
s.random_seed = 0xCAFFE

# Specify locations of the train and (maybe) test networks.
s.train_net = train_net_path
s.test_net.append(test_net_path)
s.test_interval = 50  # Test after every 50 training iterations.
s.test_iter.append(10) # Test on 10 batches each time we test.

s.max_iter = 1000     # no. of times to update the net (training iterations)

s.iter_size = 1
# EDIT HERE to try different solvers
# solver types include "SGD", "Adam", and "Nesterov" among others.
s.type = "SGD"

# Set the initial learning rate for SGD.
s.base_lr = 0.001  # EDIT HERE to try different learning rates
# Set momentum to accelerate learning by
# taking weighted average of current and previous updates.
s.momentum = 0.9
# Set weight decay to regularize and prevent overfitting
s.weight_decay = 1e-3

# Set `lr_policy` to define how the learning rate changes during training.
# This is the same policy as our default LeNet.
s.lr_policy = 'inv'
s.gamma = 0.0001
s.power = 0.75

# Display the current training loss and accuracy every 50 iterations.
s.display = 50

# Snapshots are files used to store networks we've trained.
# We'll snapshot every 200 iterations.
s.snapshot = 200
s.snapshot_prefix = path+'snap'

# Train on the CPU
s.solver_mode = caffe_pb2.SolverParameter.CPU

# Write the solver to a temporary file and return its filename.
with open(solver_config_path, 'w') as f:
    f.write(str(s))

#%%
# Copy and paste the follwing commands in the terminal window
#!/usr/bin/env sh
#!set -e
#!/home/Users/caffe/build/tools/caffe train --solver=/home/Users/Documents/folder/my_hdf5_solver.prototxt $@

#%%
# Define the deploy net

def lenethdf5d():
    # our version of LeNet: a series of linear and simple nonlinear transformations
    n = caffe.NetSpec()
    #cambiano il primo e gli ultimi 2 strati
    n.data = L.Input(input_param=dict(shape=dict(dim=[1,1,128,128])))
     
    # the base net
    n.conv1, n.relu1 = conv_relu(n.data, 32)
    n.pool1 = max_pool(n.relu1)
    #n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.pool1, 64)
    n.pool2 = max_pool(n.relu2)    
    n.conv3, n.relu3 = conv_relu(n.pool2, 128)
    n.pool3 = max_pool(n.relu3)      
    
    # fully convolutional
    n.fc1, n.rlfc1  = conv_relu(n.pool3, 512, ks=3, pad=1)
    
                               
    n.decov5 = deconv(n.rlfc1, 128, pad=1)
    n.relu5, n.conv5 = relu_conv(n.decov5, 128, pad=0)
    n.decov6 = deconv(n.conv5, 64, pad=1)
    n.relu6, n.conv6 = relu_conv(n.decov6, 64, pad=0)
    n.decov7 = deconv(n.conv6, 32, pad=1)
    n.relu7, n.conv7 = relu_conv(n.decov7, 32, pad=0)
    
    n.relu8, n.conv8 = relu_conv(n.conv7, 2, pad=0)  
    
    
    n.prob = L.Softmax(n.conv8)
    
    return n.to_proto()
    
# create file prototxt for deployment
with open(path+'real_deploy.prototxt', 'w') as f:
    f.write(str(lenethdf5d()))

#%%

# Do the following in a terminal window
#/home/Users/caffe/build/tools/caffe train --solver=/home/Users/Documenti/folder/my_hdf5_solver.protxt 2>&1 | tee train.log
#!python /home/Users/caffe/tools/extra/parse_log.py train.log .

train_log = pd.read_csv(path+'train.log.train')
test_log = pd.read_csv(path+'train.log.test')
_, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()

p1, = ax1.plot(train_log["NumIters"], train_log["loss"], alpha=0.4, label='train loss')
p3, = ax2.plot(train_log["NumIters"], train_log["accuracy"], 'b', label='train accuracy')

p2, = ax1.plot(test_log["NumIters"], test_log["loss"], 'g', label='test loss')
p4, = ax2.plot(test_log["NumIters"], test_log["accuracy"], 'r', label='test accuracy')

ax1.set_xlabel('iteration')
ax1.set_ylabel('train loss')
ax2.set_ylabel('accuracy')

lgd = plt.legend( (p1, p2, p3, p4), ('train loss', 'test loss', 'train accuracy', 'test accuracy'), loc='center right' )
plt.show()


print("\n","Final train accuracy = ", train_log["accuracy"].iloc[-1],
      "\n","Final test accuracy  = ",  test_log["accuracy"].iloc[-1],"\n")

_.savefig(path+'log.png')
#%% Load the model and the deploy net

model = path + 'real_deploy.prototxt'
weights = path + 'snap_iter_1000.caffemodel'

caffe.set_mode_cpu()
#caffe.set_device(0) #per gpu

net = caffe.Classifier(model, weights,
                       raw_scale=1,
                       image_dims=(128,128)) 
                       #(1200,1600) for the whole real images 
                       #change the deploy file as well 
                       
#%% Test the net
                       
                       
input_image = cv2.imread(path+'/images/a55.tif',-1) #55,52
#input_image = caffe.io.load_image(path + 'real/a1.tif', color=False)
input_image = np.reshape(input_image, (128,128,1))
plt.figure(figsize=(7,7))
plt.imshow(input_image[:,:,0], cmap='gray')
input_image = cv2.normalize(input_image,input_image, alpha=0, beta=1, 
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                            
input_image = np.reshape(input_image, (128,128,1))

print (input_image.shape, input_image.dtype)

prediction = net.predict( [input_image], oversample=False)

print (prediction.shape, prediction.dtype)
#%%

width, height, channel = input_image.shape
res = prediction[0,1,:,:] 

#select the threshold 
plt.figure(figsize=(7,7))
plt.imshow(res>0.67, cmap='gray')
#%% 

plt.imshow(input_image[:,:,0], cmap='gray')

#%% show the result

#res = np.power(res,50)
#res = res/np.max(res)

plt.figure(figsize=(7,7))
plt.imshow(res, cmap='gray')
#%%

