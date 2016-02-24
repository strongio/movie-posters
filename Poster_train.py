from __future__ import print_function
import numpy as np
import pandas as pd
import os
import h5py
from nolearn.lasagne import NeuralNet, TrainSplit
from lasagne.updates import adam
from nolearn.lasagne import BatchIterator
import pickle
import lasagne
import PosterExtras as phf
    
img_width = 299 
img_height = 299 

##Read file - it's saved as a hdf5 file
f = h5py.File("PosterData.hdf5", "r")
X_compressed = f['X']
y_compressed = f['y']
X = X_compressed[()]
y = y_compressed[()]
print('Loaded Data')

# Model Specifications
net = phf.build_GoogLeNet(img_width, img_height)
values = pickle.load(open('\models\\blvc_googlenet.pkl', 'rb'))['param values'][:-2]
lasagne.layers.set_all_param_values(net['pool5/7x7_s1'], values)

# Shift image array to BGR for pretrained caffe models
X = X[:, ::-1, :, :]


net0 = NeuralNet(
    net['softmax'],
    max_epochs=300,
    update=adam,
    update_learning_rate=.00001, #start with a really low learning rate
    #objective_l2=0.0001, 
    
    batch_iterator_train = BatchIterator(batch_size=32),
    batch_iterator_test = BatchIterator(batch_size=32),

    train_split=TrainSplit(eval_size=0.2),
    verbose=3,
)


net0.fit(X, y)
net0.save_params_to('ModelWeights')