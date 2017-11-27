#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:24:53 2017

@author: juliagomes
"""

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
#from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
#from keras.optimizers import SGD
#import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data

def base_CNN(input_size, num_classes, k = 9):
    
    model = Sequential()
    
    model.add(Convolution2D(32, k, k, activation='relu', padding = 'same'))
    model.add(Convolution2D(32, k, k, activation='relu', padding = 'same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(64, k, k, activation='relu', padding = 'same'))
    model.add(Convolution2D(64, k, k, activation='relu', padding = 'same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(128, k, k, activation='relu', padding = 'same'))
    model.add(Convolution2D(128, k, k, activation='relu', padding = 'same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(256, k, k, activation='relu', padding = 'same'))
    model.add(Convolution2D(256, k, k, activation='relu', padding = 'same'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    return model

#model = base_CNN()
#sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(optimizer=sgd, loss='categorical_crossentropy')
#out = model.predict(im)
#print np.argmax(out)
