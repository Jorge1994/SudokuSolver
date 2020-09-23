# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:43:24 2020

@author: PJ
"""

import os
import cv2
import numpy as np
import random 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from scipy import ndimage

DATASET = "dataset"
LABELS = ["1","2","3","4","5","6","7","8","9"]
INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 9

def get_best_shift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted

def shift_according_to_center_of_mass(img):
    img = cv2.bitwise_not(img)

    # Center image according to center of mass
    shiftx,shifty = get_best_shift(img)
    shifted = shift(img,shiftx,shifty)
    img = shifted

    img = cv2.bitwise_not(img)
    return img

def load_dataset():
    dataset = []
    for label in LABELS:
        path = os.path.join(DATASET, label)
        class_num = LABELS.index(label)
        
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resize_img = cv2.resize(img_array, (28, 28))
            resize_img = shift_according_to_center_of_mass(resize_img)
            dataset.append([resize_img, class_num])
            
    return dataset

def split_dataset(dataset):
    split_ratio = 0.8 #80% Train and 20% Test
    random.shuffle(dataset)
    train_images = []
    train_lables = []
    test_images = []
    test_labels = []
    split_index = int(len(dataset) * split_ratio)
    
    # Create train set
    for i in range (split_index):
        train_images.append(dataset[i][0])
        train_lables.append(dataset[i][1])

    # Create test set
    for j in range (split_index, len(dataset)):
        test_images.append(dataset[j][0])
        test_labels.append(dataset[j][1])
    
    return train_images, train_lables, test_images, test_labels

def reshape(train_X, test_X):
    train_X = np.array(train_X)
    train_X = train_X.reshape(-1,28,28,1)
    test_X = np.array(test_X)
    test_X = test_X.reshape(-1,28,28,1)
    return train_X, test_X

def prepare(train_X, train_y, test_X, test_y):
    train_X, test_X = reshape(train_X, test_X) 
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')
    
    # Normalize data
    train_X = train_X / 255.0
    test_X = test_X / 255.0
    
    train_y = keras.utils.to_categorical(train_y, 9)
    test_y = keras.utils.to_categorical(test_y, 9)
    
    return train_X, train_y, test_X, test_y
    
def create_model(train_X, train_y, test_X, test_y):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=INPUT_SHAPE))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
    
    model.fit(train_X, train_y,
          batch_size=128,
          epochs=35,
          verbose=1,
          validation_split=0.1)
    
    model.evaluate(test_X, test_y, verbose=0)
    
    model.save('digit_model.h5')

if __name__ == "__main__":
    dataset = load_dataset()
    train_X, train_y, test_X, test_y = split_dataset(dataset)
    train_X, train_y, test_X, test_y = prepare(train_X, train_y, test_X, test_y)
    create_model(train_X, train_y, test_X, test_y)
    