# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 17:49:09 2020

@author: PJ
"""
import cv2
import numpy as np
import sudoku1
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 9

def load_model():
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
    model.load_weights('digit_model.h5')
    return model

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720) 
    model = load_model()
    old_sudoku = None
    while True:
        ret, frame = cap.read()
        if ret == True:
            new_frame = sudoku1.test(frame, model, old_sudoku)
            cv2.imshow("Solved", new_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    cap.release()
    cv2.destroyAllWindows()