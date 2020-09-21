# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:43:24 2020

@author: PJ
"""

import os

DATASET = "dataset"
LABELS = ["1","2","3","4","5","6","7","8","9"]

def load_dataset():
    for label in LABELS:
        path = os.path.join(DATASET, label)
        print(os.listdir(path))
        
    return 0

def split_dataset(dataset):
    return 0