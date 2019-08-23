#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:27:02 2019

@author: philliphungerford

Purpose: Machine learning models for plan violation task
"""
import pandas as pd
import numpy as np
from models import *
# Set seed
seeds = 0
np.random.seed(seeds)

# =============================================================================
# Run models
# =============================================================================
def main():
    
    # 1. load data -------------------------------------------------------------------
    X_cnn = np.load('../data/processed/X_cnn.npy')
    y =  np.load('../data/processed/y.npy')
    k=2

    # 2. prepare cnn data ------------------------------------------------------------
    size = 16
    h, w, d = size, size, size
    c = 1  # Channels 1 = grey scale, 3 = colour

    #taking random indices to split the dataset into train and test
    test_ids = np.random.permutation(X_cnn.shape[0])

    #splitting data and labels into train and test
    #keeping last 10 entries for testing, rest for training

    X_train_cnn = X_cnn[test_ids[:-int(np.ceil(X_cnn.shape[0]*0.25))]]
    X_test_cnn = X_cnn[test_ids[-int(np.ceil(X_cnn.shape[0]*0.25)):]]

    X_train_cnn = X_train_cnn.reshape(X_train_cnn.shape[0], h, w, d, c)
    X_test_cnn = X_test_cnn.reshape(X_test_cnn.shape[0], h, w, d, c)

    # 3. prepare the labels ------------------------------------------------------
    y = pd.get_dummies(y).values
    # last 75/25 split
    y_train = y[test_ids[:-int(np.ceil(X_cnn.shape[0]*0.25))]]
    y_test = y[test_ids[-int(np.ceil(X_cnn.shape[0]*0.25)):]]

    # tags
    my_tags = sorted(["No violation", "Violation"])

    # 4. checks -----------------------------------------------------------------
    print("CNN train shape: \t", X_train_cnn.shape)
    print("Label train shape: \t", y_train.shape)
    print("\n")
    print("CNN test shape: \t", X_test_cnn.shape)
    print("Label test shape: \t", y_test.shape)
    
    # Run models --------------------------------------------------------------------
    CNN(X_train_cnn, X_test_cnn, y_train, y_test, k, my_tags)
    pointnet_full(y, my_tags, test_ids, num_classes=2)

if __name__ == '__main__':
    main()
