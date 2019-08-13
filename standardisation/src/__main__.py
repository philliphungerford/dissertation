#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:27:02 2019

@author: philliphungerford

Purpose: Machine learning models for radiology data standardisation
"""
import pandas as pd
import numpy as np
import models
# =============================================================================
# Run models
# =============================================================================
def main():
    
    # 1. parameters  ----------------------------------------------------------
    k = 18
    size = 16
    h, w, d = size, size, size
    c = 1  # Channels 1 = grey scale, 3 = colour
    
    # 2.1. load data ----------------------------------------------------------
    print("Loading data -----------------------------------------------------")
    filenames_y = pd.read_csv('data/processed/dataset2labelsorgansclean.csv')
    X_cnn = np.load('data/processed/dataset2voxels16.npy')
    X_rnn = filenames_y['organs']
    
    if k == 8:
        y_orig = filenames_y['class8'] # actually 8 different structures
    
    elif k == 18:
            y_orig = filenames_y['class18'] # actually 18 different structures


    # 2.1.2. prepare cnn data -------------------------------------------------
    # taking random indices to split the dataset into train and test
    test_ids = np.random.permutation(X_cnn.shape[0])

    # splitting data and labels into train and test
    X_train_cnn = X_cnn[test_ids[:-int(np.ceil(X_cnn.shape[0] * 0.25))]]
    X_test_cnn = X_cnn[test_ids[-int(np.ceil(X_cnn.shape[0] * 0.25)):]]

    X_train_cnn = X_train_cnn.reshape(X_train_cnn.shape[0], h, w, d, c)
    X_test_cnn = X_test_cnn.reshape(X_test_cnn.shape[0], h, w, d, c)

    # 2.1.3. prepare rnn data -------------------------------------------------

    MAX_NB_CHARS = 26
    MAX_SEQUENCE_LENGTH = 5

    # 2.1.3.1 Tokenize the data
    from keras.preprocessing.text import Tokenizer
    tokenizer = Tokenizer(num_words=MAX_NB_CHARS,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~',
                          lower=True,
                          char_level=True)
    tokenizer.fit_on_texts(y_orig)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X_rnn = tokenizer.texts_to_sequences(y_orig)
    
    from keras.preprocessing.sequence import pad_sequences
    X_rnn = pad_sequences(X_rnn, maxlen=MAX_SEQUENCE_LENGTH)
    X_train_rnn = X_rnn[test_ids[:-int(np.ceil(X_cnn.shape[0] * 0.25))]]
    X_test_rnn = X_rnn[test_ids[-int(np.ceil(X_cnn.shape[0] * 0.25)):]]

    # 2.1.4. prepare the labels -----------------------------------------------
    y = pd.get_dummies(y_orig).values
    # last 75/25 split
    y_train = y[test_ids[:-int(np.ceil(X_cnn.shape[0] * 0.25))]]
    y_test = y[test_ids[-int(np.ceil(X_cnn.shape[0] * 0.25)):]]
    # tags
    my_tags = sorted([i for i in set(y_orig)])

    # 2.1.5. checks -----------------------------------------------------------
    print("CNN train shape: \t", X_train_cnn.shape)
    print("RNN train shape: \t", X_train_rnn.shape)
    print("Label train shape: \t", y_train.shape)
    print("\n")
    print("CNN test shape: \t", X_test_cnn.shape)
    print("RNN test shape: \t", X_test_rnn.shape)
    print("Label test shape: \t", y_test.shape)

    # =========================================================================
    # Run models
    # =========================================================================
    print("Running models ---------------------------------------------------")
    # pointnet, pointnet_report = models.pointnet_full(y, my_tags, test_ids)
    cnn, cnn_report = models.CNN(X_train_cnn, X_test_cnn, y_train, y_test, k, my_tags=my_tags)
    # rnn = models.rnn(X_train_rnn, X_test_rnn, y_train, y_test, k, my_tags=my_tags)
    # final_model = models.standardisation_model(X_train_cnn, X_train_rnn, X_test_cnn, X_test_rnn, y_train, y_test, k, my_tags=my_tags)
    
if __name__ == '__main__':
    main()
