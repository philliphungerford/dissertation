#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:27:02 2019

@author: philliphungerford

Purpose: Machine learning models for radiology data standardisation
"""
# =============================================================================
# Run models
# =============================================================================
if __name__ == '__main__':
    # 2.1. load data --------------------------------------------------------------
    filenames_y = pd.read_csv('data/processed/dataset2labelsorgansclean.csv')
    X_cnn = np.load('data/processed/dataset2voxels16.npy')
    X_rnn = filenames_y['organs']
    # y_orig = filenames_y['class7']
    # k=8
    y_orig = filenames_y['class13']  # actually 18 different organs
    k = 18

    # 2.1.2. prepare cnn data -----------------------------------------------------
    size = 16
    h, w, d = size, size, size
    c = 1  # Channels 1 = grey scale, 3 = colour

    # taking random indices to split the dataset into train and test
    test_ids = np.random.permutation(X_cnn.shape[0])

    # splitting data and labels into train and test
    # keeping last 10 entries for testing, rest for training

    X_train_cnn = X_cnn[test_ids[:-int(np.ceil(X_cnn.shape[0] * 0.25))]]
    X_test_cnn = X_cnn[test_ids[-int(np.ceil(X_cnn.shape[0] * 0.25)):]]

    X_train_cnn = X_train_cnn.reshape(X_train_cnn.shape[0], h, w, d, c)
    X_test_cnn = X_test_cnn.reshape(X_test_cnn.shape[0], h, w, d, c)

    # 2.1.3. prepare rnn data -----------------------------------------------------

    MAX_NB_CHARS = 26
    MAX_SEQUENCE_LENGTH = 5
    EMBEDDING_DIM = 20

    # 2.1.3.1 Tokenize the data
    tokenizer = Tokenizer(num_words=MAX_NB_CHARS,
                          filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True,
                          char_level=True)
    tokenizer.fit_on_texts(y_orig)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X_rnn = tokenizer.texts_to_sequences(y_orig)
    X_rnn = pad_sequences(X_rnn, maxlen=MAX_SEQUENCE_LENGTH)
    X_train_rnn = X_rnn[test_ids[:-int(np.ceil(X_cnn.shape[0] * 0.25))]]
    X_test_rnn = X_rnn[test_ids[-int(np.ceil(X_cnn.shape[0] * 0.25)):]]

    # 2.1.4. prepare the labels ---------------------------------------------------
    y = pd.get_dummies(y_orig).values
    # last 75/25 split
    y_train = y[test_ids[:-int(np.ceil(X_cnn.shape[0] * 0.25))]]
    y_test = y[test_ids[-int(np.ceil(X_cnn.shape[0] * 0.25)):]]
    # tags
    my_tags = sorted([i for i in set(y_orig)])

    # 2.1.5. checks -------------------------------------------------------------
    print("CNN train shape: \t", X_train_cnn.shape)
    print("RNN train shape: \t", X_train_rnn.shape)
    print("Label train shape: \t", y_train.shape)
    print("\n")
    print("CNN test shape: \t", X_test_cnn.shape)
    print("RNN test shape: \t", X_test_rnn.shape)
    print("Label test shape: \t", y_test.shape)

    # =============================================================================
    # Run models
    # =============================================================================
    cnn, cnn_report = CNN(X_train_cnn, X_test_cnn, y_train, y_test, k, my_tags=my_tags)
    rnn = rnn(X_train_rnn, X_test_rnn, y_train, y_test, k, my_tags=my_tags)
    final_model = standardisation_model(X_train_cnn, X_train_rnn, X_test_cnn, X_test_rnn, y_train, y_test, k,
                                        my_tags=my_tags)
