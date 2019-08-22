#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 16:27:02 2019

@author: philliphungerford

Purpose: Deep learning models for radiology data standardisation
"""
# =============================================================================
# Dependencies
# =============================================================================

import numpy as np
import pandas as pd

#cnn
from keras.optimizers import Adam
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Dropout, BatchNormalization

# rnn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping

# combined model
from keras.utils import plot_model
from keras.layers import Conv3D, MaxPooling3D, Flatten
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
import keras
import matplotlib.pyplot as plt

# pointnet
import random
import numpy as np
import tensorflow as tf
from numpy.random import seed
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from keras.layers import Dense, MaxPooling1D, Convolution1D, Dropout, Flatten, BatchNormalization, Reshape, Lambda
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping

# evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# =============================================================================
# PointNet
# =============================================================================
def mat_mul(A, B):
    return tf.matmul(A, B)

# Rotate and jitter points
def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, rotated batch of point clouds
        """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
        BxNx3 array, original batch of point clouds
        Return:
        BxNx3 array, jittered batch of point clouds
        """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

# ==========================================================================
# PointNet Full Model
# ==========================================================================
def pointnet_full(y, my_tags, test_ids, num_classes=18):
    # -------------------------------------------------------------------------
    # Load data
    desired_points = 1024
    X = np.load('../data/processed/X_pointnet.npy', allow_pickle=True)
    
    X_train = X[test_ids[:-int(np.ceil(X.shape[0]*0.25))]]
    X_test = X[test_ids[-int(np.ceil(X.shape[0]*0.25)):]]
    y_train = y[test_ids[:-int(np.ceil(y.shape[0]*0.25))]]
    y_test = y[test_ids[-int(np.ceil(X.shape[0]*0.25)):]]
    
    # Training set
    train_points_r = X_train
    train_labels_r = y_train
    
    # Test set
    test_points_r = X_test
    test_labels_r = y_test
    
    # label to categorical
    from keras.utils import to_categorical
    #y_test = to_categorical(y_test)
    #y_train = to_categorical(y_train)
    # Let's examine the data.
    
    print("Training shape: ", train_points_r.shape)
    print("Test shape: \t", test_points_r.shape)
    
    # ------------------------------------------------------------------------------
    # hyperparameter
    # number of points in each sample
    num_points = desired_points
    
    # number of categories
    k = 18
    
    # define optimizer
    opt = Adam(lr=0.001, decay=0.7)
    
    max_epochs=250
    batch_size=32
    dropout_rate = 0.7
    
    # ------------------------------------------------------------------------------
    ### POINTNET ARCHITECTURE
    
    input_points = Input(shape=(num_points, 3))
    x = Convolution1D(64, 1, activation='relu', input_shape=(num_points, 3))(input_points)
    x = BatchNormalization()(x)
    x = Convolution1D(128, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Convolution1D(1024, 1, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=num_points)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
    input_T = Reshape((3, 3))(x)
    
    # For affine transformation need to matrix multiply
    # forward net
    g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu')(g)
    g = BatchNormalization()(g)
    
    # feature transform net
    f = Convolution1D(64, 1, activation='relu')(g)
    f = BatchNormalization()(f)
    f = Convolution1D(128, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Convolution1D(1024, 1, activation='relu')(f)
    f = BatchNormalization()(f)
    f = MaxPooling1D(pool_size=num_points)(f)
    f = Dense(512, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(256, activation='relu')(f)
    f = BatchNormalization()(f)
    f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
    feature_T = Reshape((64, 64))(f)
    
    
    # forward net
    g = Lambda(mat_mul, arguments={'B': feature_T})(g)
    g = Convolution1D(64, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(128, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    g = Convolution1D(1024, 1, activation='relu')(g)
    g = BatchNormalization()(g)
    
    
    # global_feature
    global_feature = MaxPooling1D(pool_size=num_points)(g)
    
    
    # point_net_cls
    c = Dense(512, activation='relu')(global_feature)
    c = BatchNormalization()(c)
    c = Dropout(rate=dropout_rate)(c)
    c = Dense(256, activation='relu')(c)
    c = BatchNormalization()(c)
    c = Dropout(rate=dropout_rate)(c)
    c = Dense(num_classes, activation='sigmoid')(c)
    prediction = Flatten()(c)
    
    
    # print the model summary
    model = Model(inputs=input_points, outputs=prediction)
    print(model.summary())
    
    
    # compile classification model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        
    # ------------------------------------------------------------------------------
    # Fit model on training data
    for i in range(1,max_epochs+1):
      # model.fit(train_points_r, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1)
      # rotate and jitter the points
      train_points_rotate = rotate_point_cloud(train_points_r)
      train_points_jitter = jitter_point_cloud(train_points_rotate)
      history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1,\
                          shuffle=True, verbose=0, validation_split=0.1)

      s = "Current epoch is:" + str(i)
      print(s)
      if i % 5 == 0:
          score = model.evaluate(test_points_r, y_test, verbose=1)
          print('Test loss: ', score[0])
          print('Test accuracy: ', score[1])


    # ## 10. Evaluate the Model
    # score the model
    score = model.evaluate(test_points_r, y_test, verbose=1)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

    print("###################################################################")
    print("\nResults:\n")
    # ------------------------------------------------------------------------------
    # Classification Report

    # make predictions on the test set
    y_pred = model.predict(X_test)
    
    ################################################################################
    from sklearn.metrics import accuracy_score, confusion_matrix
    print("\n###################### Model Performance ############################")
    # evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('\nTrain: %.3f, Test: %.3f' % (train_acc, test_acc))
    ################################################################################
    print("\n#####################################################################")
    
    #Classification report
    report = classification_report(y_test, y_pred.round(), target_names=my_tags)
    print("\nClassfication Report for test:\n", report)
    print("\n#####################################################################")
    
    return(model, report)

# =============================================================================
# CNN
# =============================================================================
def CNN(X_train, X_test, y_train, y_test, k, my_tags):
    
    # Hyper parameters ---------------------------------------------------------
    max_epochs = 25
    batch_size = 128
    dropout_rate=0.5
    
    size = 16
    h, w, d = size, size, size
    c=1
    
    # Model Architecture -------------------------------------------------------
    model = Sequential()
    # Convolution layers
    model.add(Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu',\
                     input_shape=(h, w, d, c)))
    model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu'))
    # Add max pooling to obtain the most informative features
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    # Convolution layers
    model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu'))
    model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu'))
    # Add max pooling to obtain the most informative features
    model.add(MaxPool3D(pool_size=(2, 2, 2)))

    # perform batch normalization on the convolution outputs before
    # feeding it to MLP architecture
    model.add(BatchNormalization())
    model.add(Flatten())

    # create an MLP architecture with dense layers : 4096 -> 512 -> 10
    # add dropouts to avoid over-fitting / perform regularization
    model.add(Dense(units=(h*w*d), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(units=k, activation='softmax'))

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer='adam', \
               metrics=['accuracy'])
    model.summary()

    print("\n####################### Training Model #############################")
    print("Training...")
    history = model.fit(x=X_train, y=y_train,
                     batch_size=batch_size,
                     epochs=max_epochs,
                     validation_split=0.1,
                     verbose=1)

    # ==========================================================================
    # 3. Results
    # ==========================================================================
    print("###################################################################")
    print("\nResults:\n")
    accr = model.evaluate(X_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],
                                                               accr[1]))
    print("-------------------------------------------------------------------")
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show();

    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show();
    print("###################################################################")

    # Make predictions
    y_pred = model.predict(X_test)

    # evaluate the model
    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print('\nTrain: %.3f, Test: %.3f' % (train_acc, test_acc))
    print("\n####################################################################")

    # Classification report
    report = classification_report(y_test, y_pred.round(), target_names=my_tags)
    #report = classification_report(y_test, y_pred.round())
    print("\nClassfication Report for test:\n", report)
    print("\n####################################################################")
                     
    return(model, report)

# =============================================================================
# RNN
# =============================================================================
def rnn(X_train, X_test, y_train, y_test, k, my_tags):
    print("Building Document Classifier... \n")
    # 0. Hyperparameters -------------------------------------------------------
    # The maximum number of words to be used
    MAX_NB_WORDS = 26
    
    # Max number of words in each file name
    MAX_SEQUENCE_LENGTH = 5
    
    # This is fixed.
    EMBEDDING_DIM = 20
    
    epochs = 250
    batch_size = 128
    
    # 3. Build the model -------------------------------------------------------
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
    # add dense layer to add to cnn
    model.add(Dense(k, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                      validation_split=0.1,
                      callbacks=[EarlyStopping(monitor='val_loss',
                                               patience=3, min_delta=0.0001)])

    # 4. Results ---------------------------------------------------------------
    print("###################################################################")
    print("\nResults:\n")
    accr = model.evaluate(X_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],
                                                                accr[1]))
    print("-------------------------------------------------------------------")
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show();

    plt.title('Accuracy')
    plt.plot(history.history['acc'], label='train')
    plt.plot(history.history['val_acc'], label='test')
    plt.legend()
    plt.show();

    # Make predictions
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred.round(), target_names=my_tags)
    print("\nClassfication Report for test:\n", report)
    print("###################################################################")

    return(model)
# =============================================================================
# Our model (combined)
# =============================================================================
from keras.layers import Conv3D, MaxPooling3D, Flatten
from keras.layers import Input, GRU, Embedding, Dense
from keras.models import Model, Sequential
import keras
import matplotlib.pyplot as plt

def standardisation_model(X_train_cnn, X_train_rnn, X_test_cnn, X_test_rnn, y_train, y_test, k, my_tags):
    
    # Hyperparameters ----------------------------------------------------------
    dropout_rate=0.5
    epochs=250
    batch_size=128
    size = 16
    h, w, d = size, size, size
    c=1
    MAX_SEQUENCE_LENGTH = 5
    MAX_NB_CHARS = 26
    # First, let's define a vision model using a Sequential model.
    # This model will encode an image into a vector.
    vision_model = Sequential()
    vision_model.add(Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', input_shape=(h, w, d, c)))
    vision_model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu'))
    vision_model.add(MaxPool3D(pool_size=(2, 2, 2)))
    vision_model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu'))
    vision_model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu'))
    vision_model.add(MaxPool3D(pool_size=(2, 2, 2)))
    vision_model.add(BatchNormalization())
    vision_model.add(Flatten())
    vision_model.add(Dense(units=(h*w*d), activation='relu'))
    vision_model.add(Dropout(dropout_rate))
    vision_model.add(BatchNormalization())
    vision_model.add(Dense(units=512, activation='relu'))
    vision_model.add(Dropout(dropout_rate))
    vision_model.add(BatchNormalization())
    
    # Now let's get a tensor with the output of our vision model:
    cnn_inputs = Input(shape=(h, w, d, c))
    encoded_image = vision_model(cnn_inputs)
    
    # Next, let's define a language model to encode the filename into a vector.
    # Each filename will be at most 20 characters long,
    # and we will index words as integers from 1 to 99.
    filename_inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_filename = Embedding(input_dim=MAX_NB_CHARS, output_dim=MAX_SEQUENCE_LENGTH, input_length=MAX_SEQUENCE_LENGTH)(filename_inputs)
    embedded_filename = SpatialDropout1D(0.2)(embedded_filename)
    encoded_filename = GRU(100)(embedded_filename)
    
    # Let's concatenate the filename vector and the image vector:
    merged = keras.layers.concatenate([encoded_filename, encoded_image])
    
    # And let's train a logistic regression over 100 words on top:
    output = Dense(k, activation='softmax')(merged)
    
    # This is our final model:
    vqa_model = Model(inputs=[cnn_inputs, filename_inputs], outputs=output)
    
    vqa_model.compile(loss='categorical_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
    vqa_model.summary()
                      
    history = vqa_model.fit([X_train_cnn, X_train_rnn], y_train,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_split=0.1,
                          callbacks=[EarlyStopping(monitor='val_loss',
                                                   patience=3,
                                                   min_delta=0.0001)])
    plot_model(vqa_model, to_file='model.png')

    print("###################################################################")
    print("\nResults:\n")
    accr = vqa_model.evaluate([X_test_cnn, X_test_rnn], y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],
                                                                accr[1]))
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Make predictions
    y_pred = vqa_model.predict([X_test_cnn, X_test_rnn])
    report = classification_report(y_test, y_pred.round(), target_names=my_tags)
    print("\nClassfication Report for test:\n", report)
    print("###################################################################")
    return(vqa_model)
