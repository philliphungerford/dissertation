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

# cnn
import keras
from keras.optimizers import Adam
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Dropout, BatchNormalization

# rnn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

# combined model
from keras.utils import plot_model

# evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Set seed
seeds = 0


# ==============================================================================
# CNN model
# ==============================================================================
def CNN(X_train, X_test, y_train, y_test, k, my_tags):
    # Hyper parameters ---------------------------------------------------------
    max_epochs = 25
    batch_size = 128
    dropout_rate = 0.5

    # Optimizers
    # from keras.optimizers import SGD
    # opt = Adadelta(lr=0.001)
    opt = Adam(lr=0.01, decay=0.7)
    # opt = SGD(lr=0.001, momentum=0.9)

    # Model Architecture -------------------------------------------------------

    model = Sequential()
    # Convolution layers
    model.add(Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', \
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
    model.add(Dense(units=(h * w * d), activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(Dense(units=k, activation='softmax'))

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer=opt, \
                  metrics=['accuracy'])
    model.summary()

    print("\n####################### Training Model #############################")
    print("Training...")
    history = model.fit(x=X_train, y=y_train,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        validation_split=0.1,
                        verbose=1)
    model.save('cnn_model.h5')
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
    # report = classification_report(y_test, y_pred.round())
    print("\nClassfication Report for test:\n", report)
    print("\n####################################################################")

    return (model, report)


# ==============================================================================
# RNN model
# ==============================================================================
def rnn(X_train, X_test, Y_train, Y_test, k, my_tags):
    print("Building Document Classifier... \n")
    # 0. Hyperparameters -------------------------------------------------------
    # The maximum number of words to be used
    MAX_NB_WORDS = 26

    # Max number of words in each file name
    MAX_SEQUENCE_LENGTH = 5

    # This is fixed.
    EMBEDDING_DIM = 20

    epochs = 25
    batch_size = 32

    '''
    # 1. Tokenize the data -----------------------------------------------------
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, char_level=True)
    tokenizer.fit_on_texts(name_df['synthetic'].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(name_df['synthetic'].values)
    X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
    print('Shape of data tensor:', X.shape)

    Y = pd.get_dummies(name_df['tags']).values
    print('Shape of label tensor:', Y.shape)

    # 2. Split the data --------------------------------------------------------
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
    print(X_train.shape,Y_train.shape)
    print(X_test.shape,Y_test.shape)
    '''
    # 3. Build the model -------------------------------------------------------
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    # add dense layer to add to cnn
    model.add(Dense(k, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss',
                                                 patience=3, min_delta=0.0001)])

    model.save('rnn_model.h5')
    # 4. Results ---------------------------------------------------------------
    print("###################################################################")
    print("\nResults:\n")
    accr = model.evaluate(X_test, Y_test)
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

    return (model)


# ==============================================================================
# Combined model
# ==============================================================================
def standardisation_model(X_train_cnn, X_train_rnn, X_test_cnn, X_test_rnn, y_train, y_test, k, my_tags):
    # Hyperparameters ----------------------------------------------------------
    dropout_rate = 0.5
    epochs = 25
    batch_size = 128

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
    vision_model.add(Dense(units=(h * w * d), activation='relu'))
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
    embedded_filename = Embedding(input_dim=MAX_NB_CHARS, output_dim=MAX_SEQUENCE_LENGTH,
                                  input_length=MAX_SEQUENCE_LENGTH)(filename_inputs)
    embedded_filename = SpatialDropout1D(0.2)(embedded_filename)
    encoded_filename = LSTM(100)(embedded_filename)

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
    vqa_model.save('final_model.h5')
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
    return (vqa_model)


# ==============================================================================
# KNN Reference model
# ==============================================================================

def knn(X, y):
    '''
    This is a reference model for the dissertation-standardisation task
    The dimensionality for the feature set currently is (1716, 4096)
    Where each row represents an organ of which has been reshaped into a 1d V
    Performance will be compared to a 3D convolutional neural network
    '''
    print("\n####################################################################")
    print("Building KNN Reference Model")

    # Dependencies ------------------------------------------------------------
    import numpy as np
    from sklearn.metrics import classification_report
    # Load data ---------------------------------------------------------------
    print("Loading data...")
    print('X shape is: ', X.shape)
    print('y shape is: ', y.shape)

    # Split the data ----------------------------------------------------------
    print("Splitting the data...")
    from sklearn.model_selection import train_test_split
    # split data into 1: train+validation set and 2: test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seeds, test_size=0.25)

    # Build model -------------------------------------------------------------
    print("Building model...")
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    print("Training model...")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Evaluation --------------------------------------------------------------
    print("Evaluation -------------------------------------------------------")
    print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
    # Classification report ----------------------------------------------------
    report = classification_report(y_test, y_pred.round(), \
                                   target_names=my_tags)
    print("\nClassfication Report for test:\n", report)
    print("\n####################################################################")
    return (knn, report)
