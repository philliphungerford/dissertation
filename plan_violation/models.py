#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 8 13:15:01 2019

@author: philliphungerford

Purpose: Deep learning models for predicting plan violations in radiotherapy data
"""
# =============================================================================
# Dependencies
# =============================================================================

# Pointnet dependencies -------------------------------------------------------
import random
import numpy as np
import tensorflow as tf
from numpy.random import seed
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from keras.layers import Dense, MaxPooling1D, Convolution1D, Dropout, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
# for reading the ply files 
#from open3d import *

# Pointnet basic dependencies --------------------------------------------------
# Import dependencies

from keras.optimizers import Adam
from mlxtend.plotting import plot_confusion_matrix

# Pointnet basic with l dependencies -------------------------------------------

# 3D CNN dependencies ----------------------------------------------------------
# Install depedencies
from keras.layers import Conv3D, MaxPool3D
from keras.layers import Input
from sklearn.metrics import confusion_matrix, accuracy_score
#from mlxtend.plotting import plot_confusion_matrix
from keras.losses import categorical_crossentropy
from keras.models import Model


# fix random seed for reproducibility
seeds=42
random.seed(seeds)
seed(seeds)
set_random_seed(seeds)

def mat_mul(A, B):
	return tf.matmul(A, B)

# =============================================================================
# PointNet Full Model
# =============================================================================	
def pointnet_full():
	# ------------------------------------------------------------------------------
	# Load data
	desired_points = 1024
	#X = downsample_dataset(data_points, desired_points)
	#np.save('../2_pipeline/prostate-no-nodes-'+ str(desired_points) +'.npy', X)
	X = np.load('../2_pipeline/prostate-no-nodes-1024.npy')
	#X = np.load('../2_pipeline/no-body-4096.npy')
	y = np.load('../2_pipeline/labels.npy')

	#split data into 1: train+validation set and 2: test set 
	X_train_val, X_test, y_train_val, y_test = \
	train_test_split(X, y, random_state=0, test_size=0.2)

	# split train+validation set into 1a) training and 1b) validation sets
	X_train, X_val, y_train, y_val = \
	train_test_split(X_train_val, y_train_val, random_state=1, test_size=0.2)

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
	k = 2

	# define optimizer
	opt = optimizers.Adam(lr=0.001, decay=0.7)

	max_epochs=250
	batch_size=32
	dropout_rate = 0.7

	# Class weights
	class_weight = {0: 0.2, 1: 0.8}

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
	c = Dense(1, activation='sigmoid')(c)
	prediction = Flatten()(c)


	# print the model summary
	model = Model(inputs=input_points, outputs=prediction)
	print(model.summary())


	# compile classification model
	model.compile(optimizer='adam',
				  loss='binary_crossentropy',
				  metrics=['accuracy'])

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

	# ------------------------------------------------------------------------------
	# Fit model on training data
	for i in range(1,max_epochs+1):
		# model.fit(train_points_r, Y_train, batch_size=32, epochs=1, shuffle=True, verbose=1)
		# rotate and jitter the points
		train_points_rotate = rotate_point_cloud(train_points_r)
		train_points_jitter = jitter_point_cloud(train_points_rotate)
		history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1,\
						shuffle=True, verbose=1, validation_data=(X_val, y_val),\
						class_weight=class_weight)
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
	# Create the confusion matrix
	cm = confusion_matrix(y_true = y_test, y_pred = y_pred.round())
	print("\nOur test confusion matrix yields: ")
	print(cm)
	print("\n#####################################################################")

	#Classification report
	report = classification_report(y_test, y_pred.round())
	print("\nClassfication Report for test:\n", ann_report)
	print("\n#####################################################################")

	#Calculate AUC score
	from sklearn.metrics import roc_auc_score
	ann_auc = roc_auc_score(y_test, y_pred.round())
	print("\nOur testing AUC for ann is: ", ann_auc)

	from sklearn.metrics import roc_curve
	fpr_ann, tpr_ann, thresholds_ann = roc_curve(y_test, y_pred.round())

	# Plot AUC 
	plt.figure()
	plt.plot(fpr_ann, tpr_ann, color='purple', lw=2, label='ANN (area = {:.3f})'.format(ann_auc))
	plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
	plt.xlabel('FPR')
	plt.ylabel('TPR')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()
	print("\n#####################################################################")
return(model, cm, auc, report)

# =============================================================================
# PointNet Basic
# =============================================================================

def pointnet_basic():
	# Load data ----------------------------------------------------------------
	num_points=3140
	X = np.load('../2_pipeline/3140-xyz.npy')
	y = np.load('../2_pipeline/labels.npy')

	#split data into 1: train+validation set and 2: test set 
	X_train_val, X_test, y_train_val, y_test = \
	train_test_split(X, y, random_state=0, test_size=0.2)

	# split train+validation set into 1a) training and 1b) validation sets
	X_train, X_val, y_train, y_val = \
	train_test_split(X_train_val, y_train_val, random_state=1, test_size=0.2)

	#from keras.utils import to_categorical
	#y_test = to_categorical(y_test)
	#y_train = to_categorical(y_train)

	print('Training shape is: ', X_train.shape)
	print('Validation shape is: ', X_val.shape)
	print('Test shape is: ', X_test.shape)
	
	# hyperparameters ----------------------------------------------------------
	# number of categories
	k = 1
	# define optimizer
	opt = Adam(lr=0.001, decay=0.7)
	max_epochs=25
	batch_size=32
	dropout_rate = 0.7

	# Class weights
	class_weight = {0: 0.2, 1: 0.8}
	
	# PointNet Basic -----------------------------------------------------------
	# Point functions (MLP implemented as conv1d)
	model = Sequential()
	model.add(Convolution1D(64, 1, input_shape=(num_points, 3), activation='relu'))
	model.add(Convolution1D(64, 1, activation='relu'))
	model.add(Convolution1D(64, 1, activation='relu'))
	model.add(Convolution1D(128, 1, activation='relu'))
	model.add(Convolution1D(1024, 1, activation='relu'))

	# Symmetric function: max pooling
	model.add(MaxPooling1D(pool_size=num_points))

	#fully connected
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(rate=dropout_rate))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(rate=dropout_rate))
	model.add(Dense(1, activation='sigmoid'))
	model.add(Flatten())

	 # MLP on global point cloud vector
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	
	# Train --------------------------------------------------------------------
	# Fit the model
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=max_epochs,\
						shuffle=True, verbose=1, validation_data=(X_val, y_val),\
						class_weight=class_weight)
	# Evaluate Model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("Average loss of: ", scores[0])
	print("Average accuracy of: ", scores[1])
	
	# Performance --------------------------------------------------------------
	print("\n###################### Model Performance ############################")
	# Make predictions 
	y_pred = model.predict(X_test)

	# evaluate the model
	_, train_acc = model.evaluate(X_train, y_train, verbose=0)
	_, test_acc = model.evaluate(X_test, y_test, verbose=0)
	print('\nTrain: %.3f, Test: %.3f' % (train_acc, test_acc))
	print("\n#####################################################################")

	# Create the confusion matrix
	cm = confusion_matrix(y_true = y_test, y_pred = y_pred.round())
	print("\nOur test confusion matrix yields: ")
	fig, ax = plot_confusion_matrix(conf_mat=cm)
	plt.show()
	print("\n#####################################################################")

	#Classification report
	report = classification_report(y_test, y_pred.round())
	print("\nClassfication Report for test:\n", report)
	print("\n#####################################################################")

	# Plot AUC
	from sklearn.metrics import roc_curve, auc
	fpr, tpr, thresholds = roc_curve(y_test, y_pred)
	auc = auc(fpr, tpr)

	print("\nThe AUC is", auc)
	# Create AUC plot
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()
	# Zoom in view of the upper left corner.
	plt.figure(2)
	plt.xlim(0, 0.2)
	plt.ylim(0.8, 1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve (zoomed in at top left)')
	plt.legend(loc='best')
	plt.show()
	print("\n#####################################################################")
return(model, cm, report)

# =============================================================================
# PointNet Basic with added 'l' column 
# =============================================================================

def pointnet_basic_l():
	# Load data ----------------------------------------------------------------
	num_points=3140
	X = np.load('../2_pipeline/3140-xyzl.npy')
	y = np.load('../2_pipeline/labels.npy')

	#split data into 1: train+validation set and 2: test set 
	X_train_val, X_test, y_train_val, y_test = \
	train_test_split(X, y, random_state=0, test_size=0.2)

	# split train+validation set into 1a) training and 1b) validation sets
	X_train, X_val, y_train, y_val = \
	train_test_split(X_train_val, y_train_val, random_state=1, test_size=0.2)

	#from keras.utils import to_categorical
	#y_test = to_categorical(y_test)
	#y_train = to_categorical(y_train)

	print('Training shape is: ', X_train.shape)
	print('Validation shape is: ', X_val.shape)
	print('Test shape is: ', X_test.shape)
	# Build model --------------------------------------------------------------
	# hyperparameters
	# number of categories
	k = 1
	# define optimizer
	opt = Adam(lr=0.001, decay=0.7)
	max_epochs=25
	batch_size=32
	dropout_rate = 0.7

	# Class weights
	class_weight = {0: 0.2, 1: 0.8}
	
	################################################################################
	### POINTNET ARCHITECTURE
	################################################################################
	# Point functions (MLP implemented as conv1d)
	model = Sequential()
	model.add(Convolution1D(64, 1, input_shape=(num_points, 4), activation='relu'))
	model.add(Convolution1D(64, 1, activation='relu'))
	model.add(Convolution1D(64, 1, activation='relu'))
	model.add(Convolution1D(128, 1, activation='relu'))
	model.add(Convolution1D(1024, 1, activation='relu'))

	# Symmetric function: max pooling
	model.add(MaxPooling1D(pool_size=num_points))

	#fully connected
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(rate=dropout_rate))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(rate=dropout_rate))
	model.add(Dense(1, activation='sigmoid'))
	model.add(Flatten())

	 # MLP on global point cloud vector
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	# Train model --------------------------------------------------------------
	################################################################################
	# Fit the model
	history = model.fit(X_train, y_train, batch_size=batch_size, epochs=max_epochs,\
						shuffle=True, verbose=1, validation_data=(X_val, y_val),\
						class_weight=class_weight)

	# Evaluate Model
	scores = model.evaluate(X_test, y_test, verbose=1)
	print("Average loss of: ", scores[0])
	print("Average accuracy of: ", scores[1])
	# Evaluate model -----------------------------------------------------------
	print("\n###################### Model Performance ############################")
	# Make predictions 
	y_pred = model.predict(X_test)

	# evaluate the model
	_, train_acc = model.evaluate(X_train, y_train, verbose=0)
	_, test_acc = model.evaluate(X_test, y_test, verbose=0)
	print('\nTrain: %.3f, Test: %.3f' % (train_acc, test_acc))
	print("\n#####################################################################")

	# Create the confusion matrix
	cm = confusion_matrix(y_true = y_test, y_pred = y_pred.round())
	print("\nOur test confusion matrix yields: ")
	fig, ax = plot_confusion_matrix(conf_mat=ann_cm)
	plt.show()
	print("\n#####################################################################")

	#Classification report
	report = classification_report(y_test, y_pred.round())
	print("\nClassfication Report for test:\n", report)
	print("\n#####################################################################")

	# Plot AUC
	from sklearn.metrics import roc_curve, auc
	fpr, tpr, thresholds = roc_curve(y_test, y_pred)
	auc = auc(fpr, tpr)

	print("\nThe AUC is", auc)
	# Create AUC plot
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()
	# Zoom in view of the upper left corner.
	plt.figure(2)
	plt.xlim(0, 0.2)
	plt.ylim(0.8, 1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve (zoomed in at top left)')
	plt.legend(loc='best')
	plt.show()
	print("\n#####################################################################")
return(model, cm, report)

# =============================================================================
# 3D CNN
# =============================================================================
def cnn_3d():
	'''
	Input and Output layers:
	* One Input layer with dimentions 16, 16, 16, 3
	* Output layer with dimensions 2

	Convolutions :
	* Apply 4 Convolutional layer with increasing order of filter size (standard size : 8, 16, 32, 64) and fixed kernel size = (3, 3, 3)
	* Apply 2 Max Pooling layers, one after 2nd convolutional layer and one after fourth convolutional layer.

	MLP architecture:
	* Batch normalization on convolutiona architecture
	* Dense layers with 2 layers followed by dropout to avoid overfitting
	'''
	# Load data ----------------------------------------------------------------
	size = 16
	h,w,d = size,size,size

	if size == 16:
		X = np.load('../2_pipeline/voxeldata_16.npy')
	else:
		X = np.load('../2_pipeline/voxeldata_32.npy')
	y = np.load('../2_pipeline/labels.npy')

	#split the data
	print("Splitting the data...")
	from sklearn.model_selection import train_test_split
	#split data into 1: train+validation set and 2: test set 
	X_train_val, X_test, y_train_val, y_test = \
	train_test_split(X, y, random_state=0, test_size=0.2)

	# split train+validation set into 1a) training and 1b) validation sets
	X_train, X_val, y_train, y_val = \
	train_test_split(X_train_val, y_train_val, random_state=1, test_size=0.2)

	# Check train and test size
	print("X training size is: ", X_train.shape)
	print("y training size is: ", y_train.shape)
	print("\nX val size is: ", X_val.shape)
	print("y val size is: ", y_val.shape)
	print("\nX test size is: ", X_test.shape)
	print("y test size is: ", y_test.shape)

	x_train = X_train.reshape(X_train.shape[0], h, w, d,1)
	x_val = X_val.reshape(X_val.shape[0], h, w, d,1)
	x_test = X_test.reshape(X_test.shape[0], h, w, d,1)
	
	# Build model --------------------------------------------------------------
	# Hyper parameters
	max_epochs = 25
	batch_size = 16 # 8 < 16 > 32 > 128
	#opt = Adadelta(lr=0.001)
	opt = Adam(lr=0.001, decay=0.7)
	dropout_rate=0.2

	#from keras.optimizers import SGD
	#opt = SGD(lr=0.001, momentum=0.9)

	# Class weights
	class_weight = {0: 0.3,
					1: 0.7}

	model = Sequential()
	# Convolution layers
	model.add(Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', input_shape=(h, w, d, 1)))
	model.add(Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu'))
	# Add max pooling to obtain the most informatic features
	model.add(MaxPool3D(pool_size=(2, 2, 2)))

	# Convolution layers
	model.add(Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu'))
	model.add(Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu'))
	# Add max pooling to obtain the most informatic features
	model.add(MaxPool3D(pool_size=(2, 2, 2)))

	## perform batch normalization on the convolution outputs before feeding it to MLP architecture
	model.add(BatchNormalization())
	model.add(Flatten())

	## create an MLP architecture with dense layers : 4096 -> 512 -> 10
	## add dropouts to avoid overfitting / perform regularization
	model.add(Dense(units=(h*w*d), activation='relu'))
	model.add(Dropout(dropout_rate))
	model.add(BatchNormalization())
	model.add(Dense(units=512, activation='relu'))
	model.add(Dropout(dropout_rate))
	model.add(BatchNormalization())
	model.add(Dense(units=1, activation='sigmoid'))

	# Compile
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()
	# Train model --------------------------------------------------------------
	print("\n####################### Training Model ##############################")
	print("Training...")
	history = model.fit(x=x_train, y=y_train, batch_size=batch_size, \
						epochs=max_epochs, class_weight=class_weight, \
						validation_data=(x_val, y_val), verbose=1)
	# Evaluate model -----------------------------------------------------------
	print("\n###################### Model Performance ############################")
	# Make predictions 
	y_pred = model.predict(x_test)

	# evaluate the model
	_, train_acc = model.evaluate(x_train, y_train, verbose=0)
	_, test_acc = model.evaluate(x_test, y_test, verbose=0)
	print('\nTrain: %.3f, Test: %.3f' % (train_acc, test_acc))
	print("\n#####################################################################")

	#Classification reporta
	report = classification_report(y_test, y_pred.round())
	print("\nClassfication Report for test:\n", report)
	print("\n#####################################################################")
	# Create the confusion matrix
	cm = confusion_matrix(y_true = y_test, y_pred = y_pred.round())
	print("\nOur test confusion matrix yields: ")
	fig, ax = plot_confusion_matrix(conf_mat=cm)
	plt.show()
	print("\n#####################################################################")

	# Plot AUC
	fpr, tpr, thresholds = roc_curve(y_test, y_pred)
	auc = auc(fpr, tpr)

	print("\nThe AUC is", auc)
	# Create AUC plot
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.show()
	# Zoom in view of the upper left corner.
	plt.figure(2)
	plt.xlim(0, 0.2)
	plt.ylim(0.8, 1)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve (zoomed in at top left)')
	plt.legend(loc='best')
	plt.show()
	print("\n#####################################################################")
    return(model, cm, report)
