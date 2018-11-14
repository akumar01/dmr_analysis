from keras import backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import pdb

from scipy.stats import pearsonr


# R2 score metric to use when evaluating model - same as sklearn r2_score
def r2score(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#define the ConvNet
def buildConv2D(input_shape, npoints):
	model = Sequential()
	# CONV => RELU => POOL
	model.add(Conv2D(32, kernel_size=4, padding="same",
	input_shape=input_shape))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1), dim_ordering = 'th'))
	# CONV => RELU => POOL
	model.add(Conv2D(16, kernel_size=4, border_mode="same"))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1), dim_ordering = 'th'))
	# Flatten => RELU layers
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(Activation("relu"))

	# Dense layer for regression
	model.add(Dense(npoints))
	model.add(Activation("linear"))
	return model

# A dense feedforward network for regression
def build1DDense(input_shape, npoints):
	model = Sequential()
	model.add(Dense(input_shape))
	model.add(Activation("relu"))
#	model.add(Dropout(0.5))
	model.add(Dense(256))
	model.add(Activation("relu"))
	model.add(Dropout(0.5))
	model.add(Dense(npoints))
	model.add(Activation("linear"))
	return model


def run(xtrain, ytrain, xtest, ytest, model='Conv2D'):

	# network and training
	NB_EPOCH = 20
	BATCH_SIZE = 100
	VERBOSE = 1
	OPTIMIZER = Adam()
	VALIDATION_SPLIT=0.12
	if model == 'Conv2D':
		IMG_ROWS, IMG_COLS = xtrain.shape[1], xtrain.shape[2] # input image dimensions

	#	NB_CLASSES = 10 # number of outputs = number of digits
		INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
		# Xtrain/Xtest needs to have shape (Nexamples, INPUT_SHAPE)
		xtrain = xtrain[:, np.newaxis, :, :]
		xtest = xtest[:, np.newaxis, :, :]

		# initialize the optimizer and model
		model = buildConv2D(input_shape=INPUT_SHAPE, npoints = 1)
	
	elif model == 'Dense1D':

		model = buildDense1D(input_shape = INPUT_SHAPE, npoints = 1)

	elif model == 'Conv1D':

		model = buildConv1D(input_shape = INPUT_SHAPE, npoints = 1)

	model.compile(loss="mean_squared_error", optimizer=OPTIMIZER,
	metrics=["accuracy", r2score])

	history = model.fit(xtrain, ytrain,
	batch_size=BATCH_SIZE, epochs=NB_EPOCH,
	verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

	score = model.evaluate(xtest, ytest, verbose=VERBOSE)

	print("Test score:", score[0])
	print('Test accuracy:', score[1])

	# list all data in history
	print(history.history.keys())

	# summarize history for accuracy
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()

	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()