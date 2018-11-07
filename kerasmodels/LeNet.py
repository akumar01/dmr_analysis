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


# R2 score metric to use when evaluating model
def r2score(y_true, y_pred):
    # SS_res =  K.sum(K.square( y_true-y_pred ))
    # SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    # return ( 1 - SS_res/(SS_tot + K.epsilon()) )
    return r2_score(y_true, y_pred)

#define the ConvNet
def build(input_shape, npoints):
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

def run(xtrain, ytrain, xtest, ytest):

	# network and training
	NB_EPOCH = 20
	BATCH_SIZE = 100
	VERBOSE = 1
	OPTIMIZER = Adam()
	VALIDATION_SPLIT=0.12

	IMG_ROWS, IMG_COLS = xtrain.shape[1], xtrain.shape[2] # input image dimensions

#	NB_CLASSES = 10 # number of outputs = number of digits
	INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS)
	# # data: shuffled and split between train and test sets
	# (X_train, y_train), (X_test, y_test) = mnist.load_data()
	# k.set_image_dim_ordering("th")

	# # consider them as float and normalize
	# X_train = X_train.astype('float32')
	# X_test = X_test.astype('float32')
	# X_train /= 255
	# X_test /= 255

	# # we need a 60K x [1 x 28 x 28] shape as input to the CONVNET
	# X_train = X_train[:, np.newaxis, :, :]
	# X_test = X_test[:, np.newaxis, :, :]
	# print(X_train.shape[0], 'train samples')
	# print(X_test.shape[0], 'test samples')

	# # convert class vectors to binary class matrices
	# y_train = np_utils.to_categorical(y_train, NB_CLASSES)
	# y_test = np_utils.to_categorical(y_test, NB_CLASSES)

	# Xtrain/Xtest needs to have shape (Nexamples, INPUT_SHAPE)
	xtrain = xtrain[:, np.newaxis, :, :]
	xtest = xtest[:, np.newaxis, :, :]

	# initialize the optimizer and model
	model = build(input_shape=INPUT_SHAPE, npoints=1)
	
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