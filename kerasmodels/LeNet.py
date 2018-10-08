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
import numpy as np
import matplotlib.pyplot as plt
import pdb

from FormatData import format_data

#define the ConvNet
class LeNet:
	@staticmethod
	def build(input_shape, npoints):
		model = Sequential()
		# CONV => RELU => POOL
		model.add(Conv2D(20, kernel_size=3, padding="same",
		input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1), dim_ordering = 'th'))
		# CONV => RELU => POOL
		model.add(Conv2D(50, kernel_size=3, border_mode="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 1), dim_ordering = 'th'))
		# Flatten => RELU layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# Dense layer for regression
		model.add(Dense(npoints))
		model.add(Activation("linear"))
		return model

if __name__ == "__main__":
	# network and training
	NB_EPOCH = 20
	BATCH_SIZE = 100
	VERBOSE = 1
	OPTIMIZER = Adam()
	VALIDATION_SPLIT=0.12
	IMG_ROWS, IMG_COLS = 100, 30 # input image dimensions
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

	training_set, test_set = format_data()

	training_x = training_set[0][:, np.newaxis, :, :]

	test_x = test_set[0][:, np.newaxis, :, :]

	# initialize the optimizer and model
	model = LeNet.build(input_shape=INPUT_SHAPE, npoints=10)
	
	model.compile(loss="mean_squared_error", optimizer=OPTIMIZER,
	metrics=["accuracy"])

	history = model.fit(training_x, training_set[1],
	batch_size=BATCH_SIZE, epochs=NB_EPOCH,
	verbose=VERBOSE, validation_split=VALIDATION_SPLIT)

	score = model.evaluate(test_x, test_set[1], verbose=VERBOSE)

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