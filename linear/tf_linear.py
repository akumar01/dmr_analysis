import numpy as np
import tensorflow as tf
import pdb

# Convert our numpy arrays to tf Dataset
def tf_input(x, y):
	features = {
		'spct': x
	}
	dataset = tf.data.Dataset.from_tensor_slices((features, y))
	return dataset

def linear_regression(xtrain, ytrain, xtest, ytest):
	tf.logging.set_verbosity(tf.logging.INFO)
	STEPS = 1000
	spct_feature = [tf.feature_column.numeric_column(key = 'spct')]
	model = tf.estimator.LinearRegressor(feature_columns = spct_feature, optimizer='Adam')

	model.train(input_fn = lambda: tf_input(xtrain, ytrain[:, np.newaxis]), steps = STEPS)
	eval_result = model.evaluate(input_fn = tf_input(xtest, ytest[:, np.newaxis]))

	return model, eval_result




def deep_regression():
	pass