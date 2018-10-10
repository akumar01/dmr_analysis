import numpy as np
from numpy.linalg import det
import math
from scipy.io import loadmat, savemat
from scipy.signal import decimate
from scipy.integrate import simps
from pynwb import NWBHDF5IO
from matplotlib import pyplot as plt
from collections import namedtuple
import pdb
import time	

from utils.preprocess import get_data_path
from utils.misc import count_leading_zeros
import sklearn
from sklearn.metrics import r2_score

# Remove certain frequency channels from the stimulus (e.g. DC component)
# stim has dimensions (nsamples, # freq bins, delay time)
# channels: list
def remove_channels(stim, ncutoff):
	mean_subtracted_stim = np.zeros((stim.shape[0], stim.shape[1] - ncutoff, stim.shape[2]))
	for i in range(stim.shape[0]):
		mean_subtracted_stim = stim[:, ncutoff:, :]
	return mean_subtracted_stim

# Takes a (n freq bins)*delay_time x 1 derived response kernel, reshapes it, and applies it to
# the spectrogram samples x
# x has dimensions (n_samples, # freq bins, delay time)
def predict_response(h, x):
	x = flatten_spct(x)
	return x @ h

def evaluate_strf(xtest, ytest, h):
	y_pred = predict_response(h, xtest)
	return r2_score(ytest, y_pred)

# Flatten spectrogram for linear regression
# This takes in the output of split_data
def flatten_spct(spct):
	reshaped_spct = spct.copy()
	return np.reshape(reshaped_spct, (spct.shape[0], spct.shape[1] * spct.shape[2]))

# Take the logarithm of the spectrogram
# Input dimensions are (n samples, # freq bins, delay time)
# Reshape outputs the flattened version. Turn this off to
# return a 2D image
def compress_spct(spct, reshape = 1):

	raw_dims = spct.shape
	spct = flatten_spct(spct)
	for i in range(spct.shape[0]):
		spct[i, :] = np.log(spct[i, :])
	if reshape:
		return spct
	else:
		return np.reshape(spct, raw_dims)

# Bin response data according to binsize
# Uses a simple moving average filter
# Return binned indices for use in matching up binned 
# data points with the stimulus
def bin_data(binsize, data):
	if binsize == 1:
		return binned_data, np.arange(binned_data.size)

	if binsize == 2:
		binned_data = np.zeros(math.floor(data.size/2))
		binned_indices = np.zeros(binned_data.size)
		for i in range(binned_data.size):
			binned_data[i] = np.mean(data[2*i:min(2*i + 2, data.size)])
			binned_indices[i] = 2 * i
	else:
		# Implement this when needed
		pass
	return binned_data, binned_indices

# Split the data between training, validation, and test sets randomly
# If val = False, then we do not choose a validation split, presumably because 
# we are using cross-validation, or are using built in modules that automatically
# select validation sets from the training data
# Bin size - how much to bin the response by
# Delay time give the length of stimulus preceeding the response to associate with the particular
# data point, in units of samples
def split_data(stim, resp, binsize, train_split = 0.8, delay_time=50, val=False, val_split = 0.1):

	# Cutoff the first few and last few seconds of the stimulus for two reasons:
	# (i) If using a moving z_score, there may be leading and trailing zeros
	# for a period of time half the window size used for z-scoring
	# (ii) Physically, neurons will have unpredictable responses to sudden onset
	# and termination of stimulus
	lead_padding_length = count_leading_zeros(resp)
	trail_padding_length = count_leading_zeros(np.flip(resp, 0))
	# Assumes a 100 Hz sampling rate
	start_ind = max(500, lead_padding_length)
	end_ind = max(500, trail_padding_length)

	resp = resp[start_ind:-end_ind]
	stim = stim[:, start_ind:-end_ind]

	# Need to cutoff the first part of the respsonse so we don't draw points for which stim[t - delay_time]
	# doesn't exist

	resp = resp[delay_time:]
	stim_org = stim.copy()
	resp_org = resp.copy()

	if not val:
		val_split = 0
	test_split = 1 - train_split - val_split

	# Bin the data
	resp, binned_indices = bin_data(binsize, resp)

	train_inds = np.random.choice(np.arange(resp.size), math.floor(train_split * resp.size), replace = False)
	# Remove the train_inds from the list
	rem_inds = np.delete(np.arange(resp.size), train_inds)
	if val:
		val_inds = np.random.choice(rem_inds, math.floor(val_split * resp.size), replace = False)
		# Test is whatever is leftover
		test_inds = np.setdiff1d(rem_inds, val_inds)
	else:
		test_inds = rem_inds

	train_resp = resp[train_inds]
	if val:
		val_resp = resp[val_inds]
	else:
		val_resp = []
	test_resp = resp[test_inds]

	# Select stimulus preceeding the particular response data point
	train_stim = np.zeros((train_resp.size, stim.shape[0], delay_time))

	for i in range(train_resp.size):
		try:
			train_stim[i, :, :] = stim[:, int(binned_indices[train_inds[i]]):int(binned_indices[train_inds[i]]) + delay_time]
		except:
			pdb.set_trace()
	if val:
		val_stim = np.zeros(val_resp.size, stim.shape[0], delay_time)

		for i in range(val_resp.size):
			val_stim[i, :, :] = stim[:, int(binned_indices[val_inds[i]]):int(binned_indices[val_inds[i]] + delay_time)]
	else:
		val_stim = []

	test_stim = np.zeros((test_resp.size, stim.shape[0], delay_time))

	for i in range(test_resp.size):
		test_stim[i, :, :] = stim[:, int(binned_indices[test_inds[i]]):int(binned_indices[test_inds[i]] + delay_time)]
	Pack = namedtuple('Pack', ['train_resp', 'val_resp', 'test_resp', 'train_stim', 'val_stim', 'test_stim'], verbose = False)
	return Pack(train_resp, val_resp, test_resp, train_stim, val_stim, test_stim)
