import numpy as np
from numpy.linalg import det
from scipy.io import loadmat, savemat
from scipy.signal import decimate
from scipy.integrate import simps
from pynwb import NWBHDF5IO
from matplotlib import pyplot as plt
from collections import namedtuple
import math, pdb, time

from utils.misc import get_data_path, count_leading_zeros, read_htk

# Align the data and stimulus using the mark track
def align(stim, resp):
	data_path = get_data_path()
	try:
		io = NWBHDF5IO('%s/R32_B8.nwb' % data_path)
		nwbfile = io.read()
		mark = nwbfile.stimulus['recorded_mark'].data[:]
	except:
		mark = loadmat('%s/mark.mat' % data_path)
		mark = mark['mark']
	# Align the DMR with the response

	# First downsample the mark to 100 Hz
	mark_ds = decimate(mark.flatten(), 4)
	start_index = np.min(np.argwhere(mark_ds > 0.1))
	end_index = start_index + stim.shape[1]

	resp = resp[start_index:end_index]

	assert resp.size == stim.shape[1]

	return stim, resp

def process_freq_bands(data, dsample, z_method = 'baseline'):
	# Cutoff the last three data points (artifact of the timeseries ECOG
	# recordings being three points longer than the stimulus)
	data = data[:-3,:]

	# Downsample the data from 400 Hz to 100 Hz
	if dsample:
		dd = downsample(data[:, 0])
		downsampled_data = np.zeros((dd.size, data.shape[1]))
		downsampled_data[:, 0] = dd	
		for i in range(1, data.shape[1]):
			downsampled_data[:, i] = downsample(data[:, i])		

		data = downsampled_data
		z_scored_data = np.zeros(downsampled_data.shape)

	else:
		z_scored_data = np.zeros(data.shape)

	for i in range(data.shape[1]):
		if z_method == 'baseline':
			z_scored_data[:, i] = Z_score(data[:, i])
		elif z_method == 'moving':
			z_scored_data[:, i] = running_Z_score(data[:, i])

	# Average together all frequency bands
	return np.mean(z_scored_data, axis = 1)

# Load data one channel at a time and process to keep size loaded into memory
# manageable
def get_gamma_from_nwb(channels, dsample, z_method, save_file, *filename):
	# 128 total electrodes
	# 29-36 are the indicies of the gamma channels
	data_path = get_data_path()
	io = NWBHDF5IO('%s/R32_B8.nwb' % data_path)
	nwbfile = io.read()	

	raw_shape = nwbfile.modules['preprocessed'].data_interfaces['Wvlt_4to1200_54band_CAR1'].electrical_series['Wvlt_ECoG128'].data.shape
	# Remember to cutoff the last three data points
	if dsample:
		data = np.zeros((math.ceil((raw_shape[0] - 3)/4), len(channels)))
	else:
		data = np.zeros((raw_shape[0] - 3, len(channels)))
	for i in range(len(channels)):
		channel = np.array(nwbfile.modules['preprocessed'].data_interfaces['Wvlt_4to1200_54band_CAR1'].electrical_series['Wvlt_ECoG128'].data[:,channels[i],29:36])
		data[:, i] = process_freq_bands(channel, dsample, z_method)		
	if save_file:
		savemat('%s/%s.mat' % (data_path, filename[0]), dict([('data', data)]))
	return data

# Handle htk files separately
def get_gamma_from_htk(path, channels, dsample, z_method, save_file, *filename):
	# 64 total electrodes
	# 29-36 are indices of the gamma channels
	data_path = get_data_path()

	data_path = get_data_path()
	data_path = '%s/%s' % (data_path, path)
	
	d1 = read_htk('%s/Wave1.htk' % data_path)[0]
	raw_shape = d1.shape
	if dsample:
		data = np.zeros((math.ceil((raw_shape[0] - 3)/4), len(channels)))
	else:
		data = np.zeros((raw_shape[0] - 3, len(channels)))
	for i in range(len(channels)):
		channel = read_htk('%s/Wave%d.htk' % (data_path, channels[i]))[0][:, 29:36]
		data[:, i] = process_freq_bands(channel, dsample, z_method)
	if save_file:
		savemat('%s/%s.mat' % (data_path, filename[0]), dict([('data', data)]))
	return data

def Z_score(data):
	# sampling rate is 400 Hz, Use a 500 ms window at the start
	mean_baseline = np.mean(data[0:200])
	std_baseline = np.std(data[0:200])
	return (data - mean_baseline)/std_baseline

# Better to calaculate a Z-score based on a running average
# since we aren't sure about the long term stability of the 
# electrode recordings. To do this, use a sliding time window
# that exceeds the autocorrelation time of the neural data
def running_Z_score(data):
	# Use a 10 second running window to Z_score the data
	window_length = 1000
	Z_scored = np.zeros(data.shape)
	means = np.zeros(data.size - window_length)
	stds = np.zeros(data.size - window_length)

	# Symmetric window
	for i in range(int(window_length/2), int(data.size - window_length/2)):
		mean_baseline = np.mean(data[i - int(window_length/2):int(i + window_length/2)])
		std_baseline = np.std(data[i - int(window_length/2):i + int(window_length/2)])
		Z_scored[i] = (data[i] - mean_baseline)/std_baseline
#		means[i - window_length/2] = mean_baseline
#		stds[i - window_length] = std_baseline
	return Z_scored

# Downsample the data to a 100 Hz sampling rate
def downsample(data):
	return decimate(data, 4)

def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

