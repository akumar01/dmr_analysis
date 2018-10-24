import os, pdb, time, math
import numpy as np
from numpy.linalg import det
from scipy.io import loadmat, savemat
from scipy.signal import decimate
from scipy.integrate import simps
from pynwb import NWBHDF5IO
from matplotlib import pyplot as plt
from htkfile import HTKFile

# Return the folder in which store data
def get_data_path():
	return os.path.abspath('../data')
def get_fig_path():
	return os.path.abspath('../figures')

# Check if the array contains any NaN values
def check_nan(x):
	return np.isnan(np.sum(x))

# Count consecutives zeros at the beginning of an arra
def count_leading_zeros(vals):
	count = 0
	for v in vals:
		if v == 0:
			count += 1
		else:
			return count
	return count

# Read the .htk file format
def read_htk(path):
    file = HTKFile(path)
    data = file.read_data()
    return data, file.sample_rate

# Calculate the mean and standard deviation of the autocorrelation width and the inflection
# point delay across all electrodes
def analysis_across_electrodes():
	aw_means = np.zeros(128)
	aw_stds = np.zeros(128)

	sdw_means = np.zeros(128)
	sdw_stds = np.zeros(128)

	# Iterate over all electrodes
	for i in range(128):
		data = get_gamma_from_grid([i], 'baseline', False)
		aw = autocorr_width(data, 'sliding')
		# Cutoff portion of this data at start and end that is just 0
		aw[np.nonzero(aw)]

		aw_means[i] = np.mean(aw)
		aw_stds[i] = np.std(aw)

		sdw = second_derivative_width(data, 'sliding')
		sdw[np.nonzero(sdw)]

		sdw_means[i] = np.mean(sdw)
		sdw_stds[i] = np.std(sdw)
		print(i)

	return {'aw_means': aw_means, 'aw_stds': aw_stds, 'sdw_means': sdw_means, 'sdw_stds': sdw_stds}

# What is the timescale of the autocorrelation peak and how does it vary in time?
def autocorr_width(data, window_length, slide_index, average_window, threshold, mode = 'fixed'):

	# Effect of downsampling:
#	data = downsample(data)	

	# Shift by some random amount
#	shift_index = np.random.randint(0, 1000)
#	data = data[shift_index:]

	# How long of a subset to take the autocorrelation over 
#	window_length = 200
	acorr_diag = 0
	if mode == 'fixed':

		# Truncate the data so it is a multiple of window length
		rem = data.size % window_length
		data = data[:-rem]	

		# Partition data into equal subsets of window_length
		partitioned_data = np.split(data, int(data.size/window_length))

		autocorr_time = np.zeros(len(partitioned_data))
		inflection_time = np.zeros(len(partitioned_data))

		for i in range(len(partitioned_data)):
			subset = partitioned_data[i]
			# Remember to substract mean when calculating autocorrelation
			autocorr = np.correlate(subset - np.mean(subset), subset - np.mean(subset), mode = 'same')
			# Normalize and take only the second half (symmetric)
			autocorr = autocorr[math.floor(autocorr.size/2):]/np.max(autocorr)
			# Find the first zero crossing in the autocorrelation
			acorrsign = np.sign(autocorr)
			signchange = ((np.roll(acorrsign, 1) - acorrsign) != 0).astype(int)
			signchange[0] = 0
			autocorr_time[i] = np.argmax(signchange)


	# Instead of using non-overlapping windows, slide the window over the data
	# one point at a time

	elif mode == 'sliding':

		# How much to move over by for each autocorrelation calculation
#		slide_index = 200

#		threshold = 0.005

		autocorr_time = np.zeros(math.ceil(len(data)/slide_index))
		inflection_time = np.zeros(math.ceil(len(data)/slide_index))
		np.seterr(all='raise')
		for i in range(int(window_length/2), len(data) - int(window_length/2), slide_index):
			index = int((i - window_length/2)/slide_index)

			dat = data[i - int(window_length/2):min(i + int(window_length/2), len(data))] 
			dat = dat - np.mean(dat)

			try:
				autocorr = np.correlate(dat, dat, mode ='same')
			except:
				autocorr = np.correlate(dat.flatten(), dat.flatten(), mode ='same')

			if index == 150:
				acorr_diag = autocorr

			try:
				# 
				sd = np.diff(autocorr/np.max(autocorr), 2)

				sd_peak_loc = np.argmin(sd)

				sd = sd[sd_peak_loc:]

				sdsign = np.sign(sd)
				# sd_temp = np.diff(autocorr/np.max(autocorr), 2)
				# autocorr = autocorr[math.floor(autocorr.size/2):]/np.max(autocorr)
				# sd = np.diff(autocorr, 2)
				sd_signchange = ((np.roll(sdsign, 1) - sdsign) != 0).astype(int)
				sd_signchange[0] = 0
				inflection_time[index] = np.argmax(sd_signchange)

				autocorr = autocorr[math.floor(autocorr.size/2):]/np.max(autocorr)
				acorrsign = np.sign(autocorr)
				signchange = ((np.roll(acorrsign, 1) - acorrsign) != 0).astype(int)
				signchange[0] = 0
#				autocorr_time[index] = np.argmax(signchange)

				# Instead of just finding the first zero time, find the point at which the moving average
				# has fallen below a certain threshold

#				average_window = 50
				averages = moving_average(autocorr, average_window)
				check = np.argwhere(averages < threshold)

				if check.size:
#					pass
					autocorr_time[index] = np.min(check) + math.floor(average_window/2)
				else:
					print('Warning! Either expand window size or increase threshold')

			except Exception as e:
				autocorr_time[index] = 0
				inflection_time[index] = 0
			# Find the first zero crossing in the autocorrelation

	return autocorr_time, inflection_time, acorr_diag

# Create plots for all electrodes
def gen_plots():


	aw_means = np.zeros(128)
	aw_stds = np.zeros(128)

	sdw_means = np.zeros(128)
	sdw_stds = np.zeros(128)

	aw_max = np.zeros(128)
	aw_min = np.zeros(128)

	sdw_max = np.zeros(128)
	sdw_min = np.zeros(128)

	x = loadmat('full_grid_baseline_zscored.mat')

	# Generate autocorrelation plots
	for i in range(128):
		start_time = time.time()

		acorrwidth, sd_width = autocorr_width(x['data'][:,i], 3000, 300, 25, 0.005, mode = 'sliding')		

		# Only include non-zero elements
		acorrwidth = acorrwidth[np.nonzero(acorrwidth)]
		sd_width = sd_width[np.nonzero(sd_width)]
		
		# Cutoff the first 2 seconds and last 2 seconds
		acorrwidth = acorrwidth[10:-10]
		sd_width = sd_width[10:-10]

		aw_max[i] = np.max(acorrwidth)
		aw_min[i] = np.min(acorrwidth)


		sdw_max[i] = np.max(sd_width)
		sdw_min[i] = np.min(sd_width)

		aw_means[i] = np.mean(acorrwidth)
		aw_stds[i] = np.std(acorrwidth)

		sdw_means[i] = np.mean(sd_width)
		sdw_stds[i] = np.std(sd_width)

		# Plot autocorrelation width
		fig = plt.figure(figsize = (15, 10))
		a = fig.add_subplot(111)
		plt.plot(acorrwidth)
		plt.title('Autocorrelation Time to Zero, Electrode %d, 30 second windows' % i)
		plt.ylabel('Delay Time to a moving average of < 0.005 of peak')
		plt.xlabel('Time (s)')
		ticks = a.get_xticks().tolist()
		ticks[:] = [t * 2 for t in ticks]
		a.set_xticklabels(ticks)
		yticks = a.get_yticks().tolist()
		ticks[:] = [t/100 for t in ticks]
		a.set_yticklabels(ticks)
		plt.savefig('Figures/092418/batch/ac/%d.png' % i)
		plt.close()
#		Plot inflection point times
		# fig = plt.figure(figsize = (15, 10))
		# a = fig.add_subplot(111)
		# plt.plot(sd_width)
		# plt.title('Autocorrelation Inflection Point, Electrode %d, 2 second sliding windows' % i)
		# plt.ylabel('Inflection point delay time (ms)')
		# plt.xlabel('Time (s)')
		# ticks = a.get_xticks().tolist()
		# ticks[:] = [t/100 for t in ticks]
		# a.set_xticklabels(ticks)
		# yticks = a.get_yticks().tolist()
		# yticks[:] = [t * 10 for t in yticks]
		# a.set_yticklabels(yticks)
		# plt.savefig('Figures/092418/batch/acsd/%d.png' % i)
		# plt.close()

		# Record mean and standard deviation
		acorrwidth = acorrwidth[np.nonzero(acorrwidth)]
		aw_means[i] = np.mean(acorrwidth)
		aw_stds[i] = np.std(acorrwidth)

		sdw_means[i] = np.mean(sd_width)
		sdw_stds[i] = np.std(sd_width)

		print("---%s seconds---" % (time.time() - start_time))

	return {'aw_means': aw_means, 'aw_stds': aw_stds, 'sdw_means':sdw_means, 'sdw_stds':sdw_stds,
			 'aw_max':aw_max, 'aw_min': aw_min, 'sdw_max': sdw_max, 'sdw_min': sdw_min}

# Use the results of https://asa.scitation.org/doi/pdf/10.1121/1.1913074 to try and find the optimal
# window over which to calculate the autocorrelation
def determine_optimal_window_length(data):

	# Trial window lengths 
	T = [20, 40, 60, 80]

	# Cutoff the first and last 10 seconds of data
	# Randomly shift the start index, can do this over many realizations

	data = data[1000:-1000]

	shift_index = np.random.randint(0, 1000)
	data = data[shift_index:]

	# Total errors accrued over the time series for a given window size
	QT = np.zeros(len(T))
	T_index = 0
	# Partition dataset into fixed window sizes 
	for t in T:
		start_time = time.time()
		data_copy = data.copy()
		# Truncate the data so it is a multiple of window length
		rem = data.size % t
		if rem != 0:
			data_copy = data_copy[:-rem]	

		# Partition data into equal subsets of window_length
#		partitioned_data = np.split(data, int(data.size/t))
	
		# Further subdivide T into 3 sub-windows. Get as close as one can
		subdivided_partition = np.array_split(np.arange(t), 3)

		subdiv_lengths = [len(sp) for sp in subdivided_partition]
		t1 = [0]
		t1 = np.append(t1, np.cumsum(subdiv_lengths[0:2]))
		tcenter = [math.floor(len(sp)/2) for sp in subdivided_partition]

		tj = tcenter + t1

		# Errors
		try:
			EK = np.zeros(int(data.size/t - 2))
		except Exception as e:
			pdb.set_trace()

		# Need to further throw out the first and last windows for the algorithm to
		# work
		for i in range(1, int(data.size/t) - 1):

			t_ref = i * t

			corr = np.zeros((t + 1, 3))

			# Number of Fourier coefficients a_i to extract for each t_j
			n = t
			a = np.zeros((3, n))


			for j in range(3):
				data_slice = data_copy[int(t_ref + tj[j] - t/2):int(t_ref + tj[j] + t/2 + 1)]
				# Data slice is centered on t0
				t0 = math.floor(data_slice.size/2)
				corr[:, j] = symmetric_correlation(data_slice - np.mean(data_slice), t0, t)
				# Find the Fourier coefficients of the calculated autocorrelation
				corr_temp = corr[:, j].copy()
				a[j, :] = manual_FT(corr_temp, n , t)
			# Solve for the f_i
			f_i = np.zeros(n)
			Delta = det(np.array([[1, 1, 1], tj, tj * tj]).T)				

			for j in range(n):
				f_i[j] = 1/Delta * det(np.array([[1, 1, 1], tj, tj *tj]).T)

			# Calculate T_K
			# Use the autocorrelation centered in the window
			fi, fk = np.meshgrid(f_i, f_i)
			f_sum = np.sum(fi * fk)
			T_K = np.power(144 * (corr[:, 2] @ corr[:,2])/f_sum, 0.2)

			# Calculate the error for this particular window size
			EK[i - 1] = 1 - T_K/float(t)

		print("---%s seconds---" % (time.time() - start_time))
		QT[T_index] = np.sum(EK) * 1/i 
		T_index += 1

	return QT

# Implement the autocorrelation in the same way as that defined in the paper above
def symmetric_correlation(x, t0, T):
	# Pad the shit out x
	x = np.pad(x, (T, T), 'constant', constant_values=(0, 0))
	t0 += T
	delays = np.linspace(0, T, int(T + 1))
	time = np.linspace(-T/2, T/2, T + 1)
	corr = np.zeros(delays.size)
	for i in range(delays.size):
		for j in range(T + 1):
			try:
				corr[i] += x[int(t0 + time[j] + delays[i]/2)] * x[int(t0 + time[j] - delays[i]/2)]
			except:
				pdb.set_trace()
	# Normalize
	return 1/T * corr

# Want to obtain coefficients in the expansion in eq. 15 of the paper above. Due to the slightly unusual
# basis functions, want to do this manually to make sure it is done right
# corr: autocorrelation, n: number of Fourier components to extract
def manual_FT(cor, n, T):
	a_i = np.zeros(n)
	for i in range(1, n + 1):
		tau = np.linspace(0, T, cor.size)
		basis_fn = np.cos((i - 1) * 2 * np.pi * tau/T)
		a_i[i - 1] = simps(cor * basis_fn)
	return a_i

