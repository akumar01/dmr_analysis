import time, datetime
import numpy as np
from numpy.linalg import pinv, svd
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from utils.process import split_data, flatten_spct, remove_channels
from utils.preprocess import align
from utils.misc import get_fig_path, check_nan
from scipy.stats import pearsonr
import pdb, math
import os

# Do vanilla ridge regression on a single electrode for average high gamma channel
def ridge_regression(xtrain, ytrain, xtest, ytest, alphas=[1, 3, 5, 10]):
	# Always check for NaN before running
	assert not check_nan(xtrain)
	assert not check_nan(ytrain)
	# Always check for NaN before running
	assert not check_nan(xtest)
	assert not check_nan(ytest)

	models = []
	r2_scores = np.zeros(len(alphas))
	for i, alpha in enumerate(alphas):
		start_time = time.time()
		r = Ridge(alpha, fit_intercept = True, normalize = True)
		r.fit(xtrain, ytrain)
		ypred = r.predict(xtest)
		r2_scores[i] = r2_score(ytest, ypred)
		models.append(r)
		print("---%s seconds---" % (time.time() - start_time))

	return r2_scores, models


# Do a ridge regression fit across all electrodes
def batch_ridge(stim, resp, alpha):
	rscores = []
	models = []
	for i in range(resp.shape[1]):
		try:
			s, r = align(stim['spct'], resp[:, i])
		except Exception as e:
			pdb.set_trace()

		p = split_data(s, r, 2, train_split = 0.8)
		xtrain = norm_spct(remove_channels(p.train_stim, [3, 3]))
		xtest = norm_spct(remove_channels(p.test_stim, [3, 3]))
		ytrain = p.train_resp
		ytest = p.test_resp
		model, rscore = ridge_regression(xtrain, ytrain, xtest, ytest, alphas = [5])

		models.append(model)
		rscores.append(rscore)

	return rscores, models

# Plot a bunch of separate 
def batch_STRF_plot(models):
	fig_path = get_fig_path()
	for i in range(len(models)):
		plot_STRF(models[i][0], 50, 'Electrode %d' % i, '%s/102418/STRFs' % fig_path, '%d' % i)


# Boosting as described in "Estimating sparse spectro-temporal receptive fields with
# natural stimuli"
def boosting(x, y, xtest, ytest, n_iterations):
	# Always check for NaN before running
	assert not check_nan(x)
	assert not check_nan(y)
	# Always check for NaN before running
	assert not check_nan(xtest)
	assert not check_nan(ytest)

	# Reserve a fraction of the training data to determine an early stopping point
	xtrain, xval, ytrain, yval = train_test_split(x, y, test_size = 0.05)
	# step size is calculated according to the heuristic 1/50 * sqrt(var(r(t))/ var(s(x, t)))
	step_size = 1/50 * np.sqrt(np.var(ytrain)/np.var(xtrain))
	# Inintially set the STRF to 0
	h = np.zeros(xtrain.shape[1])
	# Baseline mean square error
	pred_resp = xtrain @ h
	baseline_mse = calc_mse(pred_resp, ytrain)
	# Test all possible perturbations to the strf

	# Baseline model performance on validation set
	val_pred = xval @ h
	running_rscore = 0
	for j in range(n_iterations):	
		start_time = time.time()
		mses = np.zeros((h.size, 2))
		# randomly sample 10 % of h to change
#		rand_indices = np.random.choice(np.arange(h.size), math.floor(h.size/2))
		for k, eta in enumerate([-1, 1]):
			for l in range(h.size):
				htest = h.copy()
				htest[l] += step_size * eta
				pred_resp = xtrain @ htest
				# Calculate the  mean squared error of this new estimate
				mses[l, k] = calc_mse(pred_resp, ytrain)
		# Accept the perturbation that reduces the mean square error the greatest
		min1 = np.argmin(mses, axis = 0)
		min2 = np.argmin(np.amin(mses, axis = 0))
		h[min1[min2]] += step_size * np.array([-1, 1])[min2]
		# Test the new model against thes validation set. If it no longer improves 
		# prediction accuracy, stop early
		val_pred = xval @ h
		rscore = np.power(pearsonr(val_pred, yval)[0], 2)
		if rscore < running_rscore:
			running_rscore = rscore
			#break
		else:
			running_rscore = rscore
		print("---%s seconds---" % (time.time() - start_time))
		print(running_rscore)
	pdb.set_trace()
	# Evaluate the model on the test data
	test_pred = xtest @ h
	rscore = np.power(pearsonr(test_pred, ytest)[0], 2)
	return h, rscore

def calc_mse(x, y):
	x = x.flatten()
	y = y.flatten()
	return 1/x.size * (x - y) @ (x - y)

# Vary over time delays and regularization strengths
def search_ridge_params(stim, resp):
	# Assuming 100 Hz sampling rate
	delays = np.array([50])
	alphas = np.array([1, 3, 5, 10, 15, 20, 25])
	r2scores = np.zeros(len(delays))
	mean_cv_score = np.zeros((len(delays), len(alphas)))
	std_cv_score = np.zeros((len(delays), len(alphas)))
	in_sample_r = np.zeros((len(delays), len(alphas)))
	for i in range(len(delays)):
		dat = split_data(stim, resp, 2, 0.8, delays[i])
		xtrain = flatten_spct(remove_channels(dat.train_stim, [3, 3]))
		ytrain = dat.train_resp
	
		# Always check for NaN before running
		assert not check_nan(xtrain)
		assert not check_nan(ytrain)

		for j in range(len(alphas)):
			start_time = time.time()
			r = Ridge(alphas[j], normalize = True)
			cv_scores = cross_val_score(r, xtrain, ytrain, cv = 5)
			mean_cv_score[i, j] = np.mean(cv_scores)
			std_cv_score[i, j] = np.std(std_cv_score)
			r.fit(xtrain, ytrain)
			plot_STRF(r, delays[i], 'Delay: %f, alpha: %f r2: %f'
			% (delays[i], alphas[np.argmax(mean_cv_score[i, :])], mean_cv_score[i, j]), '%d_%d' % (i, j))

			print("---%s seconds---" % (time.time() - start_time))


		rmax = Ridge(alphas[np.argmax(mean_cv_score)], normalize = True)
		rmax.fit(xtrain, ytrain)
		xtest = flatten_spct(remove_channels(dat.test_stim, [3, 3]))
		ypred = rmax.predict(xtest)
		r2scores[i] = r2_score(dat.test_resp, ypred)
	return r2scores, in_sample_r

'''normalized reverse correlation method as described in 
Estimating spatio-temporal receptive fields of auditory and visual neurons from 
their responses to natural stimuli.

and

Estimating sparse spectro-temporal receptive fields with natural stimuli.'''
# Threshold: Proportion of the singular values to retain. The smallest
# (1 - threshold) proportion of singular values will be discarded
def nrc(x, y, threshold, approach = 1):
	# Always check for NaN before running
	assert not check_nan(x)
	assert not check_nan(y)

	# Iterate over data points and average together to find the stimulus
	# autocorrelation and stimulus-response cross correlation matricies 
	# Assumes stimulus spectrogram is flattened
	
	# Approach 1: Average together the matricies and then calculate h
	if approach == 1:
		Css = np.zeros((x.shape[1], x.shape[1]))
		Csr = np.zeros(x.shape[1])
		for i in range(y.size):		
			# Always check for NaN before running
			assert not check_nan(x)
			assert not check_nan(y)

#			start_time = time.time()
			if (i % 100) == 0:
				print('%d/%d\n' % (i, y.size))
			# Flip the stimulus so that data points that are most recent to 
			# the recorded response come first
			Css += np.outer(np.flip(x[i, :]), np.flip(x[i, :]))
			Csr += y[i] * np.flip(x[i, :])
#			print("---%s seconds---" % (time.time() - start_time))
		Css *= 1/y.size
		Csr *= 1/y.size
		# Convert threshold, which is a fraction of the response to keep, to
		# a numerical value
		_, s, _ = svd(Css)
		s = np.sort(s)
		s = s[:math.ceil(threshold * s.size)]
		cutoff = s[0]
		h = pinv(Css, cutoff) @ Csr
		# Always check for NaN before running
		assert not check_nan(x)
		assert not check_nan(y)

		# Approach 2: Calculate h for each data point and then average together 
		# h at the end
	elif approach == 2:
		h = np.zeros(y.size)
		for i in range(y.size):
			Css = np.outer(np.flip(x[i, :]), np.flip(x[i, :]))
			Csr = y[i] * np.flip(x[i, :])
			h += pinv(Css, threshold) @ Csr
		h *= 1/y.size

	return h

# Plot the STRF derived by the model (sklearn object)
def plot_STRF(model, delay_time, title, figdir, fname):
	weights = np.reshape(model.coef_, (delay_time, int(model.coef_.size/delay_time)))
	plt.pcolor(weights)
	plt.title(title)
#	figdir = os.path.join('%s/STRF' % figpath, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	plt.savefig('%s/%s.png' % (figdir, fname))
	plt.close()
