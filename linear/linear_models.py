import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from utils.process import split_data, flatten_spct
from utils.misc import get_fig_path
from scipy.stats import pearsonr
import pdb, math

# Do vanilla ridge regression on a single electrode for average high gamma channel
def ridge_regression(xtrain, ytrain, xtest, ytest, alphas=[1]):
	
	# Sweep over a range of regularization strength and use cross-validation 
	# to select the best one
	mean_cv_score = np.zeros(len(alphas))
	std_cv_score = np.zeros(len(alphas))
	for i, alpha in enumerate(alphas):
		start_time = time.time()
		r = Ridge(alpha, fit_intercept = True, normalize = True)
		cv_scores = cross_val_score(r, xtrain, ytrain, cv = 5)
		mean_cv_score[i] = np.mean(cv_scores)
		std_cv_score[i] = np.std(std_cv_score)
		print("---%s seconds---" % (time.time() - start_time))

	# Return the fit with the best mean_cv_score
	r.fit(xtrain, ytrain)
	r.predict(xtest)
	ypred = r.predict(xtest)
	print(r2_score(ytest, ypred))

	return r

# To do:
# Cross validation

# Boosting as described in "Estimating sparse spectro-temporal receptive fields with
# natural stimuli"
def boosting(x, y, xtest, ytest, n_iterations):
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
def search_params(stim, resp):
	# Assuming 100 Hz sampling rate
	delays = np.array([50, 100, 150])
	alphas = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2])
	r2scores = np.zeros(len(delays))
	for i in range(len(delays)):
		dat = split_data(stim, resp, 2, delays[i])
		xtrain = flatten_spct(dat.train_stim)
		ytrain = dat.train_resp
		mean_cv_score = np.zeros(len(alphas))
		std_cv_score = np.zeros(len(alphas))
		for j in range(len(alphas)):
			start_time = time.time()
			r = Ridge(alphas[j], normalize = True)
			cv_scores = cross_val_score(r, xtrain, ytrain, cv = 5)
			mean_cv_score[j] = np.mean(cv_scores)
			std_cv_score[j] = np.std(std_cv_score)
			print("---%s seconds---" % (time.time() - start_time))
		rmax = Ridge(alphas[np.argmax(mean_cv_score)], normalize = True)
		rmax.fit(xtrain, ytrain)
		xtest = flatten_spct(dat.test_stim)
		ypred = rmax.predict(xtest)
		r2scores[i] = r2_score(dat.test_resp, ypred)
		plot_STRF(rmax, delays[i], 'Delay: %f, alpha: %f r2: %f'
			% (delays[i], alphas[np.argmax(mean_cv_score)], r2scores[i]), i)
	return r2scores

# Plot the STRF derived by the model (sklearn object)
def plot_STRF(model, delay_time, title, fname):
	weights = np.reshape(model.coef_, (delay_time, 100))
	plt.pcolor(weights)
	plt.title(title)
	figpath = get_fig_path()
	plt.savefig('%s/STRF/%s.png' % (figpath, fname))
	plt.close()
