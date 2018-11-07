# import required libraries
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.metrics import r2_score
from tensorflow.python.framework import ops
import pdb, time
import itertools


# Modified from 
# https://github.com/nfmcclure/tensorflow_cookbook/blob/master/
# 03_Linear_Regression/06_Implementing_Lasso_and_Ridge_Regression/
# 06_lasso_and_ridge_regression.py

def ridge_regression(xtrain, ytrain, xtest, ytest, params):
    # Default values:
    if 'batch_size' in params.keys():
        batch_size = params['batch_size']
    else:
        batch_size = 50
    if 'lrate' in params.keys():
        lrate = params['lrate']
    else:
        lrate = 0.001
    if 'alpha' in params.keys():
        alpha = params['alpha']
    else:
        alpha = 1
    if 'nsteps' in params.keys():
        nsteps = params['nsteps']
    else:
        nsteps = 15000

    # Do ridge regression
    regression_type = 'Ridge'

    # clear out old graph
    ops.reset_default_graph()

    # Create graph
    sess = tf.Session()

    ###
    # Model Parameters
    ###


    # Initialize placeholders
    x_data = tf.placeholder(shape=[None, xtrain.shape[1]], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

    # make results reproducible
    seed = 13
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Create variables for linear regression
    # A = tf.Variable(tf.random_normal(shape=[1,1]))
    # b = tf.Variable(tf.random_normal(shape=[1,1]))

    params = tf.Variable(tf.random_normal(shape = [xtrain.shape[1], 1]))

    # Declare model operations
    model_output = tf.matmul(x_data, params)

    ###
    # Loss Functions
    ###

    # Select appropriate loss function based on regression type

    if regression_type == 'LASSO':
        # Declare Lasso loss function
        # Lasso Loss = L2_Loss + heavyside_step,
        # Where heavyside_step ~ 0 if A < constant, otherwise ~ 99
        lasso_param = tf.constant(alpha)
        heavyside_step = tf.truediv(1., tf.add(1., tf.exp(tf.multiply(-50., tf.subtract(A, lasso_param)))))
        regularization_param = tf.multiply(heavyside_step, 99.)
        loss = tf.add(tf.reduce_mean(tf.square(y_target - model_output)), regularization_param)

    elif regression_type == 'Ridge':
        # Declare the Ridge loss function
        # Ridge loss = L2_loss + L2 norm of slope
        ridge_param = tf.constant(alpha, dtype=tf.float32)
        ridge_loss = tf.norm(params, ord=2)
        loss = tf.expand_dims(tf.add(tf.norm(y_target - model_output, ord=2), tf.multiply(ridge_param, ridge_loss)), 0)       

    else:
        print('Invalid regression_type parameter value',file=sys.stderr)


    ###
    # Optimizer
    ###

    # Declare optimizer
    my_opt = tf.train.AdamOptimizer(lrate)
    train_step = my_opt.minimize(loss)

    ###
    # Run regression
    ###

    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)

    # Training loop
    loss_vec = []
    for i in range(nsteps):

        # rand_index = np.random.choice(len(x_vals), size=batch_size)
        # rand_x = np.transpose([x_vals[rand_index]])
        # rand_y = np.transpose([y_vals[rand_index]])

        # Parcel data into batch size
        # Since the data is already randomized, loop through the data sequentially, 
        # repating when reaching the end
        batch_index = batch_size * (i - 1) % xtrain.shape[0]       
        xbatch = xtrain[batch_index:min(batch_index + batch_size, xtrain.shape[0]), :]
        ybatch = ytrain[batch_index:min(batch_index + batch_size, ytrain.shape[0]), np.newaxis]
        sess.run(train_step, feed_dict={x_data: xbatch, y_target: ybatch})
        temp_loss = sess.run(loss, feed_dict={x_data: xbatch, y_target: ybatch})
        loss_vec.append(temp_loss[0])
#         if (i+1)%300==0:
# #            print('Step #' + str(i+1) + ' A = ' + str(sess.run(A)) + ' b = ' + str(sess.run(b)))
#             print('Loss = ' + str(temp_loss))
#             print('\n')

    # Need to evaluate the model
    
    # Extract Parameters
    params = sess.run(params)

    # In sample performance
    ypinsamp = np.zeros(ytrain.shape)
    for i in range(xtrain.shape[0]):
        ypinsamp[i] = xtrain[i, :] @ params
    insamp_r2 = r2_score(ytrain, ypinsamp)

    ypoutsamp = np.zeros(ytest.shape)
    for i in range(xtest.shape[0]):
        ypoutsamp[i] = xtest[i, :] @ params
    outsamp_r2 = r2_score(ytest, ypoutsamp)
    
    return params, insamp_r2, outsamp_r2, loss_vec


# Iterate through control parameters
def batch_ridge(xtrain, ytrain, xtest, ytest, save_params = False, **kwargs):
    # Make sure all arguments are iterable
    for key, val in kwargs.items():
        if not isinstance(val, list):
            kwargs[key] = [val]


    keys, values = kwargs.keys(), kwargs.values()
    i = 0
    data = []
    for inst in itertools.product(*values):
        start_time = time.time()
        arg_inst = dict(zip(keys, inst))
        p, r1, r2, l = ridge_regression(xtrain, ytrain, xtest, ytest, arg_inst)
        if not save_params:
            p = []
        data.append({'params': p, 'insamp_r2': r1, 'outsamp_r2': r2, 'loss_vec': l})
        print("---%s seconds---" % (time.time() - start_time))
    return data