xtrain = process.norm_spct(process.remove_channels(p.train_stim, [3, 3]))
xtest = process.norm_spct(process.remove_channels(p.test_stim, [3, 3]))
ytrain = p.train_resp
ytest = p.test_resp