x = loadmat('%s/R6_B11_10sMovingZ.mat' % misc.get_data_path())
stim = loadmat('%s/DynRip.cchspct.6oct.100hz' % misc.get_data_path())
stim, resp = preprocess.align(stim['spct'], x['data'][:, 40])
p = process.split_data(stim, resp, 2, train_split = 0.8)