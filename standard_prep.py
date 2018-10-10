x = loadmat('%s/HighGammaAllChannels10sMovingZ.mat' % misc.get_data_path())
e105 = x['data'][:, 104]
stim = loadmat('%s/DynRip.cchspct.6oct.100hz' % misc.get_data_path())
stim, resp = preprocess.align(stim['spct'], e105)
p = process.split_data(stim, resp, 2, train_split = 0.4)