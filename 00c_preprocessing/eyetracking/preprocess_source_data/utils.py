def get_behavior(args, session):
	"""Get the behavioral results.

	Parameters
	----------
	args : Namespace
		Input arguments.
	session : str
		EEG recording session.

	Returns
	-------
	epoched_data : float
		Epoched EEG data.
	stim_order : int
		Stimuli conditions presentation order.
	ch_names : list of str
		EEG channel names.
	times : float
		EEG time points.
	info : Info
		EEG data info.
	beh_ground_truth : int
		Ground truth behavioral responses.
	beh_response : int
		Behavioral responses.

	"""

	import os
	import numpy as np
	from glob import glob
	from scipy import io

	### Load the stimuli presentation order ###
	stim_order = io.loadmat(os.path.join(args.project_dir, 'dataset',
		'source_data', 'dataset_02', 'sub-'+format(args.sub,'02'), 'ses-'+
		format(session+1,'02'), 'stim_order_sub-'+format(args.sub,'02')+
		'_sess-'+format(session+1,'02')+'.mat'))
	stim_order = stim_order['stim_order']
	stim_order = np.reshape(stim_order, -1, order='F')

	### Load the behavioral data ###
	behav_files = glob(os.path.join(args.project_dir, 'dataset', 'source_data',
		'dataset_02', 'sub-'+format(args.sub,'02'), 'ses-'+
		format(session+1,'02'), 'run-*.mat'))
	behav_files.sort()
	for f, file in enumerate(behav_files):
		behav = io.loadmat(file)['data']
		res = np.asarray(behav[0][0][7]['response'][0], dtype=float)
		correct = np.asarray(behav[0][0][7]['correctness'][0], dtype=float)
		if f == 0:
			response = res
			correctness = correct
		else:
			response = np.append(response, res, 0)
			correctness = np.append(correctness, correct, 0)

	### Create the behavioral results matrices of each trial ###
	# (0==left_key_press, 1==right_key_press, 2==no_response,
	# 99==no_task_trials)
	beh_response = np.zeros(len(response), dtype=int)
	beh_response[:] = 99
	beh_correctness = np.zeros(len(correctness), dtype=int)
	beh_correctness[:] = 99
	for t in range(len(beh_response)):
		if ~np.isnan(response[t]):
			beh_response[t] = response[t]
		if ~np.isnan(correctness[t]):
			beh_correctness[t] = correctness[t]

	### Output ###
	return stim_order, beh_response, beh_correctness


def epoch_eyetrack(args, session):
	"""Import the eytrack data and perform epoching, baseline correction and
	frequency downsampling.
	https://github.com/cvnlab/nsddatapaper/blob/main/main/nsd_et_preprocessing.m
	https://stackoverflow.com/questions/44779315/detrending-data-with-nan-value-in-scipy-signal

	Parameters
	----------
	args : Namespace
		Input arguments.
	session : str
		EEG recording session.

	Returns
	-------
	eyetrack_data : float
		Epoched eyetrack data.
	eyetrack_times : float
		Eyetrack time points.

	"""

	import os
	import numpy as np
	import pandas as pd
	from scipy import stats
	from copy import copy
	from tqdm import tqdm

	### Data directories ###
	data_dir = os.path.join(args.project_dir, 'dataset', 'source_data',
		'dataset_02', 'sub-'+format(args.sub,'02'), 'ses-'+
		format(session+1,'02'))
	num_runs = 16

	for f in tqdm(range(num_runs)):
		### Load the data ###
		# Import the eyelink samples, and set to NaN missing values during eye
		# blinks
		file_path = os.path.join(data_dir, 'r'+format(f+1, '02'))
		samples = pd.read_table(file_path+'_samples.asc', na_values=['   .'],
			header=None)
		samples = samples.drop(axis=1, labels=4)
		samples = samples.to_numpy()
		# Import the eyelink events
		events = pd.read_table(file_path+'_events.asc', sep='\s+', header=None)
		if len(events) != 66:
			raise Exception('Incorrect number of eyelink events!')
		events = np.squeeze(events[[1]].to_numpy())
		# Import the eyeblinks
		blinks = pd.read_table(file_path+'_eblink.asc', sep='\s+', header=None)
		blinks = np.squeeze(blinks[[2, 3]].to_numpy())
		# Set negative and zero (artefacts) to NaNs
		samples[samples <= 0] = np.nan

		### Remove eye blinks ###
		# Set to NaN samples 250 ms before and 250 ms after eye blinks. For
		# this we consider the eyelink-detected blinks & when the pupil size is
		# zero
		blinks[:,0] = blinks[:,0] - 250
		blinks[:,1] = blinks[:,1] + 250
		for b in range(len(blinks)):
			if blinks[b,0] < samples[0,0]:
				idx_start = 0
			else:
				idx_start = np.where(samples[:,0] == blinks[b,0])[0][0]
			if blinks[b,1] > samples[-1,0]:
				idx_end = int(samples[-1,0])
			else:
				idx_end = np.where(samples[:,0] == blinks[b,1])[0][0]
			samples[idx_start:idx_end,1:4] = np.nan
		if any(samples[:,1:4].flatten() == 0):
			raise Exception('Zero values!')

		### Align eyetracking data with imaging data ###
		samples_start = int(np.where(samples[:,0] == events[0])[0][0] - \
			(abs(args.tmin) * 1000) - int(1 * 1000))
		samples_end = int(np.where(samples[:,0] == events[-1])[0][0] + \
			(args.tmax * 1000) + int(1 * 1000))
		samples = samples[samples_start:samples_end]

		### Convert gaze data to visual degrees ###
		stim_deg = 5 # stimulus size in degrees
		stim_px = 194 # stimulus size in pixels
		deg_per_px = stim_deg / stim_px
		samples[:,1:3] = samples[:,1:3] * deg_per_px

		### Detrend and mean center the X-Y gaze data ###
		# Remove slow signal drift by linear detrending and mean-centering of
		# the gaze position time series (X and Y). This step assumes that the
		# mean gaze position corresponds to central fixation (the presentation
		# screen pixel resolution is 1680×1050). The pupil size is also
		# mean-centered
#		for c in range(1,samples.shape[1]-1):
#			non_nan_idx = ~np.isnan(samples[:,c])
#			m, b, _, _, _ = stats.linregress(
#				np.arange(len(samples))[non_nan_idx],
#				samples[:,c][non_nan_idx])
#			samples[:,c] = samples[:,c] - ((m * np.arange(len(samples))) + b)
#		for c in range(1,samples.shape[1]):
#			samples[:,c] = samples[:,c] - np.nanmean(samples[:,c])

		### Epoch the data ###
		epoched_data = []
		for e in events:
			idx_onset = np.where(samples[:,0] == e)[0]
			idx_start = int(idx_onset - (abs(args.tmin) * 1000))
			idx_end = int(idx_onset + (args.tmax * 1000))
			epoched_data.append(samples[idx_start:idx_end,1:4])
		epoched_data = np.asarray(epoched_data)
		eyetrack_times = np.round(np.arange(args.tmin, args.tmax, 0.001), 3)
		eyetrack_times[eyetrack_times==0] = 0

		### Mean centering ###
		idx_start = np.where(eyetrack_times == args.tmin)[0][0]
		idx_end = np.where(eyetrack_times == 0)[0][0]
		if args.mean_centering != 'none':
			for e in range(len(epoched_data)):
				if args.mean_centering == 'baseline':
					mean = np.nanmean(epoched_data[e,idx_start:idx_end], 0)
				elif args.mean_centering == 'epoch':
					mean = np.nanmean(epoched_data[e], 0)
				if np.isnan(mean[0]):
					epoched_data[e] = np.nan
				else:
					epoched_data[e] = epoched_data[e] - mean

		### Exclude samples deviating more than 3° from central fixation ###
		# Excise data 250 ms before and 250 ms after each occurrence
		cutoff = 3
		for e in range(len(epoched_data)):
			euclDist = np.sqrt(np.power((epoched_data[e,:,0]), 2) + \
				np.power((epoched_data[e,:,1]), 2))
			idx_bad = np.where(euclDist > cutoff)[0]
			for s in idx_bad:
				epoched_data[e,s-250:s+250,0:3] = np.nan

		### Temporal smoothing using a 50ms running average ###
		# Smooth the time-series data for gaze position and pupil size using a
		# 50-ms running average
#		smooth_epoched_data = copy(epoched_data)
#		smooth_ms = 50
#		for e in tqdm(range(len(epoched_data))):
#			for t in range(epoched_data.shape[1]):
#				if t < (smooth_ms / 2):
#					idx_start = 0
#				else:
#					idx_start = int(t - (smooth_ms/2))
#				if t > (epoched_data.shape[1] - smooth_ms / 2):
#					idx_end = epoched_data.shape[1]
#				else:
#					idx_end = int(t + (smooth_ms/2))
#				for c in range(epoched_data.shape[2]):
#					if np.isnan(epoched_data[e,t,c]) == False:
#						smooth_epoched_data[e,t,c] = np.nanmean(
#							epoched_data[e,idx_start:idx_end,c])

		### Downsample the data ###
		if args.sfreq < 1000:
			step = int(1000 / args.sfreq)
			idx = np.arange(0, len(eyetrack_times), step)
			epoched_data = epoched_data[:,idx]
			eyetrack_times = eyetrack_times[idx]

		# Store the epoched data
		if f == 0:
			eyetrack_data = epoched_data.astype(np.float32)
		else:
			eyetrack_data = np.append(eyetrack_data,
				epoched_data.astype(np.float32), 0)

	### Output ###
	return eyetrack_data, eyetrack_times
