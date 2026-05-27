def epoching(args):
	"""This function converts the EEG data to MNE raw format and performs
	filtering, epoching, current source density transform, frequency
	downsampling and baseline correction.

	Parameters
	----------
	args : Namespace
		Input arguments.

	Returns
	-------
	epoched_data : list of float
		List of epoched EEG data.
	stimuli_presentation_order : list of int
		List of stimuli conditions presentation order.
	eeg_info : Info
		EEG data info.
	times : float
		EEG time points.
	behavior : dict
		Dictionary containing behavioral responses.

	"""

	import os
	import numpy as np
	from glob import glob
	from scipy import io
	import mne
	import copy

	### Loop across data collection sessions ###
	epoched_data = []
	stimuli_presentation_order = []
	behavior = []
	eeg_info = []
	for s in range(args.n_ses):

		### Load the stimuli presentation order ###
		stim_order = io.loadmat(os.path.join(args.project_dir, 'dataset',
			'source_data', 'dataset_02', 'sub-'+format(args.sub,'02'), 'ses-'+
			format(s+1,'02'), 'stim_order_sub-'+format(args.sub,'02')+
			'_sess-'+format(s+1,'02')+'.mat'))
		stim_order = stim_order['stim_order']
		stim_order = np.reshape(stim_order, -1, order='F')
		# Remove missing EEG data:
		# Subject 1, session 3, run 10, trial 1
		if args.sub == 1 and (s+1) == 3:
			stim_order = np.delete(stim_order, 594)
		stimuli_presentation_order.append(copy.copy(stim_order))

		### Load the EEG trigger numbers ###
		eeg_triggers = np.zeros(len(stim_order), dtype=int)
		for i, stim in enumerate(stim_order):
			if stim < 100:
				eeg_triggers[i] = stim
			else:
				eeg_triggers[i] = int(str(stim)[:2])

		### Load the behavioral data ###
		behav_files = glob(os.path.join(args.project_dir, 'dataset',
			'source_data', 'dataset_02', 'sub-'+format(args.sub,'02'), 'ses-'+
			format(s+1,'02'), 'run-*.mat'))
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
		beh = {}
		beh['response'] = beh_response
		beh['correctness'] = beh_correctness
		behavior.append(copy.copy(beh))
		del beh

		### Get the EEG files, and loop over session runs ###
		eeg_files = glob(os.path.join(args.project_dir, 'dataset',
			'source_data', 'dataset_02', 'sub-'+format(args.sub,'02'), 'ses-'+
			format(s+1,'02'), '*.vhdr'))
		eeg_files.sort()
		for f, file in enumerate(eeg_files):

			### Load the raw EEG data ###
			# Load the raw EEG data
			raw = mne.io.read_raw_brainvision(file, preload=True)

			### Filter the data ###
			if args.highpass != None and args.lowpass != None:
				raw = raw.copy().filter(l_freq=args.highpass, h_freq=args.lowpass)

			### Epoch the raw EEG data ###
			# Get the event samples info
			events, _ = mne.events_from_annotations(raw)
			events = events[1:]
			# Epoch the EEG trials
			epochs = mne.Epochs(raw, events, tmin=args.tmin, tmax=args.tmax,
				baseline=None, preload=True)
			del raw

			### Compute current source density ###
			if args.csd == 1:
				# Create the channels montage file
				ch_pos = {}
				for c, dig in enumerate(epochs.info['dig']):
					if c > 2:
						if epochs.info['ch_names'][c-3] in epochs.info['ch_names']:
							ch_pos[epochs.info['ch_names'][c-3]] = dig['r']
				montage = mne.channels.make_dig_montage(ch_pos)
				# Apply the montage to the epoched data
				epochs.set_montage(montage)
				# Compute current source density
				epochs = mne.preprocessing.compute_current_source_density(
					epochs, lambda2=1e-05, stiffness=4)

			### Resample the epoched data ###
			if args.sfreq < 1000:
				epochs.resample(args.sfreq)

			### Store the EEG info ###
			if s == 0:
				eeg_info.append(copy.deepcopy(epochs.info))
				times = epochs.times

			### Baseline correction ###
			if args.baseline_correction == 1:
				epochs = mne.baseline.rescale(epochs.get_data(), epochs.times,
					baseline=(None, 0), mode=args.baseline_mode)

			### Sort the epoched data ###
			if f == 0:
				epochs_all_runs = epochs
				all_events_num = events[:,2]
			else:
				epochs_all_runs = np.append(epochs_all_runs, epochs, 0)
				all_events_num = np.append(all_events_num, events[:,2], 0)
			del epochs, events

		### Store the epoched data ###
		epoched_data.append(epochs_all_runs)
		del epochs_all_runs

		### Match the EEG triggers with stimuli presentation order ###
		if not all(all_events_num == eeg_triggers):
			# Missing EEG data:
			# Subject 1, session 3, run 10, trial 1
			if args.sub == 1 and (s+1) == 3:
				pass
			else:
				raise Exception('EEG events do not match with stimuli presentation order!')

	### Output ###
	return epoched_data, stimuli_presentation_order, eeg_info, times, behavior


def zscore(epoched_data):
	"""z-score the EEG data at each recording session.

	Parameters
	----------
	epoched_data : list of float
		List of epoched EEG data.

	Returns
	-------
	eeg_data_zscored : list of float
		List of z-scored epoched EEG data.

	"""

	import numpy as np
	from sklearn.preprocessing import StandardScaler

	### z-score the data of each session ###
	eeg_data_zscored = []
	for s in range(len(epoched_data)):
		ses_data = epoched_data[s]
		ses_data_shape = ses_data.shape
		ses_data = np.reshape(ses_data, (len(ses_data), -1))
		scaler = StandardScaler()
		ses_data = scaler.fit_transform(ses_data)
		ses_data = np.reshape(ses_data, ses_data_shape)
		eeg_data_zscored.append(ses_data)

	### Output ###
	return eeg_data_zscored


def compute_ncsnr(zscored_test):
	"""Compute the noise ceiling as in the NSD paper, using the test split
	data.

	Parameters
	----------
	zscored_test : list of float
		zscored test EEG data.

	Returns
	-------
	ncsnr : float
		Noise ceiling SNR.

	"""

	import numpy as np
	from copy import copy
	from sklearn.preprocessing import StandardScaler

	### Standardize the data at each scan session ###
	for s in range(len(zscored_test)):
		data_shape = zscored_test[s].shape
		provv_data = np.reshape(copy(zscored_test[s]),
			(data_shape[0]*data_shape[1],-1))
		scaler = StandardScaler()
		provv_data = scaler.fit_transform(provv_data)
		if s == 0:
			zscored_data = np.reshape(provv_data, data_shape)
		else:
			zscored_data = np.append(zscored_data, np.reshape(
				provv_data, data_shape), 1)

	### Compute the ncsnr ###
	std_noise = np.sqrt(np.mean(np.var(zscored_data, axis=1, ddof=1), 0))
	std_signal = 1 - (std_noise ** 2)
	std_signal[std_signal<0] = 0
	std_signal = np.sqrt(std_signal)
	ncsnr = std_signal / std_noise

	### Output ###
	return ncsnr
