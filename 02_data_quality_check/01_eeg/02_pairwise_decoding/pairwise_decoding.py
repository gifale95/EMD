"""Pairwise decoding of the biological EEG test data.

Parameters
----------
sub : int
	Used subject.
sfreq : int
	Downsampling frequency.
channels : str
	Whether to retain occipital ['O'], posterior ['P'], temporal ['T'],
	central ['C'], frontal ['F'] occipital/parital ['OP'] or all ['all']
	channels.
time_split : int
	Integer from 1 to 10 indicating the time points split used.
zscore : int
	If '1', z-score the data prior to decoding.
data_split : str
	Whether to decode the 'test' or 'train' split.
pseudo_trials : int
	If 1, create pseudo-trials.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.svm import SVC


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int) # [1, 2, 3, 4, 5, 6]
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--channels', default='all', type=str) # ['O', 'P', 'T', 'C', 'F', 'all']
parser.add_argument('--time_split', default=1, type=int) # ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--data_split', default='test', type=str)
parser.add_argument('--pseudo_trials', default=1, type=int)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()

print('>>> Pairwise decoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the biological EEG data
# =============================================================================
data_dir = os.path.join(args.project_dir, 'dataset', 'preprocessed_data',
	'dataset_02', 'eeg', 'sub-'+format(args.sub,'02'), 'sfreq-'+
	format(args.sfreq,'04'), 'preprocessed_data.npy')
data_dict = np.load(data_dir, allow_pickle=True).item()
times = data_dict['times']
ch_names = data_dict['ch_names']
eeg_info = data_dict['eeg_info']
eeg_data_list = data_dict['eeg_data']
stimuli_presentation_order_list = data_dict['stimuli_presentation_order']
del data_dict

# z-score the data
for s in range(len(eeg_data_list)):
	if s == 0:
		eeg_data = eeg_data_list[s]
		stimuli_presentation_order = stimuli_presentation_order_list[s]
	else:
		eeg_data = np.append(eeg_data, eeg_data_list[s], 0)
		stimuli_presentation_order = np.append(stimuli_presentation_order,
			stimuli_presentation_order_list[s], 0)
del eeg_data_list, stimuli_presentation_order_list

# Select the data split
if args.data_split == 'train':
	idx = np.where(stimuli_presentation_order <= 1000)[0]
elif args.data_split == 'test':
	idx = np.where(stimuli_presentation_order > 1000)[0]
eeg_data = eeg_data[idx]
stimuli_presentation_order = stimuli_presentation_order[idx]


# =============================================================================
# Channels selection
# =============================================================================
# Retain the EEG channels of the chosen channel type
if args.channels != 'OP' and args.channels != 'all':
	kept_ch_names = []
	idx_ch = []
	for c, chan in enumerate(ch_names):
		if args.channels in chan:
			kept_ch_names.append(chan)
			idx_ch.append(c)
	idx_ch = np.asarray(idx_ch)
	eeg_data = eeg_data[:,idx_ch]
	ch_names = kept_ch_names
elif args.channels == 'OP':
	kept_ch_names = []
	idx_ch = []
	for c, chan in enumerate(ch_names):
		if 'O' in chan or 'P' in chan:
			kept_ch_names.append(chan)
			idx_ch.append(c)
	idx_ch = np.asarray(idx_ch)
	eeg_data = eeg_data[:,idx_ch]
	ch_names = kept_ch_names


# =============================================================================
# Time points split selection
# =============================================================================
time_per_split = int(np.ceil(len(times) / 10))
idx_start = time_per_split * (args.time_split - 1)
idx_end = idx_start + time_per_split
eeg_data = eeg_data[:,:,idx_start:idx_end]


# =============================================================================
# Perform the pairwise decoding
# =============================================================================
# Results array of shape:
# (Video conditions × Video conditions × EEG time points)
video_conditions = np.unique(stimuli_presentation_order)
pairwise_decoding = np.zeros((len(video_conditions), len(video_conditions),
	eeg_data.shape[2]))

# Loop over EEG time points, image-conditions and EEG repetitions
for t in tqdm(range(eeg_data.shape[2])):
	for v1 in range(len(video_conditions)):
		for v2 in range(v1):
			# Select the video conditions data
			idx_1 = resample(np.where(
				stimuli_presentation_order == video_conditions[v1])[0],
				replace=False)
			idx_2 = resample(np.where(
				stimuli_presentation_order == video_conditions[v2])[0],
				replace=False)
			eeg_cond_1 = eeg_data[idx_1,:,t]
			eeg_cond_2 = eeg_data[idx_2,:,t]
			# Select a minimum amount of trials in case repeats are not the
			# same between two test image conditions
			if len(eeg_cond_1) < len(eeg_cond_2):
				eeg_cond_2 = eeg_cond_2[:len(eeg_cond_1)]
			elif len(eeg_cond_2) < len(eeg_cond_1):
				eeg_cond_1 = eeg_cond_1[:len(eeg_cond_2)]
			# Create pseudo-trials
			if args.pseudo_trials == 1:
				if args.data_split == 'test':
					n_ptrials_repeats = 6
				elif args.data_split == 'train':
					n_ptrials_repeats = 2
				n_pseudo_trials = int(
					np.ceil(len(eeg_cond_1) / n_ptrials_repeats))
				pseudo_data_1 = np.zeros((n_pseudo_trials,
					eeg_cond_1.shape[1]))
				pseudo_data_2 = np.zeros((n_pseudo_trials,
					eeg_cond_2.shape[1]))
				for r in range(n_pseudo_trials):
					idx_start = r * n_ptrials_repeats
					idx_end = idx_start + n_ptrials_repeats
					pseudo_data_1[r] = np.mean(eeg_cond_1[idx_start:idx_end],
						0)
					pseudo_data_2[r] = np.mean(eeg_cond_2[idx_start:idx_end],
						0)
				eeg_cond_1 = pseudo_data_1
				eeg_cond_2 = pseudo_data_2
			# SVM target vectors
			y_train = np.zeros(((len(eeg_cond_1)-1)*2))
			y_train[int(len(y_train)/2):] = 1
			y_test = np.asarray((0, 1))
			scores = np.zeros(len(eeg_cond_1))
			for r in range(len(eeg_cond_1)):
				# Define the training/test partitions
				X_train = np.append(np.delete(eeg_cond_1, r, 0),
					np.delete(eeg_cond_2, r, 0), 0)
				X_test = np.append(np.expand_dims(eeg_cond_1[r], 0),
					np.expand_dims(eeg_cond_2[r], 0), 0)
				# Train the classifier
				dec_svm = SVC(kernel='linear')
				dec_svm.fit(X_train, y_train)
				# Test the classifier
				y_pred = dec_svm.predict(X_test)
				scores[r] = sum(y_pred == y_test) / len(y_test)
			# Store the accuracy
			pairwise_decoding[v1,v2,t] = np.mean(scores)
			pairwise_decoding[v2,v1,t] = pairwise_decoding[v1,v2,t]


# =============================================================================
# Save the results
# =============================================================================
results_dict = {
	'args': args,
	'pairwise_decoding': pairwise_decoding,
	'times': times,
	'ch_names': ch_names,
	'eeg_info': eeg_info
}

save_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eeg',
	'pairwise_decoding', 'sub-'+format(args.sub,'02'), 'sfreq-'+
	format(args.sfreq,'04'), 'data_split-'+args.data_split, 'channels-'+
	args.channels, 'pseudo_trials-'+format(args.pseudo_trials,'02'))
file_name = 'pairwise_decoding_time_split-' + format(args.time_split, '02')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), results_dict)
