"""Temporal generalization analysis on the biological EEG test data. A SVM
classifier trained on one time point is tested on all time points.

Parameters
----------
sub : int
	Used subject.
baseline_correction : int
	Whether to baseline correct [1] or not [0] the data.
mvnn : str
	Type of MVNN applied to preprocess the data ['none', 'time'].
sfreq : int
	Downsampling frequency.
lowpass : float
	Lowpass filter frequency.
highpass : float
	Highpass filter frequency.
zscore : int
	Whether to z-score [1] or not [0] the data.
channels : str
	Whether to retain occipital ['O'], posterior ['P'], temporal ['T'],
	central ['C'], frontal ['F'] occipital/parital ['OP'] or all ['all']
	channels.
tot_time_splits : int
	Amount of total time splits.
time_split : int
	Integer indicating the time points split used.
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
parser.add_argument('--sub', default=1, type=int) # ['1' '2']
parser.add_argument('--baseline_correction', default=1, type=int)
parser.add_argument('--mvnn', default='time', type=str) # ['none' 'time']
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--lowpass', default=100, type=float)
parser.add_argument('--highpass', default=0.01, type=float)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--channels', default='all', type=str) # ['OP' 'all']
parser.add_argument('--tot_time_splits', default=185, type=int)
parser.add_argument('--time_split', default=1, type=int) # `seq 1 185`
parser.add_argument('--data_split', default='test', type=str)
parser.add_argument('--pseudo_trials', default=1, type=int)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()

print('>>> Temporal generalization analysis <<<')
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
	'dataset_02', 'eeg', 'sub-'+format(args.sub,'02'), 'mvnn-'+args.mvnn,
	'baseline_correction-'+format(args.baseline_correction, '02'), 'highpass-'+
	format(args.highpass)+'_lowpass-'+format(args.lowpass), 'sfreq-'+
	format(args.sfreq,'04'), 'preprocessed_data.npy')
data_dict = np.load(data_dir, allow_pickle=True).item()
times = data_dict['times']
ch_names = data_dict['ch_names']
info = data_dict['info']
eeg_data_list = data_dict['eeg_data']
stimuli_presentation_order_list = data_dict['stimuli_presentation_order']
del data_dict

# z-score the data
for s in range(len(eeg_data_list)):
	data_shape = eeg_data_list[s].shape
	data_provv = np.reshape(eeg_data_list[s], (len(eeg_data_list[s]), -1))
	if args.zscore == 1:
		scaler = StandardScaler()
		data_provv = scaler.fit_transform(data_provv)
	data_provv = np.reshape(data_provv, data_shape)
	if s == 0:
		eeg_data = data_provv
		stimuli_presentation_order = stimuli_presentation_order_list[s]
	else:
		eeg_data = np.append(eeg_data, data_provv, 0)
		stimuli_presentation_order = np.append(stimuli_presentation_order,
			stimuli_presentation_order_list[s], 0)
	del data_provv
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
time_per_split = int(np.ceil(len(times) / args.tot_time_splits))
idx_start = time_per_split * (args.time_split - 1)
idx_end = idx_start + time_per_split
train_time_range = np.arange(idx_start, idx_end)


# =============================================================================
# Perform the temporal generalization analysis
# =============================================================================
# Results array of shape:
# (Pairwise comparisons × Pairwise comparisons × EEG train time × EEG test time)
video_conditions = np.unique(stimuli_presentation_order)
temporal_generalization = np.zeros((len(video_conditions),
	len(video_conditions), len(train_time_range), eeg_data.shape[2]))

# Loop over video-conditions
for v1 in tqdm(range(len(video_conditions))):
	for v2 in range(v1):

		# Select the video conditions data
		idx_1 = resample(np.where(
			stimuli_presentation_order == video_conditions[v1])[0],
			replace=False)
		idx_2 = resample(np.where(
			stimuli_presentation_order == video_conditions[v2])[0],
			replace=False)
		eeg_cond_1 = eeg_data[idx_1]
		eeg_cond_2 = eeg_data[idx_2]

		# Select a minimum amount of trials in case repeats are not the
		# same between two test image conditions
		if len(eeg_cond_1) < len(eeg_cond_2):
			eeg_cond_2 = eeg_cond_2[:len(eeg_cond_1)]
		elif len(eeg_cond_2) < len(eeg_cond_1):
			eeg_cond_1 = eeg_cond_1[:len(eeg_cond_2)]

		# Create pseudo-trials
		if args.pseudo_trials == 1:
			if args.data_split == 'test':
				n_ptrials_repeats = 4
			elif args.data_split == 'train':
				n_ptrials_repeats = 2
			n_pseudo_trials = int(np.ceil(len(eeg_cond_1) / n_ptrials_repeats))
			pseudo_data_1 = np.zeros((n_pseudo_trials, eeg_cond_1.shape[1],
				eeg_cond_1.shape[2]))
			pseudo_data_2 = np.zeros((n_pseudo_trials, eeg_cond_2.shape[1],
				eeg_cond_2.shape[2]))
			for r in range(n_pseudo_trials):
				idx_start = r * n_ptrials_repeats
				idx_end = idx_start + n_ptrials_repeats
				pseudo_data_1[r] = np.mean(eeg_cond_1[idx_start:idx_end], 0)
				pseudo_data_2[r] = np.mean(eeg_cond_2[idx_start:idx_end], 0)
			eeg_cond_1 = pseudo_data_1
			eeg_cond_2 = pseudo_data_2

		# SVM target vectors
		y_train = np.zeros(((len(eeg_cond_1)-1)*2))
		y_train[int(len(y_train)/2):] = 1
		y_test = np.asarray((0, 1))

		# Loop over EEG train time points and repeats
		for t1, train_time in enumerate(train_time_range):
			scores = np.zeros((len(eeg_cond_1), eeg_data.shape[2]))
			for r in range(len(eeg_cond_1)):

				# Define the SVM training partition
				X_train = np.append(np.delete(eeg_cond_1[:,:,train_time], r, 0),
					np.delete(eeg_cond_2[:,:,train_time], r, 0), 0)

				# Train the classifier
				dec_svm = SVC(kernel='linear')
				dec_svm.fit(X_train, y_train)

				# Loop over the EEG test time points
				for t2 in range(eeg_data.shape[2]):

					# Define the SVM test partition
					X_test = np.append(np.expand_dims(eeg_cond_1[r,:,t2], 0),
						np.expand_dims(eeg_cond_2[r,:,t2], 0), 0)

					# Test the trained classifies
					y_pred = dec_svm.predict(X_test)
					scores[r,t2] = sum(y_pred == y_test) / len(y_test)
					
			# Store the SVM decoding accuracy
			temporal_generalization[v1,v2,t1] = np.mean(scores, 0)
			temporal_generalization[v2,v1,t1] = \
				temporal_generalization[v1,v2,t1]


# =============================================================================
# Save the results
# =============================================================================
results_dict = {
	'temporal_generalization': temporal_generalization,
	'times': times,
	'ch_names': ch_names,
	'info': info
}

save_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eeg',
	'temporal_generalization_analysis', 'sub-'+format(args.sub,'02'), 'mvnn-'+
	args.mvnn, 'baseline_correction-'+format(args.baseline_correction, '02'),
	'highpass-'+format(args.highpass)+'_lowpass-'+format(args.lowpass),
	'sfreq-'+format(args.sfreq,'04'), 'data_split-'+args.data_split,
	'channels-'+args.channels, 'zscore-'+format(args.zscore, '02'),
	'pseudo_trials-'+format(args.pseudo_trials,'02'))
file_name = 'temporal_generalization_analysis_time_split-' + \
	format(args.time_split, '02')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), results_dict)
