"""Pairwise decoding of the eye-tracking data.

Parameters
----------
sub : int
	Used subject.
mean_centering : str
	Whether to mean center the eye-tracking data using the 500ms prior to video
	onset ['baseline'], the entire epoch ['epoch'], or to not center the data
	al all ['none'].
sfreq : int
	Downsampling frequency.
zscore : int
	Whether to z-score [1] or not [0] the data.
features : str
	Whether to decode using the X- and Y-gaze features ['gaze'], the X- and
	Y-gaze features without the variance linearly explained by the pupil size
	['gaze_independent'], the pupil size ['pupil'], all features ['all'], or
	all deatures with linearly independent gaze and pupil size
	['all_independent'].
time_split : int
	Integer from 1 to 10 indicating the time points split used.
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
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.svm import SVC


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int) # ['1' '2']
parser.add_argument('--mean_centering', default='baseline', type=str)
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--features', default='gaze', type=str) # ['gaze' 'gaze_independent' 'pupil' 'all' 'all_independent']
parser.add_argument('--time_split', default=1, type=int) # ['1' '2' '3' '4' '5' '6' '7' '8' '9' '10']
parser.add_argument('--data_split', default='test', type=str) # ['test' 'train']
parser.add_argument('--pseudo_trials', default=1, type=int)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()

print('>>> Pairwise decoding - Eye data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the eye-tracking data
# =============================================================================
data_dir = os.path.join(args.project_dir, 'dataset', 'preprocessed_data',
	'dataset_02', 'eye', 'sub-'+format(args.sub,'02'), 'mean_centering-'+
	args.mean_centering, 'sfreq-'+format(args.sfreq,'04'),
	'preprocessed_data.npy')
data_dict = np.load(data_dir, allow_pickle=True).item()
times = data_dict['eyetrack_times']
eye_data_list = data_dict['eyetrack_data']
stimuli_presentation_order_list = data_dict['stimuli_presentation_order']
del data_dict

# z-score the data
for s in range(len(eye_data_list)):
	data_shape = eye_data_list[s].shape
	data_provv = np.reshape(eye_data_list[s], (len(eye_data_list[s]), -1))
	if args.zscore == 1:
		scaler = StandardScaler()
		data_provv = scaler.fit_transform(data_provv)
	data_provv = np.reshape(data_provv, data_shape)
	if s == 0:
		eye_data = data_provv
		stimuli_presentation_order = stimuli_presentation_order_list[s]
	else:
		eye_data = np.append(eye_data, data_provv, 0)
		stimuli_presentation_order = np.append(stimuli_presentation_order,
			stimuli_presentation_order_list[s], 0)
	del data_provv
del eye_data_list, stimuli_presentation_order_list

# Select the data split
if args.data_split == 'train':
	idx = np.where(stimuli_presentation_order <= 1000)[0]
elif args.data_split == 'test':
	idx = np.where(stimuli_presentation_order > 1000)[0]
eye_data = eye_data[idx]
stimuli_presentation_order = stimuli_presentation_order[idx]


# =============================================================================
# Time points split selection
# =============================================================================
time_per_split = int(np.ceil(len(times) / 10))
idx_start = time_per_split * (args.time_split - 1)
idx_end = idx_start + time_per_split
eye_data = eye_data[:,idx_start:idx_end]


# =============================================================================
# Eye features selection
# =============================================================================
# Regress out the gaze variance linearly explained by pupil size, while ignoring
# NaN values
if args.features == 'gaze_independent' or args.features == 'all_independent':
	for t in range(eye_data.shape[1]):
		non_nan_idx = ~np.isnan(eye_data[:,t,0])
		# Regress X-gaze
		reg_x = LinearRegression().fit(np.reshape(eye_data[non_nan_idx,t,2], (-1,1)),
			eye_data[non_nan_idx,t,0])
		pred_x = reg_x.predict(np.reshape(eye_data[non_nan_idx,t,2], (-1,1)))
		eye_data[non_nan_idx,t,0] = eye_data[non_nan_idx,t,0] - pred_x
		# Regress Y-gaze
		reg_y = LinearRegression().fit(np.reshape(eye_data[non_nan_idx,t,2], (-1,1)),
			eye_data[non_nan_idx,t,1])
		pred_y = reg_y.predict(np.reshape(eye_data[non_nan_idx,t,2], (-1,1)))
		eye_data[non_nan_idx,t,1] = eye_data[non_nan_idx,t,1] - pred_y

# Select the eye-tracking features of interest
if args.features == 'gaze' or args.features == 'gaze_independent':
	eye_data = eye_data[:,:,0:2]
elif args.features == 'pupil':
	eye_data = eye_data[:,:,2]
elif args.features == 'all' or args.features == 'all_independent':
	eye_data = eye_data


# =============================================================================
# Perform the pairwise decoding
# =============================================================================
# Results array of shape:
# (Video conditions × Video conditions × Eye time points)
video_conditions = np.unique(stimuli_presentation_order)
pairwise_decoding = np.zeros((len(video_conditions), len(video_conditions),
	eye_data.shape[1]))

# Loop over eye time points, image-conditions and trials repetitions
for t in tqdm(range(eye_data.shape[1])):
	for v1 in range(len(video_conditions)):
		for v2 in range(v1):
			# Select the video conditions data
			idx_1 = resample(np.where(
				stimuli_presentation_order == video_conditions[v1])[0],
				replace=False)
			idx_2 = resample(np.where(
				stimuli_presentation_order == video_conditions[v2])[0],
				replace=False)
			eye_cond_1 = eye_data[idx_1,t]
			eye_cond_2 = eye_data[idx_2,t]
			if args.features == 'pupil':
				eye_cond_1 = np.reshape(eye_cond_1, (len(eye_cond_1), 1))
				eye_cond_2 = np.reshape(eye_cond_2, (len(eye_cond_2), 1))
			# Remove NaN trials
			idx_nan_1 = np.where(np.isnan(eye_cond_1[:,0]))[0]
			idx_nan_2 = np.where(np.isnan(eye_cond_2[:,0]))[0]
			eye_cond_1 = np.delete(eye_cond_1, idx_nan_1, 0)
			eye_cond_2 = np.delete(eye_cond_2, idx_nan_2, 0)
			# Select a minimum amount of trials in case repeats are not the
			# same
			if len(eye_cond_1) < len(eye_cond_2):
				eye_cond_2 = eye_cond_2[:len(eye_cond_1)]
			elif len(eye_cond_2) < len(eye_cond_1):
				eye_cond_1 = eye_cond_1[:len(eye_cond_2)]
			# Create pseudo-trials
			if args.pseudo_trials == 1 and len(eye_cond_1) > 2:
				n_ptrials_repeats = 2
				n_pseudo_trials = int(
					np.ceil(len(eye_cond_1) / n_ptrials_repeats))
				pseudo_data_1 = np.zeros((n_pseudo_trials,
					eye_cond_1.shape[1]))
				pseudo_data_2 = np.zeros((n_pseudo_trials,
					eye_cond_2.shape[1]))
				for r in range(n_pseudo_trials):
					idx_start = r * n_ptrials_repeats
					idx_end = idx_start + n_ptrials_repeats
					pseudo_data_1[r] = np.mean(
						eye_cond_1[idx_start:idx_end], 0)
					pseudo_data_2[r] = np.mean(
						eye_cond_2[idx_start:idx_end], 0)
				eye_cond_1 = pseudo_data_1
				eye_cond_2 = pseudo_data_2
			# The classifier needs at least 2 trials to train and test
			if len(eye_cond_1) > 1:
				# SVM target vectors
				y_train = np.zeros(((len(eye_cond_1)-1)*2))
				y_train[int(len(y_train)/2):] = 1
				y_test = np.asarray((0, 1))
				scores = np.zeros(len(eye_cond_1))
				for r in range(len(eye_cond_1)):
					# Define the training/test partitions
					X_train = np.append(np.delete(eye_cond_1, r, 0),
						np.delete(eye_cond_2, r, 0), 0)
					X_test = np.append(np.expand_dims(eye_cond_1[r], 0),
						np.expand_dims(eye_cond_2[r], 0), 0)
					# Train the classifier
					dec_svm = SVC(kernel='linear')
					dec_svm.fit(X_train, y_train)
					# Test the classifier
					y_pred = dec_svm.predict(X_test)
					scores[r] = sum(y_pred == y_test) / len(y_test)
				# Store the accuracy
				pairwise_decoding[v1,v2,t] = np.mean(scores)
				pairwise_decoding[v2,v1,t] = pairwise_decoding[v1,v2,t]
			else:
				# Store NaN values when there are less than 2 trials
				pairwise_decoding[v1,v2,t] = np.nan
				pairwise_decoding[v2,v1,t] = pairwise_decoding[v1,v2,t]


# =============================================================================
# Save the results
# =============================================================================
results_dict = {
	'args': args,
	'pairwise_decoding': pairwise_decoding,
	'times': times
}

save_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eye',
	'pairwise_decoding', 'sub-'+format(args.sub,'02'), 'sfreq-'+
	format(args.sfreq,'04'), 'data_split-'+args.data_split, 'mean_centering-'+
	args.mean_centering, 'features-'+args.features, 'zscore-'+
	format(args.zscore, '02'), 'pseudo_trials-'+format(args.pseudo_trials,'02'))
file_name = 'pairwise_decoding_time_split-' + format(args.time_split, '02')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), results_dict)