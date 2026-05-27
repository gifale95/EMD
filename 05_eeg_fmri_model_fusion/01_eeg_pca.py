"""Aggregate the EEG Moments EEG resposnes across subjects, and transform them
with PCA.

Parameters
----------
eeg_subjects : list
	List of all EEG Moments subject.
eeg_channel_policy : str
	If 'average', the EEG responses are averaged across subjects, and then
	transformed with PCA. If 'append', the EEG responses are appended across
	subjects across the channel dimension, and transformed with PCA.
sfreq : int
	EEG downsampling frequency.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--eeg_subjects', type=list, default=[1, 2, 3, 4, 5, 6])
parser.add_argument('--eeg_channel_policy', type=str, default='average')
parser.add_argument('--sfreq', type=int, default=100)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/science/projects/eeg_moments', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments', type=str)
args = parser.parse_args()

print('>>> EEG PCA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the preprocessed EEG responses
# =============================================================================
eeg_train_all_odd = []
eeg_train_all_even = []
eeg_test_all = []

for s, sub in enumerate(args.eeg_subjects):

	data_dir = os.path.join(args.project_dir, 'dataset', 'preprocessed_data',
		'eeg', 'sub-0'+str(sub), 'mvnn-time', 'baseline_correction-01',
		'highpass-0.01_lowpass-100', 'sfreq-'+format(args.sfreq, '04'),
		'preprocessed_data.npy')
	data_dict = np.load(data_dir, allow_pickle=True).item()
	times = data_dict['times']
	ch_names = data_dict['ch_names']
	eeg_data_list = data_dict['eeg_data']
	stimuli_presentation_order_list = data_dict['stimuli_presentation_order']
	del data_dict

	# Z-score and append the data of each scan session
	for ses in range(len(eeg_data_list)):
		if ses == 0:
			eeg_data = zscore(eeg_data_list[ses], 0)
			stimuli_presentation_order = stimuli_presentation_order_list[ses]
		else:
			eeg_data = np.append(eeg_data, zscore(eeg_data_list[ses], 0), 0)
			stimuli_presentation_order = np.append(stimuli_presentation_order,
				stimuli_presentation_order_list[ses], 0)
	del eeg_data_list, stimuli_presentation_order_list

	# Extract the training data, and average across repeats independently for
	# even and odd trials
	idx_train = np.arange(1, 1001)
	train_data_sub_odd = np.zeros((len(idx_train), len(ch_names), len(times)),
		dtype=np.float32)
	train_data_sub_even = np.zeros((len(idx_train), len(ch_names), len(times)),
		dtype=np.float32)
	for v, video in enumerate(idx_train):
		idx = np.where(stimuli_presentation_order == video)[0]
		idx_odd = idx[np.arange(0, len(idx), 2)]
		idx_even = idx[np.arange(1, len(idx), 2)]
		train_data_sub_odd[v] = np.mean(eeg_data[idx_odd], 0)
		train_data_sub_even[v] = np.mean(eeg_data[idx_even], 0)

	# Extract the test data, and average across repeats
	idx_test = np.arange(1001, 1103)
	test_data_sub = np.zeros((len(idx_test), len(ch_names), len(times)),
		dtype=np.float32)
	for v, video in enumerate(idx_test):
		idx = np.where(stimuli_presentation_order == video)[0]
		test_data_sub[v] = np.mean(eeg_data[idx], 0)
	del eeg_data, stimuli_presentation_order

	# Append the EEG responses across subjects
	eeg_train_all_odd.append(train_data_sub_odd)
	eeg_train_all_even.append(train_data_sub_even)
	eeg_test_all.append(test_data_sub)
	del train_data_sub_odd, train_data_sub_even, test_data_sub


# =============================================================================
# Channel policy
# =============================================================================
if args.eeg_channel_policy == 'average':

	eeg_train_odd = np.mean(eeg_train_all_odd, 0)
	eeg_train_even = np.mean(eeg_train_all_even, 0)
	eeg_test = np.mean(eeg_test_all, 0)

elif args.eeg_channel_policy == 'append':

	for s in range(len(args.eeg_subjects)):
		if s == 0:
			eeg_train_odd = eeg_train_all_odd[s]
			eeg_train_even = eeg_train_all_even[s]
			eeg_test = eeg_test_all[s]
		else:
			eeg_train_odd = np.append(eeg_train_odd, eeg_train_all_odd[s], 1)
			eeg_train_even = np.append(eeg_train_even, eeg_train_all_even[s], 1)
			eeg_test = np.append(eeg_test, eeg_test_all[s], 1)

del eeg_train_all_odd, eeg_train_all_even, eeg_test_all


# =============================================================================
# Transform the EEG responses with PCA, at each time point
# =============================================================================
eeg_pca_train_odd = np.zeros((eeg_train_odd.shape), dtype=np.float32)
eeg_pca_train_even = np.zeros((eeg_train_even.shape), dtype=np.float32)
eeg_pca_test = np.zeros((eeg_test.shape), dtype=np.float32)

# Loop across EEG time points
for t in range(len(times)):

	# Reshape the EEG resposnes to (Samples x Features)
	eeg_train_odd_t = np.reshape(eeg_train_odd[:,:,t], (len(eeg_train_odd), -1))
	eeg_train_even_t = np.reshape(eeg_train_even[:,:,t],
		(len(eeg_train_even), -1))
	eeg_test_t = np.reshape(eeg_test[:,:,t], (len(eeg_test), -1))

	# Z-score the EEG responses
	scaler = StandardScaler()
	scaler.fit(eeg_train_odd_t)
	eeg_train_odd_t = scaler.transform(eeg_train_odd_t)
	eeg_train_even_t = scaler.transform(eeg_train_even_t)
	eeg_test_t = scaler.transform(eeg_test_t)

	# Apply PCA to the EEG responses
	pca = PCA(random_state=seed)
	pca.fit(eeg_train_odd_t)
	eeg_train_odd_t = pca.transform(eeg_train_odd_t)
	eeg_train_even_t = pca.transform(eeg_train_even_t)
	eeg_test_t = pca.transform(eeg_test_t)

	# Store the PCA-transformed EEG responses
	eeg_pca_train_odd[:,:,t] = eeg_train_odd_t
	eeg_pca_train_even[:,:,t] = eeg_train_even_t
	eeg_pca_test[:,:,t] = eeg_test_t
	del eeg_train_odd_t, eeg_train_even_t, eeg_test_t
del eeg_train_odd, eeg_train_even, eeg_test


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results',
	'eeg_fmri_fusion_even_odd_eeg_split', 'eeg_pca')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name_train_odd = 'eeg_pca_train_odd_channel_policy-' + \
	args.eeg_channel_policy + '_sfreq-' + format(args.sfreq, '04') + '.npy'
file_name_train_even = 'eeg_pca_train_even_channel_policy-' + \
	args.eeg_channel_policy + '_sfreq-' + format(args.sfreq, '04') + '.npy'
file_name_test = 'eeg_pca_test_channel_policy-' + args.eeg_channel_policy + \
	'_sfreq-' + format(args.sfreq, '04') + '.npy'

eeg_pca_train_odd_dict = {
	'eeg_pca': eeg_pca_train_odd,
	'times': times,
	'ch_names': ch_names
	}

eeg_pca_train_even_dict = {
	'eeg_pca': eeg_pca_train_even,
	'times': times,
	'ch_names': ch_names
	}

eeg_pca_test_dict = {
	'eeg_pca': eeg_pca_test,
	'times': times,
	'ch_names': ch_names
	}

np.save(os.path.join(save_dir, file_name_train_odd), eeg_pca_train_odd_dict)
np.save(os.path.join(save_dir, file_name_train_even), eeg_pca_train_even_dict)
np.save(os.path.join(save_dir, file_name_test), eeg_pca_test_dict)
