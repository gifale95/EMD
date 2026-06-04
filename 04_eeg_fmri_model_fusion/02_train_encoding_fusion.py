"""Train encoding models that map EEG Moments EEG responses onto BOLD Moments
fMRI resposnes, independently for each EEG time point.

Parameters
----------
fmri_subject : int
	Number of used BOLM Moments fMRI subject.
fmri_hemi : str
	Whether to use the 'left' or 'right' fMRI hemisphere.
fmri_split : int
	fMRI split used.
tot_fmri_splits : int
	Total amount of fMRI splits.
eeg_channel_policy : str
	If 'average', the EEG responses are averaged across subjects, and then
	transformed with PCA. If 'append', the EEG responses are appended across
	subjects across the channel dimension, and transformed with PCA.
max_eeg_pcs : int
	Max number of EEG PCs used.
sfreq : int
	EEG downsampling frequency.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)
parser.add_argument('--fmri_hemi', type=str, default='left')
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--tot_fmri_splits', type=int, default=21)
parser.add_argument('--eeg_channel_policy', type=str, default='append')
parser.add_argument('--max_eeg_pcs', type=int, default=500)
parser.add_argument('--sfreq', type=int, default=100)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/science/projects/eeg_moments', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments', type=str)
args = parser.parse_args()

print('>>> EEG-fMRI fusion, train encoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the training EEG responses
# =============================================================================
# Train the EEG-to-fMRI fusion encoding models with the EEG responses from the
# odd repeats
data_dir = os.path.join(args.project_dir, 'results',
	'eeg_fmri_fusion_even_odd_eeg_split', 'eeg_pca',
	'eeg_pca_train_odd_channel_policy-'+args.eeg_channel_policy+'_sfreq-'+
	format(args.sfreq, '04')+'.npy')

eeg_dict = np.load(data_dir, allow_pickle=True).item()
eeg = eeg_dict['eeg_pca'][:,:args.max_eeg_pcs].astype(np.float32)
ch_names = eeg_dict['ch_names']
times = eeg_dict['times']
del eeg_dict


# =============================================================================
# Load the training fMRI responses
# =============================================================================
# The version B fMRI responses from BOLD Moments are already z-scored at each
# scan session
data_dir = os.path.join(args.project_dir, 'bold_moments_dataset', 'derivatives',
	'versionB', 'fsaverage', 'GLM', 'sub-'+format(args.fmri_subject, '02'),
	'prepared_betas')
file_name = 'sub-' + format(args.fmri_subject, '02') + '_organized_betas_' + \
	'task-train_hemi-' + args.fmri_hemi + '_normalized.pkl'

f = open(os.path.join(data_dir, file_name), 'rb')
fmri = pickle.load(f)[0]

# Average the fMRI responses across repeats
fmri = np.mean(fmri, 1).astype(np.float32)

# Select the vertices from the chosen fMRI response split
vertices_per_split = int(np.floor(fmri.shape[1] / args.tot_fmri_splits))
idx_start = (args.fmri_split - 1) * vertices_per_split
idx_end = idx_start + vertices_per_split
fmri = fmri[:,idx_start:idx_end]


# =============================================================================
# Train the fusion encoding models
# =============================================================================
results = {}
results['ch_names'] = ch_names
results['times'] = times
results['reg_param'] = []

for t in tqdm(range(len(times))):

	# Train the linear regression
	reg = LinearRegression().fit(eeg[:,:,t], fmri)

	# Store the regression parameters
	param = {
		'coef_': reg.coef_,
		'intercept_': reg.intercept_,
		'n_features_in_': reg.n_features_in_
		}
	results['reg_param'].append(param)
	del reg, param


# =============================================================================
# Save the regression weights
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results',
	'eeg_fmri_fusion_even_odd_eeg_split', 'regression_weights', 'sfreq-'+
	format(args.sfreq, '04'), 'eeg_channel_policy-'+ args.eeg_channel_policy,
	'fmri_hemi-'+args.fmri_hemi)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'regression_weights_fmri_split-' + format(args.fmri_split, '03') + \
	'.npy'

np.save(os.path.join(save_dir, file_name), results)
