"""Test encoding models that map EEG Moments EEG responses onto BOLD Moments
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
from scipy.stats import pearsonr


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

print('>>> EEG-fMRI fusion, test encoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the recorded testing fMRI responses
# =============================================================================
# The version B fMRI responses from BOLD Moments are already z-scored at each
# scan session
data_dir = os.path.join(args.project_dir, 'bold_moments_dataset', 'derivatives',
	'versionB', 'fsaverage', 'GLM', 'sub-'+format(args.fmri_subject, '02'),
	'prepared_betas')
file_name = 'sub-' + format(args.fmri_subject, '02') + '_organized_betas_' + \
	'task-test_hemi-' + args.fmri_hemi + '_normalized.pkl'
f = open(os.path.join(data_dir, file_name), 'rb')
fmri = pickle.load(f)[0]

# Average the fMRI responses across repeats
fmri = np.mean(fmri, 1).astype(np.float32)

# Load the noise ceiling
file_name = 'sub-' + format(args.fmri_subject, '02') + \
	'_noiseceiling_space-fsaverage_' + 'task-test_hemi-' + args.fmri_hemi + \
	'_n-10.pkl'
f = open(os.path.join(data_dir, file_name), 'rb')
noise_ceiling = pickle.load(f)[1].astype(np.float32)

# Add a very small number to noise ceiling values of 0, otherwise the
# noise-ceiling-normalized encoding accuracy cannot be calculated (division
# by 0 is not possible)
noise_ceiling[noise_ceiling==0] = 1e-14

# Select the vertices from the chosen fMRI response split
vertices_per_split = int(np.floor(fmri.shape[1] / args.tot_fmri_splits))
idx_start = (args.fmri_split - 1) * vertices_per_split
idx_end = idx_start + vertices_per_split
fmri = fmri[:,idx_start:idx_end]
noise_ceiling = noise_ceiling[idx_start:idx_end]


# =============================================================================
# Load the testing EEG responses
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'eeg_fmri_fusion',
	'eeg_pca', 'eeg_pca_test_channel_policy-'+args.eeg_channel_policy+
	'_sfreq-'+format(args.sfreq, '04')+'.npy')

eeg_dict = np.load(data_dir, allow_pickle=True).item()
eeg = eeg_dict['eeg_pca'][:,:args.max_eeg_pcs]
ch_names = eeg_dict['ch_names']
times = eeg_dict['times']
del eeg_dict


# =============================================================================
# Load the trained EEG-fMRI fusion models, use them to generate fMRI responses
# for the testing videos, and compute the encoding accuracy
# =============================================================================
# Empty results arrays
correlation = np.zeros((fmri.shape[1], len(times)), dtype=np.float32)
explained_variance = np.zeros((fmri.shape[1], len(times)), dtype=np.float32)

# Load the trained encoding model weights
data_dir = os.path.join(args.project_dir, 'results', 'eeg_fmri_fusion',
	'regression_weights', 'sfreq-'+format(args.sfreq, '04'),
	'eeg_channel_policy-'+args.eeg_channel_policy, 'fmri_hemi-'+
	args.fmri_hemi, 'regression_weights_fmri_split-'+
	format(args.fmri_split, '03')+'.npy')
regression_weights = np.load(data_dir, allow_pickle=True).item()

# Loop over EEG time points
for t in tqdm(range(len(times))):

	# Create the linear regression object
	reg = LinearRegression()
	reg.coef_ = regression_weights['reg_param'][t]['coef_']
	reg.intercept_ = regression_weights['reg_param'][t]['intercept_']
	reg.n_features_in_ = regression_weights['reg_param'][t]['n_features_in_']

	# Generate the fMRI responses for the test videos
	pred_fmri = reg.predict(eeg[:,:,t]).astype(np.float32)

	# Correlate predicted and recorded fMRI responses
	corr = np.zeros((fmri.shape[1]), dtype=np.float32)
	for v in range(fmri.shape[1]):
		corr[v] = pearsonr(pred_fmri[:,v], fmri[:,v])[0]
	correlation[:,t] = corr

	# Set negative correlation scores to zero
	corr[corr<0] = 0

	# Turn the correlations into r2 scores
	r2 = corr ** 2

	# Compute the noise-ceiling-normalized encoding accuracy
	explained_variance[:,t] = np.divide(r2, noise_ceiling) * 100

# Set the noise-ceiling-normalized encoding accuracy to 100 for vertices where
# the the correlation is higher than the noise ceiling, to prevent encoding
# accuracy values higher than 100%
explained_variance[explained_variance>100] = 100


# =============================================================================
# Save the results
# =============================================================================
encoding_accuracy = {
	'correlation': correlation,
	'noise_ceiling': noise_ceiling,
	'explained_variance': explained_variance,
	'ch_names': ch_names,
	'times': times
	}

save_dir = os.path.join(args.project_dir, 'results', 'eeg_fmri_fusion',
	'encoding_accuracy', 'sfreq-'+format(args.sfreq, '04'),
	'eeg_channel_policy-'+args.eeg_channel_policy, 'fmri_hemi-'+args.fmri_hemi)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'encoding_accuracy_fmri_split-' + format(args.fmri_split, '03') + \
	'.npy'

np.save(os.path.join(save_dir, file_name), encoding_accuracy)
