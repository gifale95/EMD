"""Preprocess the raw EEG data:
	- filtering,
	- epoching,
	- current source density transform,
	- frequency downsampling,
	- baseline correction,
	- zscoring of each recording sessions.

Parameters
----------
sub : int
	Used subject.
n_ses : int
	Number of EEG sessions.
lowpass : float
	Lowpass filter frequency.
highpass : float
	Highpass filter frequency.
tmin : float
	Start time of the epochs in seconds, relative to stimulus onset.
tmax : float
	End time of the epochs in seconds, relative to stimulus onset.
baseline_correction : int
	Whether to baseline correct [1] or not [0] the data.
baseline_mode : str
	Whether to apply 'mean' or 'zscore' baseline correction mode.
csd : int
	Whether to transform the data into current source density [1] or not [0].
sfreq : int
	Downsampling frequency.
mvnn : str
	Whether to compute the MVNN covariace matrices for each time point
	('time') or for each epoch/repetition ('epochs').
mvnn : str
	Whether to compute the MVNN covariace matrices for each epoch time point
	('time'), baseline time point ('baseline'), epoch/repetition ('epochs').
	If 'none', MVNN is not applied.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from preprocessing_utils import epoching
from preprocessing_utils import zscore
from preprocessing_utils import compute_ncsnr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--n_ses', default=8, type=int)
parser.add_argument('--lowpass', default=100, type=float)
parser.add_argument('--highpass', default=0.03, type=float)
parser.add_argument('--tmin', default=-.2, type=float)
parser.add_argument('--tmax', default=3.5, type=float)
parser.add_argument('--baseline_correction', default=1, type=int)
parser.add_argument('--baseline_mode', default='zscore', type=str)
parser.add_argument('--csd', default=1, type=int)
parser.add_argument('--sfreq', default=50, type=int)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
#parser.add_argument('--project_dir', default='/home/ale/scratch/projects/eeg_videos', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()

# Print the arguments
print('\n\n\n>>> Preprocess the source EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Filter, epoch, downsample and baseline correct the EEG data
# =============================================================================
eeg_data, stimuli_presentation_order, eeg_info, times, behavior = epoching(args)


# =============================================================================
# z-scorings
# =============================================================================
# z-scoring is applied independently to the data of each session.
eeg_data_zscored = zscore(eeg_data)
del eeg_data


# =============================================================================
# Compute the noise ceiling SNR using the test data split (as in the NSD paper)
# =============================================================================
#ncsnr = compute_ncsnr(eeg_data_zscored) # !!!


# =============================================================================
# Save the preprocessed EEG data
# =============================================================================
data_dict = {
	'args': args,
	'eeg_data': eeg_data_zscored,
	'stimuli_presentation_order': stimuli_presentation_order,
	'ch_names': eeg_info[0]['ch_names'],
	'times': times,
#	'ncsnr': ncsnr,
	'eeg_info': eeg_info,
	'behavior': behavior
	}

save_dir = os.path.join(args.project_dir, 'dataset', 'preprocessed_data',
	'dataset_02', 'eeg', 'sub-'+format(args.sub,'02'), 'sfreq-'+
	format(args.sfreq,'04'))

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

np.save(os.path.join(save_dir, 'preprocessed_data.npy'), data_dict)
