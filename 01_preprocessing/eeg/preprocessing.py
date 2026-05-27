"""Preprocess the source EEG data: filtering, epoching, frequency downsampling,
baseline correction, multivariate noise normalization (MVNN), and noise ceiling
calculation.

Parameters
----------
sub : int
	Used subject.
tot_sessions : int
	Total EEG recording sessions.
tmin : float
	Start time of the epochs in seconds, relative to stimulus onset.
tmax : float
	End time of the epochs in seconds, relative to stimulus onset.
baseline_correction : int
	Whether to baseline correct [1] or not [0] the data.
baseline : tuple
	Epoch time range used for computing the baseline correction.
sfreq : int
	Downsampling frequency.
lowpass : float
	Lowpass filter frequency.
highpass : float
	Highpass filter frequency.
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
from preprocessing_utils import epoch_eeg
from preprocessing_utils import mvnn
from preprocessing_utils import compute_noise_ceiling


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=6, type=int)
parser.add_argument('--tot_sessions', default=8, type=int)
parser.add_argument('--tmin', default=-.2, type=float)
parser.add_argument('--tmax', default=3.5, type=float)
parser.add_argument('--baseline_correction', default=1, type=int)
parser.add_argument('--baseline', default=(-.2,0), type=tuple)
parser.add_argument('--sfreq', default=100, type=int)
parser.add_argument('--lowpass', default=40, type=float)
parser.add_argument('--highpass', default=0.1, type=float)
parser.add_argument('--mvnn', default='none', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> Preprocess the source EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Filter, epoch, downsample and baseline correct the EEG data
# =============================================================================
# This step is applied independently to the data of each session
preprocessed_data = []
stimuli_presentation_order = []
beh_correctness = []
beh_response = []

for s in range(args.tot_sessions):
	epoched_data, stimuli_order, ch_names, times, info, beh_r, beh_c = \
		epoch_eeg(args, s)
	stimuli_presentation_order.append(stimuli_order)
	beh_response.append(beh_r)
	beh_correctness.append(beh_c)
	del beh_r, beh_c


# =============================================================================
# Multivariate Noise Normalization
# =============================================================================
	# MVNN is applied independently to the data of each session
	white_data = mvnn(args, epoched_data, stimuli_order, times, s)
	preprocessed_data.append(white_data)
	del white_data, epoched_data, stimuli_order


# =============================================================================
# Compute the noise ceiling using the test data split (as in the NSD paper)
# =============================================================================
noise_ceiling = compute_noise_ceiling(preprocessed_data,
	stimuli_presentation_order)


# =============================================================================
# Save the preprocessed EEG data
# =============================================================================
data_dict = {
	'args': args,
	'eeg_data': preprocessed_data,
	'stimuli_presentation_order': stimuli_presentation_order,
	'behavioral_response': beh_response,
	'behavioral_correctness': beh_correctness,
	'ch_names': ch_names,
	'times': times,
	'noise_ceiling': noise_ceiling,
	'info': info,
	}

save_dir = os.path.join(args.project_dir, 'dataset', 'preprocessed_data',
	'dataset_02', 'eeg', 'sub-'+format(args.sub,'02'), 'mvnn-'+args.mvnn,
	'baseline_correction-'+format(args.baseline_correction, '02'), 'highpass-'+
	format(args.highpass)+'_lowpass-'+format(args.lowpass), 'sfreq-'+
	format(args.sfreq,'04'))
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, 'preprocessed_data.npy'), data_dict)
