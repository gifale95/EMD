"""Preprocess the eye-tracking data.

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
mean_centering : str
	Whether to mean center the eye-tracking data using the 500ms prior to video
	onset ['baseline'], the entire epoch ['epoch'], or to not center the data
	al all ['none'].
sfreq : int
	Downsampling frequency.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from preprocessing_utils import get_behavior
from preprocessing_utils import epoch_eyetrack


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=6, type=int)
parser.add_argument('--tot_sessions', default=8, type=int)
parser.add_argument('--tmin', default=-.2, type=float)
parser.add_argument('--tmax', default=3.5, type=float)
parser.add_argument('--mean_centering', default='baseline', type=str)
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
#parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> Eye-traking preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Filter, epoch, downsample and baseline correct the eye-tracking data
# =============================================================================
# This step is applied independently to the data of each session
stimuli_presentation_order = []
beh_response = []
beh_correctness = []
eyetrack_data = []

for s in range(args.tot_sessions):
	epoched_eye_data, eyetrack_times = epoch_eyetrack(args, s)
	stimuli_order, beh_r, beh_c = get_behavior(args, s)
	stimuli_presentation_order.append(stimuli_order)
	beh_response.append(beh_r)
	beh_correctness.append(beh_c)
	epoched_eye_data, eyetrack_times = epoch_eyetrack(args, s)
	eyetrack_data.append(epoched_eye_data)
	del beh_r, beh_c


# =============================================================================
# Save the preprocessed eye-tracking data
# =============================================================================
data_dict = {
	'args': args,
	'eyetrack_data': eyetrack_data,
	'stimuli_presentation_order': stimuli_presentation_order,
	'behavioral_response': beh_response,
	'behavioral_correctness': beh_correctness,
	'eyetrack_times': eyetrack_times
	}

save_dir = os.path.join(args.project_dir, 'dataset', 'preprocessed_data',
	'dataset_02', 'eye', 'sub-'+format(args.sub,'02'), 'mean_centering-'+
	args.mean_centering, 'sfreq-'+format(args.sfreq,'04'))
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, 'preprocessed_data.npy'), data_dict)