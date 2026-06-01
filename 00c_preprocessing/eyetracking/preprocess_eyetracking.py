"""Preprocess the raw eye-tracking data.

Parameters
----------
subject : int
    Used subject.
tot_sessions : int
    Total EEG recording sessions.
tmin : float
    Start time of the epochs in seconds, relative to stimulus onset.
tmax : float
    End time of the epochs in seconds, relative to stimulus onset.
mean_centering : str
    Whether to mean center the eye-tracking data using the Nms prior to video
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
import h5py

from utils import preprocess_eyetracking


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--tot_sessions', default=8, type=int)
parser.add_argument('--tmin', default=-.1, type=float)
parser.add_argument('--tmax', default=3.5, type=float)
parser.add_argument('--mean_centering', default='baseline', type=str)
parser.add_argument('--sfreq', default=500, type=int)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

# Printing the arguments
print('\n\n\n>>> Preprocess the raw eye-tracking data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Preprocess the eye-tracking data
# =============================================================================
# The preprocessing is applied independently to the data of each session
eyetracking = []
stimulus_id = {}
run_number = {}
trial_number = {}

for ses in range(1, args.tot_sessions+1):
    preprocessed_eyetracking, times, stim_id, run_num, trial_num = \
        preprocess_eyetracking(args, ses)
    eyetracking.append(preprocessed_eyetracking)
    stimulus_id[f'ses-{ses:02}'] = stim_id
    run_number[f'ses-{ses:02}'] = run_num
    trial_number[f'ses-{ses:02}'] = trial_num
    del preprocessed_eyetracking, stim_id, run_num, trial_num


# =============================================================================
# Save the preprocessed eye-tracking data
# =============================================================================
# Save diorectory
save_dir = os.path.join(args.project_dir, 'derivatives', 'eyetracking',
    f'sub-{args.subject:02}')
os.makedirs(save_dir, exist_ok=True)

# Save the eyetracking metadata
del args.project_dir
metadata = {
    'args': args,
    'stimulus_id': stimulus_id,
    'run_number': run_number,
    'trial_number': trial_number,
    'times': times
    }
file_name = f'sub-{args.subject:02}_eyetracking_metadata.npy'
np.save(os.path.join(save_dir, file_name), metadata)

# Save the preprocessed eyetracking data of each session
for ses, session_data in enumerate(eyetracking):
    file_name = f'sub-{args.subject:02}_ses-{ses+1:02}_preprocessed_eyetracking.h5'
    with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
        f.create_dataset('eyetracking', data=session_data, dtype=np.float32)