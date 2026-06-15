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
baseline_correction : str
    Whether to baseline correct the pupil data using the baseline prior to
    video onset ['baseline'], the entire epoch ['epoch'], or to not correct the
    data at all ['none'].
sfreq : int
    Downsampling frequency.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).

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
parser.add_argument('--subject', default=6, type=int)
parser.add_argument('--tot_sessions', default=8, type=int)
parser.add_argument('--tmin', default=-.1, type=float)
parser.add_argument('--tmax', default=3.5, type=float)
parser.add_argument('--baseline_correction', default='baseline', type=str)
parser.add_argument('--sfreq', default=500, type=int)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
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
save_dir = os.path.join(args.emd_dir, 'derivatives', 'eyetracking',
    f'sub-{args.subject:02}')
os.makedirs(save_dir, exist_ok=True)

# Save the eyetracking metadata
del args.emd_dir
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