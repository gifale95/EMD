"""Preprocess the raw EEG data: filtering, eyeblink removal, epoching, bad
epoch/channel rejection and interpolation, baseline correction, and frequency
downsampling. Additionally, the code computes the NCSNR and noise ceiling on
the EEG responses for the test videos.

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
project_dir : str
    Directory of the project folder.

"""

import argparse
import os
import numpy as np
import h5py

from utils import preprocess_eeg
from utils import compute_ncsnr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--tot_sessions', default=8, type=int)
parser.add_argument('--tmin', default=-.1, type=float)
parser.add_argument('--tmax', default=3.5, type=float)
parser.add_argument('--baseline_correction', default=1, type=int)
parser.add_argument('--baseline', default=(-.2, 0), type=tuple)
parser.add_argument('--sfreq', default=500, type=int)
parser.add_argument('--lowpass', default=40, type=float)
parser.add_argument('--highpass', default=0.1, type=float)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('\n\n\n>>> Preprocess the raw EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Filter, epoch, baseline correct, and downsample the EEG data
# =============================================================================
# This step is applied independently to the data of each session
preprocessed_eeg = []
stimulus_id_list = []
stimulus_id = {}
run_number = {}
trial_number = {}

for ses in range(1, args.tot_sessions+1):
    eeg, stim_id, run_num, trial_num, ch_names, times = \
        preprocess_eeg(args, ses)
    preprocessed_eeg.append(eeg)
    stimulus_id_list.append(stim_id)
    stimulus_id[f'ses-{ses:02}'] = stim_id
    run_number[f'ses-{ses:02}'] = run_num
    trial_number[f'ses-{ses:02}'] = trial_num
    del eeg, stim_id, run_num, trial_num


# =============================================================================
# Compute the NCSNR and noise ceiling using the test data split
# =============================================================================
ncsnr, noise_ceiling = compute_ncsnr(preprocessed_eeg, stimulus_id_list)


# =============================================================================
# Save the preprocessed EEG data and metadata
# =============================================================================
# Save directory
save_dir = os.path.join(args.project_dir, 'derivatives', 'eeg_eyeblink_removal',
    f'sub-{args.subject:02}')
os.makedirs(save_dir, exist_ok=True)

# Save the EEG metadata
del args.project_dir
metadata = {
    'args': args,
    'stimulus_id': stimulus_id,
    'run_number': run_number,
    'trial_number': trial_number,
    'ch_names': ch_names,
    'times': times,
    'ncsnr': ncsnr,
    'noise_ceiling': noise_ceiling
}
file_name = f'sub-{args.subject:02}_eeg_metadata.npy'
np.save(os.path.join(save_dir, file_name), metadata)

# Save the preprocessed EEG data of each session
for ses, session_data in enumerate(preprocessed_eeg):
    file_name = f'sub-{args.subject:02}_ses-{ses+1:02}_preprocessed_eeg.h5'
    with h5py.File(os.path.join(save_dir, file_name), 'w') as f:
        f.create_dataset('eeg', data=session_data, dtype=np.float32)