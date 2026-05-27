"""Preprocess the source EEG data: filtering, epoching, frequency downsampling,
baseline correction, multivariate noise normalization (MVNN), and NCSNR
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
from utils import epoch_eeg
from utils import mvnn
from utils import compute_ncsnr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=6, type=int)
parser.add_argument('--tot_sessions', default=8, type=int)
parser.add_argument('--tmin', default=-.2, type=float)
parser.add_argument('--tmax', default=3.5, type=float)
parser.add_argument('--baseline_correction', default=1, type=int)
parser.add_argument('--baseline', default=(-.2, 0), type=tuple)
parser.add_argument('--sfreq', default=500, type=int)
parser.add_argument('--lowpass', default=40, type=float)
parser.add_argument('--highpass', default=0.1, type=float)
parser.add_argument('--mvnn', default='none', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

# Printing the arguments
print('\n\n\n>>> Preprocess the source EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Filter, epoch, downsample and baseline correct the EEG data
# =============================================================================
# This step is applied independently to the data of each session
preprocessed_eeg = []
stimulus_presentation_order = []
beh_response = []
beh_correctness = []

for s in range(args.tot_sessions):
    epoched_data, stimulus_order, ch_names, times, info, beh_r, beh_c = \
        epoch_eeg(args, s)
    stimulus_presentation_order.append(stimulus_order)
    beh_response.append(beh_r)
    beh_correctness.append(beh_c)
    del beh_r, beh_c


# =============================================================================
# Multivariate noise normalization
# =============================================================================
    # MVNN is applied independently to the data of each session
    preprocessed_eeg.append(mvnn(args, epoched_data, stimulus_order, s))
    del epoched_data, stimulus_order


# =============================================================================
# Compute the NCSNR and noise ceiling using the test data split
# =============================================================================
ncsnr, noise_ceiling = compute_ncsnr(preprocessed_eeg,
    stimulus_presentation_order)


# =============================================================================
# Save the preprocessed EEG data
# =============================================================================
data_dict = {
    'args': args,
    'eeg': preprocessed_eeg,
    'stimulus_presentation_order': stimulus_presentation_order,
    'ch_names': ch_names,
    'times': times,
    'ncsnr': ncsnr,
    'noise_ceiling': noise_ceiling,
    'info': info,
    'beh': {
        'response': beh_response,
        'correctness': beh_correctness
    }
}

save_dir = os.path.join(args.project_dir, 'dataset', 'preprocessed_data',
    'eeg')
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, f'preprocessed_eeg_sub-{args.sub:02}.npy'),
    data_dict)
