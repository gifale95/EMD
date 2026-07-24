"""Create correlation-based EEG RDMs for the 102 test videos, using the
occipital EEG channels. The EEG responses are averaged into 32 time bins
between 0-3 epoch seconds.

Parameters
----------
subject : int
    Used subject.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).

"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import zscore
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Correlation EEG RDMs <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the EEG responses for the 102 test videos
# =============================================================================
# Load the stimulus IDs
data_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg',
    f'sub-{args.subject:02}')
file_name = f'sub-{args.subject:02}_eeg_metadata.npy'
metadata = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
stimulus_id = metadata['stimulus_id']
ch_names = metadata['ch_names']
times = metadata['times']

# Load the EEG responses
eeg = []
stimulus_id_test = []
n_sessions = 8
# Loop across EEG recording sessions
for ses in tqdm(range(1, n_sessions+1)):
    # Load the EEG responses
    file_name = f'sub-{args.subject:02}_ses-{ses:02}_preprocessed_eeg.h5'
    eeg_ses = h5py.File(os.path.join(data_dir, file_name), 'r')['eeg'][:]
    # Z-score the EEG responses at each session
    eeg_ses = zscore(eeg_ses, axis=0, nan_policy='omit')
    # Select the responses for the test videos
    idx = np.where(stimulus_id[f'ses-{ses:02}'] > 1000)[0]
    eeg.append(eeg_ses[idx])
    stimulus_id_test.append(stimulus_id[f'ses-{ses:02}'][idx])
    del eeg_ses, idx
eeg = np.concatenate(eeg, 0)
stimulus_id_test = np.concatenate(stimulus_id_test, 0)


# =============================================================================
# Only keep occipital and parietal channels
# =============================================================================
idx_ch = []
for c, chan in enumerate(ch_names):
    if 'O' in chan or 'P' in chan:
        idx_ch.append(c)
idx_ch = np.array(idx_ch)
eeg = eeg[:,idx_ch]


# =============================================================================
# Average the EEG repeats of each video condition
# =============================================================================
video_conditions = np.unique(stimulus_id_test)

eeg_avg = np.zeros((len(video_conditions), eeg.shape[1], eeg.shape[2]),
    dtype=np.float32)
for v, video in enumerate(video_conditions):
    idx = np.where(stimulus_id_test == video)[0]
    eeg_avg[v] = np.nanmean(eeg[idx], 0)
del eeg


# =============================================================================
# Average the EEG resposnes into 32 time bins between 0-3 epoch seconds
# =============================================================================
# Only select EEG time points from 0 to 3 seconds
min_times = 0
max_times = 3
idx_start = np.where(times >= min_times)[0][0]
idx_end = np.where(times <= max_times)[0][-1]
eeg_avg = eeg_avg[:,:,idx_start:idx_end+1]
times = times[idx_start:idx_end+1]

# Average the EEG responses into 32 time bins, so as to match the time bins of
# the AlexNet stimulus features (for the first bin only use EEG time points
# from 60ms after simulus onset, to account for the visual processing delay)
n_bins = 32
eeg_splits = np.array_split(eeg_avg, n_bins, 2)
for s, split in enumerate(eeg_splits):
    if s == 0:
        idx_start = np.where(times >= 0.06)[0][0]
        eeg_bin = np.nanmean(split[:,:,idx_start:], 2, keepdims=True)
    else:
        eeg_bin = np.append(eeg_bin, np.nanmean(split, 2, keepdims=True), 2)
del eeg_avg

# Set NaN values to 0
eeg_bin = np.nan_to_num(eeg_bin, nan=0)


# =============================================================================
# Create the RDMs for each EEG time point, using Pearson's correlation
# =============================================================================
# Z-score across channels
eeg_mean = eeg_bin.mean(axis=1, keepdims=True)
eeg_std = eeg_bin.std(axis=1, keepdims=True)
eeg_z = (eeg_bin - eeg_mean) / (eeg_std + 1e-8)

# Reshape to (Times, Videos, Channels)
eeg_z = np.transpose(eeg_z, (2, 0, 1))

# Cross-correlation via batch matmul
rdms = (np.matmul(eeg_z, eeg_z.transpose(0, 2, 1)) / \
        (eeg_bin.shape[1])).astype(np.float32)
rdms = 1 - rdms

# Back to (Videos_X, Videos_Y, Times)
rdms = np.transpose(rdms, (1, 2, 0))


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.emd_dir, 'results', 'representational_dynamics',
    'eeg_rdms')
os.makedirs(save_dir, exist_ok=True)

file_name = f'correlation_rdms_sub-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), rdms)