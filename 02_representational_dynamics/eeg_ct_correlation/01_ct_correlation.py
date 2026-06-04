"""Run a cross-temporal correlation analysis on the EEG responses for the 102
test videos, using the occipital and parietal channels.

Parameters
----------
subject : int
    Used subject.
n_iter : int
    Number of iterations for the pseudo-trial creation.
project_dir : str
    Directory of the project folder.

"""

import argparse
import os
import numpy as np
import h5py
from scipy.stats import zscore
from tqdm import tqdm
from sklearn.utils import resample


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--n_iter', default=10000, type=int)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Cross-temporal RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the EEG responses for the 102 test videos
# =============================================================================
# Load the stimulus IDs
data_dir = os.path.join(args.project_dir, 'derivatives', 'eeg',
    f'sub-{args.subject:02}')
file_name = f'sub-{args.subject:02}_eeg_metadata.npy'
metadata = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
stimulus_id = metadata['stimulus_id']
ch_names = metadata['ch_names']

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
# Average the EEG repeats of each video condition into 2 pseudo-trials
# =============================================================================
video_conditions = np.unique(stimulus_id_test)
n_pseudo_trl = 2

# Loop acoss analysis iterations
ct_corr = []
for i in tqdm(range(args.n_iter)):

    # Create the pseudo-trials by averaging across repeats of each video
    # condition
    eeg_pseudo = np.zeros((len(video_conditions), n_pseudo_trl, eeg.shape[1],
        eeg.shape[2]), dtype=np.float32)
    for v, video in enumerate(video_conditions):
        idx = resample(np.where(stimulus_id_test == video)[0], replace=False)
        n_trl_per_pseudo = int(np.ceil(len(idx) / n_pseudo_trl))
        for p in range(n_pseudo_trl):
            idx_start = p * n_trl_per_pseudo
            idx_end = idx_start + n_trl_per_pseudo
            eeg_pseudo[v,p] = np.nanmean(eeg[idx[idx_start:idx_end]], 0)

# Set NaN values to 0
eeg_pseudo = np.nan_to_num(eeg_pseudo, nan=0)


# =============================================================================
# Correlate each time point combination of the EEG responses of the two
# pseudo-trials, across video conditions and channels
# =============================================================================
    pseudo_1 = np.reshape(eeg_pseudo[:,0], (-1, eeg_pseudo.shape[3]))
    pseudo_2 = np.reshape(eeg_pseudo[:,1], (-1, eeg_pseudo.shape[3]))
    pseudo_1z = (pseudo_1 - pseudo_1.mean(axis=0)) / pseudo_1.std(axis=0)
    pseudo_2z = (pseudo_2 - pseudo_2.mean(axis=0)) / pseudo_2.std(axis=0)

    # Cross-temporal RSA: shape (EEG time points, EEG time points)
    ct_corr.append(pseudo_1z.T @ pseudo_2z / pseudo_1.shape[0])

# Average the results across iterations
ct_corr = np.mean(ct_corr, 0)


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results',
    'representational_dynamics', 'eeg_ct_correlation', 'ct_correlation')
os.makedirs(save_dir, exist_ok=True)

file_name = f'ct_correlation_sub-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), ct_corr)


# =============================================================================
# Create the RDMs for each EEG time point, using Pearson's correlation
# =============================================================================
# # Get the responses for the two repetition splits
# X = eeg_pseudo[:,0]
# Y = eeg_pseudo[:,1]

# # Z-score across channels
# X_mean = X.mean(axis=1, keepdims=True)
# Y_mean = Y.mean(axis=1, keepdims=True)
# X_std = X.std(axis=1, keepdims=True)
# Y_std = Y.std(axis=1, keepdims=True)
# X_z = (X - X_mean) / (X_std + 1e-8)
# Y_z = (Y - Y_mean) / (Y_std + 1e-8)

# # Reshape to (Times, Videos, Channels)
# X_t = np.transpose(X_z, (2, 0, 1))
# Y_t = np.transpose(Y_z, (2, 0, 1))

# # Cross-correlation via batch matmul
# rdms_1 = (np.matmul(X_t, X_t.transpose(0, 2, 1)) / (X.shape[1])).astype(np.float32)
# rdms_2= (np.matmul(Y_t, Y_t.transpose(0, 2, 1)) / (Y.shape[1])).astype(np.float32)
# rdms_1 = 1 - rdms_1
# rdms_2 = 1 - rdms_2

# # Back to (Videos_X, Videos_Y, Times)
# rdms_1 = np.transpose(rdms_1, (1, 2, 0))
# rdms_2 = np.transpose(rdms_2, (1, 2, 0))

# # Take the lower triangle of the RDMs
# tril_idx = np.tril_indices(rdms_1.shape[0], k=-1)
# rdms_1 = rdms_1[tril_idx]
# rdms_2 = rdms_2[tril_idx]


# =============================================================================
# Cross-temporal RSA
# =============================================================================
# # Center and scale the RDM entries at each time point
# rdms_1z = (rdms_1 - rdms_1.mean(axis=0)) / rdms_1.std(axis=0)
# rdms_2z = (rdms_2 - rdms_2.mean(axis=0)) / rdms_2.std(axis=0)

# # Cross-temporal RSA: shape (EEG time points, EEG time points)
# ct_rsa = rdms_1z.T @ rdms_2z / rdms_1.shape[0]