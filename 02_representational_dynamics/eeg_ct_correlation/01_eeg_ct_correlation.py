"""Run a cross-temporal correlation analysis on the EEG responses for the 102
test videos, using the occipital and parietal channels.

Parameters
----------
subject : int
    Used subject.
n_iter : int
    Number of iterations for the pseudo-trial creation.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).

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
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
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
data_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg',
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
    eeg_pseudo = np.nan_to_num(eeg_pseudo, nan=0).astype(np.float32)


# =============================================================================
# Correlate each time point combination of the EEG responses of the two
# pseudo-trials, across video conditions and channels
# =============================================================================
    # Z-score across channels
    pseudo_1 = np.reshape(eeg_pseudo[:,0], (-1, eeg_pseudo.shape[3]))
    pseudo_2 = np.reshape(eeg_pseudo[:,1], (-1, eeg_pseudo.shape[3]))
    pseudo_1z = (pseudo_1 - pseudo_1.mean(axis=0)) / pseudo_1.std(axis=0)
    pseudo_2z = (pseudo_2 - pseudo_2.mean(axis=0)) / pseudo_2.std(axis=0)
    del eeg_pseudo, pseudo_1, pseudo_2

    # Cross-temporal correlation: shape (EEG time points, EEG time points)
    if i == 0:
        ct_corr = (pseudo_1z.T @ pseudo_2z / pseudo_1z.shape[0]).astype(np.float32)
    else:
        ct_corr += (pseudo_1z.T @ pseudo_2z / pseudo_1z.shape[0]).astype(np.float32)
    del pseudo_1z, pseudo_2z

# Average the results across iterations
ct_corr = ct_corr / args.n_iter


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.emd_dir, 'results',
    'representational_dynamics', 'eeg_ct_correlation', 'ct_correlation')
os.makedirs(save_dir, exist_ok=True)

file_name = f'ct_correlation_sub-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), ct_corr)