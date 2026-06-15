"""Create decoding-based EEG RDMs for the 102 test videos, using the occipital
and parietal EEG channels.

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
from sklearn.utils import resample
from sklearn.svm import SVC


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Decoding EEG RDMs <<<')
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
# Channel selection
# =============================================================================
idx_ch = []
for c, chan in enumerate(ch_names):
    if 'O' in chan or 'P' in chan:
        idx_ch.append(c)
idx_ch = np.array(idx_ch)
eeg = eeg[:,idx_ch]


# =============================================================================
# Average the EEG repeats of each video condition into 4 pseudo-trials
# =============================================================================
video_conditions = np.unique(stimulus_id_test)
n_pseudo_trl = 4
eeg_pseudo = np.zeros((len(video_conditions), n_pseudo_trl, eeg.shape[1],
    eeg.shape[2]), dtype=np.float32)

for v, video in enumerate(video_conditions):
    idx = resample(np.where(stimulus_id_test == video)[0], replace=False)
    n_trl_per_pseudo = int(np.ceil(len(idx) / n_pseudo_trl))
    for p in range(n_pseudo_trl):
        idx_start = p * n_trl_per_pseudo
        idx_end = idx_start + n_trl_per_pseudo
        eeg_pseudo[v,p] = np.nanmean(eeg[idx[idx_start:idx_end]], 0)
del eeg

# Set NaN values to 0
eeg_pseudo = np.nan_to_num(eeg_pseudo, nan=0)


# =============================================================================
# Pairwise decoding
# =============================================================================
# RDMs array of shape:
# (Video conditions × Video conditions × EEG time points)
rdms = np.zeros((len(video_conditions), len(video_conditions),
    eeg_pseudo.shape[3]), dtype=np.float32)

# SVM target vectors
y_train = np.zeros(((n_pseudo_trl-1)*2))
y_train[int(len(y_train)/2):] = 1
y_test = np.asarray((0, 1))

# Loop over EEG time points and video conditions
for t in tqdm(range(eeg_pseudo.shape[3])):
    for v1 in range(len(video_conditions)):
        for v2 in range(v1):

            # Select the video condition data
            eeg_cond_1 = eeg_pseudo[v1,:,:,t]
            eeg_cond_2 = eeg_pseudo[v2,:,:,t]

            # Empty scores array
            scores = np.zeros(n_pseudo_trl, dtype=np.float32)

            # Loop across pseudo-trials
            for p in range(n_pseudo_trl):

                # Define the train/test partitions
                X_train = np.append(np.delete(eeg_cond_1, p, 0),
                    np.delete(eeg_cond_2, p, 0), 0)
                X_test = np.append(np.expand_dims(eeg_cond_1[p], 0),
                    np.expand_dims(eeg_cond_2[p], 0), 0)

                # Train the classifier
                dec_svm = SVC(kernel='linear')
                dec_svm.fit(X_train, y_train)

                # Test the classifier
                y_pred = dec_svm.predict(X_test)
                scores[p] = sum(y_pred == y_test) / len(y_test)

            # Store the accuracy
            rdms[v1,v2,t] = np.mean(scores)
            rdms[v2,v1,t] = rdms[v1,v2,t]


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.emd_dir, 'results',
    'representational_dynamics', 'eeg_model_ct_rsa', 'eeg_rdms')
os.makedirs(save_dir, exist_ok=True)

file_name = f'decoding_rdms_sub-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), rdms)