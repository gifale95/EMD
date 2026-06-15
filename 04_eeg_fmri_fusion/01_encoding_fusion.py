"""Train encoding fusion models that map EMD's EEG responses onto the fMRI
responses from the BOLD Moments Dataset (BMD), independently for each EEG time
point.

Parameters
----------
eeg_subject : int
    EEG subject number (EMD).
fmri_subject : int
    fMRI subject number (BMD).
fmri_hemi : str
    Whether to use the 'left' or 'right' fMRI hemisphere.
tot_eeg_time_splits : int
    The total number of splits in which the EEG time points are divided.
eeg_time_split : int
    The EEG time split used, out of the total time splits.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).
bmd_dir : str
    Directory of the fMRI Moments Dataset (BMD).

"""

import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
import h5py
from scipy.stats import zscore
from sklearn.linear_model import RidgeCV
from scipy.stats import pearsonr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--eeg_subject', type=int, default=1)
parser.add_argument('--fmri_subject', type=int, default=1)
parser.add_argument('--fmri_hemi', type=str, default='left')
parser.add_argument('--tot_eeg_time_splits', default=5, type=int)
parser.add_argument('--eeg_time_split', default=0, type=int)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
parser.add_argument('--bmd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset/bold_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> EEG-fMRI encoding fusion <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Time point selection
# =============================================================================
# Load the EEG metadata
data_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg',
    f'sub-{args.eeg_subject:02}')
file_name = f'sub-{args.eeg_subject:02}_eeg_metadata.npy'
metadata = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()

# Get the time points
times = metadata['times']

# Select the time points from the current time split
times_per_split = int(np.ceil(len(times) / args.tot_eeg_time_splits))
start_idx = args.eeg_time_split * times_per_split
end_idx = min((args.eeg_time_split + 1) * times_per_split, len(times))


# =============================================================================
# Load the EEG responses for train and test videos
# =============================================================================
# Load the EEG responses
eeg = []
stimulus_id = []
# Loop across EEG recording sessions
n_sessions = 8
for ses in tqdm(range(1, n_sessions+1)):
    # Load the EEG responses
    file_name = f'sub-{args.eeg_subject:02}_ses-{ses:02}_preprocessed_eeg.h5'
    eeg_ses = h5py.File(os.path.join(data_dir, file_name), 'r')['eeg']
    # Z-score the EEG responses at each session
    eeg_ses = zscore(eeg_ses[:,:,start_idx:end_idx], axis=0, nan_policy='omit')
    # Store the EEG responses
    eeg.append(eeg_ses)
    stimulus_id.append(metadata['stimulus_id'][f'ses-{ses:02}'])
    del eeg_ses
eeg = np.concatenate(eeg, 0)
stimulus_id = np.concatenate(stimulus_id, 0)


# =============================================================================
# Average the EEG responses of each video condition across repeats
# =============================================================================
# EEG train responses
train_video_ids = np.arange(1, 1001)
eeg_train = np.zeros((len(train_video_ids), eeg.shape[1], eeg.shape[2]),
    dtype=np.float32)
for v, video in enumerate(train_video_ids):
    idx = np.where(stimulus_id == video)[0]
    eeg_train[v] = np.nanmean(eeg[idx], 0)

# EEG test responses
test_video_ids = np.arange(1001, 1103)
eeg_test = np.zeros((len(test_video_ids), eeg.shape[1], eeg.shape[2]),
    dtype=np.float32)
for v, video in enumerate(test_video_ids):
    idx = np.where(stimulus_id == video)[0]
    eeg_test[v] = np.nanmean(eeg[idx], 0)
del eeg

# Set NaN values to 0
eeg_train = np.nan_to_num(eeg_train, nan=0)
eeg_test = np.nan_to_num(eeg_test, nan=0)


# =============================================================================
# Load fMRI responses for the train and test videos
# =============================================================================
# The version B fMRI responses from BMD are already z-scored at each scan
# session.

# Load the fMRI train responses
data_dir = os.path.join(args.bmd_dir, 'derivatives', 'versionB', 'fsaverage',
    'GLM', f'sub-{args.fmri_subject:02}', 'prepared_betas')
file_name_train = (f'sub-{args.fmri_subject:02}_organized_betas_'
    f'task-train_hemi-{args.fmri_hemi}_normalized.pkl')
f = open(os.path.join(data_dir, file_name_train), 'rb')
fmri_train = pickle.load(f)[0]
del f

# Load the fMRI test responses
file_name_test = (f'sub-{args.fmri_subject:02}_organized_betas_'
    f'task-test_hemi-{args.fmri_hemi}_normalized.pkl')
f = open(os.path.join(data_dir, file_name_test), 'rb')
fmri_test = pickle.load(f)[0]
del f

# Average the fMRI responses across repeats
fmri_train = np.mean(fmri_train, 1).astype(np.float32)
fmri_test = np.mean(fmri_test, 1).astype(np.float32)


# =============================================================================
# Loop across EEG time points
# =============================================================================
n_vertices = fmri_train.shape[1]
n_times = eeg_train.shape[2]
correlation = np.zeros((n_vertices, n_times), dtype=np.float32)
for t in tqdm(range(n_times)):


# =============================================================================
# Train the encoding fusion models
# =============================================================================
    alphas = np.logspace(-6, 10, 17)
    reg = RidgeCV(alphas=alphas, cv=None, alpha_per_target=True)
    reg.fit(eeg_train[:,:,t], fmri_train)


# =============================================================================
# Test the encoding fusion models
# =============================================================================
    # Predict the fMRI responses for the test images
    fmri_test_pred = reg.predict(eeg_test[:,:,t])
    del reg

    # Correlate the predicted fMRI responses with the in vivo fMRI responses
    correlation[:,t] = pearsonr(fmri_test_pred, fmri_test)[0]
    del fmri_test_pred


# =============================================================================
# Save the correlation results
# =============================================================================
save_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion',
    'correlation')
os.makedirs(save_dir, exist_ok=True)

file_name = (f'correlation_eeg_sub-{args.eeg_subject:02}_'
    f'time_split-{args.eeg_time_split:02}_fmri_sub-{args.fmri_subject:02}_'
    f'hemi-{args.fmri_hemi}.npy')

np.save(os.path.join(save_dir, file_name), correlation)