"""Run a pairwise decoding analysis on the eyetracking pupil size data for the
102 test videos.

Parameters
----------
subject : int
    Used subject.
project_dir : str
    Directory of the project folder.

"""

import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.svm import SVC


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Pairwise decoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the eyetracking pupil size data for the 102 test videos
# =============================================================================
# Load the stimulus IDs
data_dir = os.path.join(args.project_dir, 'derivatives', 'eyetracking',
    f'sub-{args.subject:02}')
file_name = f'sub-{args.subject:02}_eyetracking_metadata.npy'
metadata = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
stimulus_id = metadata['stimulus_id']

# Load the eyetracking data
pupil = []
stimulus_id_test = []
n_sessions = 8
# Loop across eyetracking recording sessions
for ses in range(1, n_sessions+1):
    # Load the eyetracking pupil size data
    file_name = f'sub-{args.subject:02}_ses-{ses:02}_preprocessed_eyetracking.h5'
    pupil_ses = h5py.File(os.path.join(data_dir, file_name), 'r')['eyetracking'][:,2]
    # Select the data for the test videos
    idx = np.where(stimulus_id[f'ses-{ses:02}'] > 1000)[0]
    pupil.append(pupil_ses[idx])
    stimulus_id_test.append(stimulus_id[f'ses-{ses:02}'][idx])
    del pupil_ses, idx
pupil = np.concatenate(pupil, 0)
stimulus_id_test = np.concatenate(stimulus_id_test, 0)


# =============================================================================
# Average the eyetracking repeats of each video condition into 4 pseudo-trials
# =============================================================================
video_conditions = np.unique(stimulus_id_test)
n_pseudo_trl = 4
pupil_pseudo = np.zeros((len(video_conditions), n_pseudo_trl, pupil.shape[1]),
    dtype=np.float32)

for v, video in enumerate(video_conditions):
    idx = resample(np.where(stimulus_id_test == video)[0], replace=False)
    n_trl_per_pseudo = int(np.ceil(len(idx) / n_pseudo_trl))
    for p in range(n_pseudo_trl):
        idx_start = p * n_trl_per_pseudo
        idx_end = idx_start + n_trl_per_pseudo
        pupil_pseudo[v,p] = np.nanmean(pupil[idx[idx_start:idx_end]], 0)
del pupil

# Set NaN values to 0
pupil_pseudo = np.nan_to_num(pupil_pseudo, nan=0)


# =============================================================================
# Pairwise decoding
# =============================================================================
# RDMs array of shape:
# (Video conditions × Video conditions × Time points)
rdms = np.zeros((len(video_conditions), len(video_conditions),
    pupil_pseudo.shape[2]), dtype=np.float32)

# SVM target vectors
y_train = np.zeros(((n_pseudo_trl-1)*2))
y_train[int(len(y_train)/2):] = 1
y_test = np.asarray((0, 1))

# Loop over eyetracking time points and video conditions
for t in tqdm(range(pupil_pseudo.shape[2])):
    for v1 in range(len(video_conditions)):
        for v2 in range(v1):

            # Select the video condition data
            pupil_cond_1 = pupil_pseudo[v1,:,t]
            pupil_cond_2 = pupil_pseudo[v2,:,t]

            # Empty scores array
            scores = np.zeros(n_pseudo_trl, dtype=np.float32)

            # Loop across pseudo-trials
            for p in range(n_pseudo_trl):

                # Define the train/test partitions
                X_train = np.expand_dims(np.append(np.delete(pupil_cond_1, p),
                    np.delete(pupil_cond_2, p)), -1)
                X_test = np.expand_dims(np.append(pupil_cond_1[p],
                    pupil_cond_2[p]), -1)
                X_train = np.round(X_train.astype(np.float64), 5)
                X_test = np.round(X_test.astype(np.float64), 5)

                # Train the classifier
                dec_svm_gaze = SVC(kernel='linear')
                dec_svm_gaze.fit(X_train, y_train)

                # Test the classifier
                y_pred = dec_svm_gaze.predict(X_test)
                scores[p] = sum(y_pred == y_test) / len(y_test)

            # Store the accuracy
            rdms[v1,v2,t] = np.mean(scores)
            rdms[v2,v1,t] = rdms[v1,v2,t]


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'data_quality_check',
    'eyetracking', 'pairwise_decoding_rdms')
os.makedirs(save_dir, exist_ok=True)

file_name = f'pupil_rdms_sub-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), rdms)