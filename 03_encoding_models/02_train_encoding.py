"""Train EEG encoding models using vision or language stimulus features.

Parameters
----------
subject : int
    Used subject.
modality : str
    Whether to train the EEG encoding models using 'vision' or 'language'
    stimulus features.
project_dir : str
    Directory of the project folder.

"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import h5py
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--modality', default='language', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the EEG responses for the 1000 train videos
# =============================================================================
# Load the stimulus IDs
data_dir = os.path.join(args.project_dir, 'derivatives', 'eeg',
    f'sub-{args.subject:02}')
file_name = f'sub-{args.subject:02}_eeg_metadata.npy'
metadata = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
stimulus_id = metadata['stimulus_id']

# Load the EEG responses
eeg = []
stimulus_id_train = []
n_sessions = 8
# Loop across EEG recording sessions
for ses in tqdm(range(1, n_sessions+1)):
    # Load the EEG responses
    file_name = f'sub-{args.subject:02}_ses-{ses:02}_preprocessed_eeg.h5'
    eeg_ses = h5py.File(os.path.join(data_dir, file_name), 'r')['eeg'][:]
    # Z-score the EEG responses at each session
    eeg_ses = zscore(eeg_ses, axis=0, nan_policy='omit')
    # Select the responses for the train videos
    idx = np.where(stimulus_id[f'ses-{ses:02}'] <= 1000)[0]
    eeg.append(eeg_ses[idx])
    stimulus_id_train.append(stimulus_id[f'ses-{ses:02}'][idx])
    del eeg_ses, idx
eeg = np.concatenate(eeg, 0)
stimulus_id_train = np.concatenate(stimulus_id_train, 0)


# =============================================================================
# Average the EEG responses of each video condition across repeats
# =============================================================================
video_conditions = np.unique(stimulus_id_train)
eeg_train = np.zeros((len(video_conditions), eeg.shape[1], eeg.shape[2]),
    dtype=np.float32)

for v, video in enumerate(video_conditions):
    idx = np.where(stimulus_id_train == video)[0]
    eeg_train[v] = np.nanmean(eeg[idx], 0)
del eeg

# Set NaN values to 0
eeg_train = np.nan_to_num(eeg_train, nan=0)

# Reshape the EEG responses to (samples, features)
eeg_train = np.reshape(eeg_train, (len(eeg_train), -1))


# =============================================================================
# Load the stimulus features
# =============================================================================
# Load the stimulus features
if args.modality == 'vision':
    model = 's3d'
elif args.modality == 'language':
    model = 'all-mpnet-base-v2'
features_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
    f'{args.modality}_features', model, f'{args.modality}_features_train.npy')
features_train = np.load(features_dir)

# If training encoding models with the language features, create one sample
# for each of the 6 captions per video
if args.modality == 'language':
    # Reshape the language features
    features_train_shape = features_train.shape
    features_train = np.reshape(features_train, (-1, features_train_shape[2]))
    # Duplicate the EEG responses
    eeg_train = np.repeat(eeg_train, features_train_shape[1], axis=0)


# =============================================================================
# Train the encoding models
# =============================================================================
reg = LinearRegression()
reg.fit(features_train, eeg_train)

reg_param = {
    'coef_': reg.coef_.astype(np.float32),
    'intercept_': reg.intercept_.astype(np.float32),
    'n_features_in_': reg.n_features_in_
}


# =============================================================================
# Save the encoding model weights
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'encoding_models',
    'model_weights')
os.makedirs(save_dir, exist_ok=True)

file_name = f'weights_sub-{args.subject:02}_modality_{args.modality}.npy'

np.save(os.path.join(save_dir, file_name), reg_param)