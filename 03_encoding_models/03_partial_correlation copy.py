"""Aggregate the partial corr.

Parameters
----------
subject : int
    Used subject.
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
from scipy.stats import pearsonr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Partial correlation <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the EEG responses for the 102 test videos
# =============================================================================
# Load the stimulus IDs
data_dir = os.path.join(args.project_dir, 'derivatives', 'eeg',
    f'sub-{args.subject:02}')
file_name = f'sub-{args.subject:02}_eeg_metadata.npy'
metadata = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
stimulus_id = metadata['stimulus_id']

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
# Average the EEG responses of each video condition across repeats
# =============================================================================
video_conditions = np.unique(stimulus_id_test)
eeg_test = np.zeros((len(video_conditions), eeg.shape[1], eeg.shape[2]),
    dtype=np.float32)

for v, video in enumerate(video_conditions):
    idx = np.where(stimulus_id_test == video)[0]
    eeg_test[v] = np.nanmean(eeg[idx], 0)
del eeg

# Set NaN values to 0
eeg_test = np.nan_to_num(eeg_test, nan=0)

# Reshape the EEG responses to (samples, features)
eeg_shape = eeg_test.shape
eeg_test = np.reshape(eeg_test, (len(eeg_test), -1))


# =============================================================================
# Load the test stimulus features
# =============================================================================
# Load the stimulus features
feature_dir_vision = os.path.join(args.project_dir, 'results',
    'stimulus_features', 'vision_features', 's3d', 'vision_features_test.npy')
feature_dir_language = os.path.join(args.project_dir, 'results',
    'stimulus_features', 'language_features', 'all-mpnet-base-v2',
    'language_features_test.npy')
vision_features_test = np.load(feature_dir_vision)
language_features_test = np.load(feature_dir_language)


# =============================================================================
# Instantiate the encoding models with the trained weights
# =============================================================================
# Load the trained encoding model weights
reg_dir_vision = os.path.join(args.project_dir, 'results', 'encoding_models',
    'model_weights', f'weights_sub-{args.subject:02}_modality_vision.npy')
reg_dir_language = os.path.join(args.project_dir, 'results', 'encoding_models',
    'model_weights', f'weights_sub-{args.subject:02}_modality_language.npy')
reg_param_vision = np.load(reg_dir_vision, allow_pickle=True).item()
reg_param_language = np.load(reg_dir_language, allow_pickle=True).item()

# Instantiate the encoding models with the trained weights
reg_vision = LinearRegression()
reg_vision.coef_ = reg_param_vision['coef_']
reg_vision.intercept_ = reg_param_vision['intercept_']
reg_vision.n_features_in_ = reg_param_vision['n_features_in_']
reg_language = LinearRegression()
reg_language.coef_ = reg_param_language['coef_']
reg_language.intercept_ = reg_param_language['intercept_']
reg_language.n_features_in_ = reg_param_language['n_features_in_']
del reg_param_vision, reg_param_language


# =============================================================================
# Use the encoding models to predict the EEG responses for the test videos
# =============================================================================
# Vision models
eeg_test_pred_vision = reg_vision.predict(vision_features_test)
del reg_vision, vision_features_test

# Language models (since each video has 6 captions, average the predicted EEG
# responses across the 6 captions for each video)
eeg_test_pred_language = []
for v in range(language_features_test.shape[0]):
    eeg_test_pred_language.append(np.mean(
        reg_language.predict(language_features_test[v]), 0))
eeg_test_pred_language = np.array(eeg_test_pred_language)
del reg_language, language_features_test


# =============================================================================
# Partial correlation
# =============================================================================
# Empty result arrays of shape (channels * time points)
total_variance_vision = np.zeros((eeg_test.shape[1]), dtype=np.float32)
total_variance_language = np.zeros((eeg_test.shape[1]), dtype=np.float32)
unique_variance_vision = np.zeros((eeg_test.shape[1]), dtype=np.float32)
unique_variance_language = np.zeros((eeg_test.shape[1]), dtype=np.float32)

# Loop across EEG features
for f in tqdm(range(eeg_test.shape[1])):

    # Compute the total variance explained by the vision models
    total_variance_vision[f] = pearsonr(eeg_test[:,f],
        eeg_test_pred_vision[:,f])[0]

    # Compute the total variance explained by the language models
    total_variance_language[f] = pearsonr(eeg_test[:,f],
        eeg_test_pred_language[:,f])[0]

    # Compute the unique variance explained by vision models
    # Remove the linear relationship with language models from the EEG
    # responses
    reg = LinearRegression()
    reg.fit(eeg_test_pred_language[:,f], eeg_test[:,f])
    eeg_test_res = eeg_test[:,f] - reg.predict(eeg_test_pred_language[:,f])
    del reg
    # Remove the linear relationship with language models from the
    # vision-model-predicted EEG responses
    reg = LinearRegression()
    reg.fit(eeg_test_pred_language[:,f], eeg_test_pred_vision[:,f])
    eeg_test_pred_vision_res = eeg_test_pred_vision[:,f] - \
        reg.predict(eeg_test_pred_language[:,f])
    del reg
    # Correlate the residuals to get the unique variance explained by
    # vision models
    unique_variance_vision[f] = pearsonr(eeg_test_res,
        eeg_test_pred_vision_res)[0]

    # Compute the unique variance explained by language models
    # Remove the linear relationship with vision models from the EEG
    # responses
    reg = LinearRegression()
    reg.fit(eeg_test_pred_vision[:,f], eeg_test[:,f])
    eeg_test_res = eeg_test[:,f] - reg.predict(eeg_test_pred_vision[:,f])
    del reg
    # Remove the linear relationship with vision models from the
    # language-model-predicted EEG responses
    reg = LinearRegression()
    reg.fit(eeg_test_pred_vision[:,f], eeg_test_pred_language[:,f])
    eeg_test_pred_language_res = eeg_test_pred_language[:,f] - \
        reg.predict(eeg_test_pred_vision[:,f])
    del reg
    # Correlate the residuals to get the unique variance explained by
    # language models
    unique_variance_language[f] = pearsonr(eeg_test_res,
        eeg_test_pred_language_res)[0]

# Reshape the partial correlation results to (channels, time points)
total_variance_vision = np.reshape(total_variance_vision,
    (eeg_shape[1], eeg_shape[2]))
total_variance_language = np.reshape(total_variance_language,
    (eeg_shape[1], eeg_shape[2]))
unique_variance_vision = np.reshape(unique_variance_vision,
    (eeg_shape[1], eeg_shape[2]))
unique_variance_language = np.reshape(unique_variance_language,
    (eeg_shape[1], eeg_shape[2]))


# =============================================================================
# Save the partial correlation results
# =============================================================================
results = {
    'total_variance_vision': total_variance_vision,
    'total_variance_language': total_variance_language,
    'unique_variance_vision': unique_variance_vision,
    'unique_variance_language': unique_variance_language
}

save_dir = os.path.join(args.project_dir, 'results', 'encoding_models',
    'partial_correlation')
os.makedirs(save_dir, exist_ok=True)

file_name = f'partial_correlation_sub-{args.subject:02}.npy'

np.save(os.path.join(save_dir, file_name), results)