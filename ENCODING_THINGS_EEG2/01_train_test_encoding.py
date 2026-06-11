"""Fit a linear regression to predict EEG data using DNN feature maps as
predictors. The linear regression is trained using the training images EEG data
(Y) and feature maps (X). A separate model is trained for each EEG channel and
time point, and also for each of the four EEG train data repeats: in this way,
for each input image we can have four different instances of synthetic EEG
response.

The feature maps come from a CLIP vision transformer, and are downsampled to 250
principal components using PCA.

https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html

Parameters
----------
subject : int
    Number of the used THINGS EEG2 subject.
things_eeg_2_dir : str
    Directory of the THINGS EEG2 dataset.
    https://osf.io/3jk45/
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG

"""

import argparse
import numpy as np
import os
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', type=int, default=1)
parser.add_argument('--things_eeg_2_dir', default='/scratch/ccn_datasets/things_eeg_2', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/encoding_things_eeg2_zitong', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the visual features
# =============================================================================
data_dir = os.path.join(args.things_eeg_2_dir, 'dnn_feature_maps',
    'pca_feature_maps', 'cornet_s', 'pretrained-True', 'layers-all')
fmaps_train = np.load(os.path.join(data_dir, 'pca_feature_maps_training.npy'),
    allow_pickle=True).item()['all_layers'][:,:1000]
fmaps_test = np.load(os.path.join(data_dir, 'pca_feature_maps_test.npy'),
    allow_pickle=True).item()['all_layers'][:,:1000]


# =============================================================================
# Train the encoding models
# =============================================================================
# Load the training EEG responses
data_dir = os.path.join(args.things_eeg_2_dir, 'preprocessed_data', f'sub-{format(args.subject,"02")}')
eeg_dir = os.path.join(data_dir, 'preprocessed_eeg_training.npy')
data = np.load(eeg_dir, allow_pickle=True).item()
eeg_train = np.mean(data['preprocessed_eeg_data'], 1).astype(np.float32)
times = data['times']
del data
n_channels = eeg_train.shape[1]
n_times = eeg_train.shape[2]

# Reshape the EEG to (Samples x Features)
eeg_train = np.reshape(eeg_train, (len(eeg_train), -1))
# Fit the linear regression
reg = LinearRegression()
reg.fit(fmaps_train, eeg_train)


# =============================================================================
# Test the encoding models
# =============================================================================
# Load the test EEG responses
data_dir = os.path.join(args.things_eeg_2_dir, 'preprocessed_data', f'sub-{format(args.subject,"02")}')
eeg_dir = os.path.join(data_dir, 'preprocessed_eeg_test.npy')
data = np.load(eeg_dir, allow_pickle=True).item()
eeg_test = np.mean(data['preprocessed_eeg_data'], 1).astype(np.float32)

# Use the learned weights to generate in silico EEG responses for the test
# images
eeg_test_pred = reg.predict(fmaps_test)
eeg_test_pred = np.reshape(eeg_test_pred, (-1, n_channels, n_times))

# Load Zitong's predicted EEG responses
data_dir = os.path.join(args.project_dir, 'img2eeg_test_eeg_pred', f'sub{format(args.subject,"02")}.npy')
eeg_test_pred_img2eeg = np.load(data_dir).astype(np.float32)

# Select time points of interest
eeg_test = eeg_test[:,:,20:70]
eeg_test_pred = eeg_test_pred[:,:,20:70]
times = times[20:70]

# Test the encoding models
correlation_vitb32 = np.zeros((n_channels, len(times)), dtype=np.float32)
correlation_img2eeg = np.zeros((n_channels, len(times)), dtype=np.float32)
for t in tqdm(range(len(times)), leave=False):
    for c in range(n_channels):
        correlation_vitb32[c,t] = pearsonr(eeg_test[:,c,t], eeg_test_pred[:,c,t])[0]
        correlation_img2eeg[c,t] = pearsonr(eeg_test[:,c,t], eeg_test_pred_img2eeg[:,c,t])[0]


# =============================================================================
# Save the correlation results
# =============================================================================
acc = {
    'correlation_vitb32': correlation_vitb32,
    'correlation_img2eeg': correlation_img2eeg,
    'times': times
    }

save_dir = os.path.join(args.project_dir, 'encoding_accuracy')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

file_name = 'encoding_accuracy_sub-' + format(args.subject, '02') + '.npy'

np.save(os.path.join(save_dir, file_name), acc)