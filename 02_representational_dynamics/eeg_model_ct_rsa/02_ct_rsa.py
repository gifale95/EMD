"""Compute cross-temporal representational similarity analysis (CT-RSA) between
EEG RDMs and AlexNet RDMs. In brief, the RDMs of each EEG time points are
correlated with the RDMs of each AlexNet video time point and layer.

The AlexNet RDMs were computed using code from this GitHub repo:
https://github.com/AnneWZonneveld/eeg-moments

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
from tqdm import tqdm
from scipy.stats import spearmanr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> CT-RSA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the EEG RDMs
# =============================================================================
# Load the RDMs
data_dir = os.path.join(args.emd_dir, 'results', 'representational_dynamics',
    'eeg_model_ct_rsa', 'eeg_rdms',
    f'correlation_rdms_sub-{args.subject:02d}.npy')
eeg_rdms = np.load(data_dir)

# Take the lower triangle of the RDMs
idx_lower = np.tril_indices(eeg_rdms.shape[0], k=-1)
eeg_rdms = eeg_rdms[idx_lower]


# =============================================================================
# Load the AlexNet RDMs
# =============================================================================
# Loop across AlexNet layers
alexnet_rdms = {}
layers = [
    'features_2',
    'features_5',
    'features_7',
    'features_9',
    'features_12',
    'classifier_0',
    'classifier_1',
    'classifier_2',
    'classifier_5',
    'classifier_6'
]
for layer in layers:

    # Load the RDMs
    data_dir = os.path.join(args.emd_dir, 'results',
        'representational_dynamics', 'eeg_model_ct_rsa', 'alexnet_rdms',
        f'RDM_{layer}.npz')
    alexnet_rdms[layer] = np.swapaxes(np.squeeze(
        np.load(data_dir, allow_pickle=True)['rdm']), 0, 1)


# =============================================================================
# Cross-temporal RSA between the EEG RDMs and the AlexNet RDMs
# =============================================================================
# Loop across AlexNet layers
ct_rsa = {}
for layer in tqdm(layers):

    # Loop across EEG and AlexNet time points
    ct_rsa_layer = np.zeros((eeg_rdms.shape[1], alexnet_rdms[layer].shape[1]),
        dtype=np.float32)
    for t_eeg in range(eeg_rdms.shape[1]):
        for t_alexnet in range(alexnet_rdms[layer].shape[1]):

            # Correlate the EEG RDMs with the AlexNet RDMs
            ct_rsa_layer[t_eeg, t_alexnet] = spearmanr(
                eeg_rdms[:,t_eeg], alexnet_rdms[layer][:,t_alexnet])[0]
    
    # Store the CT-RSA results for the current layer
    ct_rsa[layer] = ct_rsa_layer
    del ct_rsa_layer


# =============================================================================
# Save the results
# =============================================================================
save_dir = os.path.join(args.emd_dir, 'results',
    'representational_dynamics', 'eeg_model_ct_rsa', 'ct_rsa')
os.makedirs(save_dir, exist_ok=True)

file_name = f'ct_rsa_sub-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), ct_rsa)