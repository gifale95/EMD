"""Get the best model time points for each EEG time point, and compute the
confidence intervals.

Parameters
----------
subjects : list
    List of used subjects.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).

"""

import argparse
import os
import numpy as np
import random
from tqdm import tqdm
from sklearn.utils import resample
from scipy.stats import spearmanr
from scipy.stats import permutation_test
from scipy.stats import linregress


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6], type=list)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the cross-temporal correlation results
# =============================================================================
# Loop across AlexNet layers
ct_rsa = {}
layers = [
    'features_2',
    'features_5',
    'features_7',
    'features_9',
    'features_12',
    'classifier_2',
    'classifier_5',
    'classifier_6'
]

# Loop across EEG subjects
for s, sub in enumerate(args.subjects):

    # Load the CT-RSA results
    data_dir = os.path.join(args.emd_dir, 'results',
        'representational_dynamics', 'ct_rsa', f'ct_rsa_sub-{sub:02d}.npy')
    ct_rsa_sub = np.load(data_dir, allow_pickle=True).item()

    # Store the CT-RSA results of each subject
    for layer in layers:
        if s == 0:
            ct_rsa[layer] = []
        ct_rsa[layer].append(ct_rsa_sub[layer])

# Format the results into numpy arrays
for layer in layers:
    ct_rsa[layer] = np.array(ct_rsa[layer])


# =============================================================================
# Get the time points
# =============================================================================
# EEG time points
min_times = 0
max_times = 3
n_bins_eeg = ct_rsa[layers[0]].shape[2]
times_eeg = np.linspace(min_times, max_times, n_bins_eeg)

# AlexNet time points
n_times_alex = ct_rsa[layers[0]].shape[1]
times_alex = np.linspace(0, 3, n_times_alex)


# =============================================================================
# Get the best model time points for each EEG time point 
# =============================================================================
# Get the best model time points for each EEG time point (based on the average
# of the top-5 model time points to get a more robust estimate)
ct_rsa_best_model_time = {}
for layer in layers:
    rsa = np.mean(ct_rsa[layer], 0)
    idx_best = np.mean(np.argsort(rsa, 0)[-5:], 0) / n_times_alex * max_times
    ct_rsa_best_model_time[layer] = idx_best
    del idx_best

# Compute the correlation between EEG and AlexNet time points
corr_eeg_model_times = {}
p_val = {}
for layer in layers:
    x = ct_rsa_best_model_time[layer]
    y = times_eeg
    corr_eeg_model_times[layer] = spearmanr(x, y, alternative='greater')
    # Compute the p-value using a permutation test
    def statistic(x): # permute only `x`
        return spearmanr(x, y).statistic
    p_val[layer] = permutation_test((x,), statistic,
        permutation_type='pairings')

# Fit a regression line between EEG and AlexNet time points
regression = {}
for layer in layers:
    regression[layer] = linregress(times_eeg, ct_rsa_best_model_time[layer])


# =============================================================================
# Compute the confidence intervals of the best model time points
# =============================================================================
ci_best_model_time = {}
for layer in layers:
    ci_best_model_time[layer] = np.zeros((2, len(times_eeg)), dtype=np.float32)

ci_dist = {}
for layer in layers:
    ci_dist[layer] = np.zeros((args.n_iter, len(times_eeg)), dtype=np.float32)

for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    for layer in layers:
        rsa = np.mean(ct_rsa[layer][idx], 0)
        idx_best = np.mean(np.argsort(rsa, 0)[-5:], 0) / n_times_alex * max_times
        ci_dist[layer][i] = idx_best

for layer in layers:
    ci_best_model_time[layer][0] = np.percentile(ci_dist[layer], 2.5, axis=0)
    ci_best_model_time[layer][1] = np.percentile(ci_dist[layer], 97.5, axis=0)


# =============================================================================
# Save the results
# =============================================================================
stats = {
    'times_eeg': times_eeg,
    'times_alex': times_alex,
    'ct_rsa': ct_rsa,
    'ct_rsa_best_model_time': ct_rsa_best_model_time,
    'ci_best_model_time': ci_best_model_time,
    'corr_eeg_model_times': corr_eeg_model_times,
    'p_val': p_val,
    'regression': regression
}

save_dir = os.path.join(args.emd_dir, 'results', 'representational_dynamics',
    'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), stats)