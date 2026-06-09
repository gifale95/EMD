"""Aggregate the partial correlation results across all subjects, and compute
the confidence intervals.

Parameters
----------
subjects : list
    List of used subjects.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
project_dir : str
    Directory of the project folder.

"""

import os
import argparse
import random
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6], type=list)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Stats <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
random.seed(seed)
np.random.seed(seed)


# =============================================================================
# Load the EEG channels and time points
# =============================================================================
data_dir = os.path.join(args.project_dir, 'derivatives', 'eeg',
    f'sub-{args.subjects[0]:02}', f'sub-{args.subjects[0]:02}_eeg_metadata.npy')
metadata = np.load(data_dir, allow_pickle=True).item()
times = metadata['times']
ch_names = metadata['ch_names']


# =============================================================================
# Load the partial correlation results for all subjects
# =============================================================================
total_variance_vision = []
total_variance_language = []
unique_variance_vision = []
unique_variance_language = []

for sub in args.subjects:

    file_name = f'partial_correlation_sub-{sub:02}.npy'
    results = np.load(os.path.join(args.project_dir, 'results',
        'encoding_models', 'partial_correlation', file_name),
        allow_pickle=True).item()

    total_variance_vision.append(results['total_variance_vision'])
    total_variance_language.append(results['total_variance_language'])
    unique_variance_vision.append(results['unique_variance_vision'])
    unique_variance_language.append(results['unique_variance_language'])
    del results

results = {}
results['total_variance_vision'] = np.array(total_variance_vision)
results['total_variance_language'] = np.array(total_variance_language)
results['unique_variance_vision'] = np.array(unique_variance_vision)
results['unique_variance_language'] = np.array(unique_variance_language)


# =============================================================================
# Average the results across EEG channel groups
# =============================================================================
channel_types = ['O', 'P', 'T', 'C', 'F']
channel_type_names = ['Occipital', 'Parietal', 'Temporal', 'Central',
    'Frontal']
idx_ch = []
for ch_type in channel_types:
    idx = []
    for c, chan in enumerate(ch_names):
        if ch_type in chan:
            idx.append(c)
    idx_ch.append(np.asarray(idx))
    del idx

results_chan_avg = {}
for res_type in results.keys():
    for i, ch_type in enumerate(channel_types):
        results_chan_avg[f'{res_type}_{ch_type}'] = np.mean(
            results[res_type][:,idx_ch[i]], 1)


# =============================================================================
# Compute the confidence intervals
# =============================================================================
ci = {}
for res_type in results_chan_avg.keys():
    ci[res_type] = np.zeros((2, len(times)), dtype=np.float32)

ci_dist = {}
for res_type in results_chan_avg.keys():
    ci_dist[res_type] = np.zeros((args.n_iter, len(times)), dtype=np.float32)

for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    for res_type in results_chan_avg.keys():
        ci_dist[res_type][i] = np.mean(results_chan_avg[res_type][idx], 0)

for res_type in results_chan_avg.keys():
    ci[res_type][0] = np.percentile(ci_dist[res_type], 2.5, axis=0)
    ci[res_type][1] = np.percentile(ci_dist[res_type], 97.5, axis=0)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'partial_correlation': results_chan_avg,
    'ci': ci
}

save_dir = os.path.join(args.project_dir, 'results', 'encoding_models',
    'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)