"""Aggregate the partial correlation results across all subjects, and compute
the confidence intervals.

Parameters
----------
subjects : list
    List of used subjects.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).

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
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
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
data_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg',
    f'sub-{args.subjects[0]:02}', f'sub-{args.subjects[0]:02}_eeg_metadata.npy')
metadata = np.load(data_dir, allow_pickle=True).item()
times = metadata['times']
ch_names = metadata['ch_names']


# =============================================================================
# Load the partial correlation results for all subjects
# =============================================================================
variance_vision = []
variance_language = []
unique_variance_vision = []
unique_variance_language = []

for sub in args.subjects:

    file_name = f'partial_correlation_sub-{sub:02}.npy'
    results = np.load(os.path.join(args.emd_dir, 'results',
        'encoding_models', 'partial_correlation', file_name),
        allow_pickle=True).item()

    variance_vision.append(results['variance_vision'])
    variance_language.append(results['variance_language'])
    unique_variance_vision.append(results['unique_variance_vision'])
    unique_variance_language.append(results['unique_variance_language'])
    del results

results = {}
results['variance_vision'] = np.array(variance_vision)
results['variance_language'] = np.array(variance_language)
results['unique_variance_vision'] = np.array(unique_variance_vision)
results['unique_variance_language'] = np.array(unique_variance_language)


# =============================================================================
# Average the results occipital and parietal EEG channels
# =============================================================================
idx_ch = []
for c, chan in enumerate(ch_names):
    if 'O' in chan or 'P' in chan:
        idx_ch.append(c)

results_chan_avg = {}
for res_type in results.keys():
    results_chan_avg[res_type] = np.mean(results[res_type][:,idx_ch], 1)


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

save_dir = os.path.join(args.emd_dir, 'results', 'encoding_models', 'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)