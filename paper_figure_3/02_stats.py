"""Compute the confidence intervals of the NCSNR and pairwise decoding scores,
as well as the significance of the pairwise decoding scores.

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

import argparse
import os
from tqdm import tqdm
import numpy as np
import random
from sklearn.utils import resample
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


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
# Load the EEG times and NCSNR scores
# =============================================================================
ncsnr = []
for sub in args.subjects:
    data_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg',
        f'sub-{sub:02}', f'sub-{sub:02}_eeg_metadata.npy')
    metadata = np.load(data_dir, allow_pickle=True).item()
    ncsnr.append(metadata['ncsnr'])
    times = metadata['times']
    ch_names = metadata['ch_names']
ncsnr = np.asarray(ncsnr)

# EEG channel selection
idx_ch = []
for c, chan in enumerate(ch_names):
    if 'O' in chan or 'P' in chan:
        idx_ch.append(c)
idx_ch = np.array(idx_ch)
ncsnr = np.mean(ncsnr[:,idx_ch], 1)


# =============================================================================
# Load the pairwise decoding results of all subjects
# =============================================================================
decoding = []
for s, sub in enumerate(args.subjects):
    data_dir = os.path.join(args.emd_dir, 'results',
        'data_quality_check', 'eeg', 'pairwise_decoding_rdms',
        f'rdms_sub-{sub:02d}_channels-all.npy')
    rdms = np.load(data_dir)
    if s == 0:
        idx_tril = np.tril_indices(rdms.shape[0], k=-1)
    decoding.append(np.mean(rdms[idx_tril], 0) * 100)
decoding = np.array(decoding)


# =============================================================================
# Compute the confidence intervals of the NCSNR and pairwise decoding results
# =============================================================================
ci_ncsnr = np.zeros((2, len(times)), dtype=np.float32)
ci_decoding = np.zeros((2, len(times)), dtype=np.float32)
ci_ncsnr_dist = np.zeros((args.n_iter, len(times)), dtype=np.float32)
ci_decoding_dist = np.zeros((args.n_iter, len(times)), dtype=np.float32)

for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.subjects)))
    ci_ncsnr_dist[i] = np.nanmean(ncsnr[idx], 0)
    ci_decoding_dist[i] = np.mean(decoding[idx], 0)

ci_ncsnr[0] = np.percentile(ci_ncsnr_dist, 2.5, axis=0)
ci_ncsnr[1] = np.percentile(ci_ncsnr_dist, 97.5, axis=0)

ci_decoding[0] = np.percentile(ci_decoding_dist, 2.5, axis=0)
ci_decoding[1] = np.percentile(ci_decoding_dist, 97.5, axis=0)


# =============================================================================
# Compute the significance of the partwise decoding results
# =============================================================================
# Compute the p-value with a t-test against 0
pval_decoding = ttest_1samp(decoding, 50, axis=0, alternative='greater')[1]

# Correct for multiple comparisons
sig_decoding = multipletests(pval_decoding, 0.05, 'fdr_bh')[0]


# =============================================================================
# Save the stats
# =============================================================================
results = {
    'decoding': decoding,
    'ncsnr': ncsnr,
    'ci_ncsnr': ci_ncsnr,
    'ci_decoding': ci_decoding,
    'pval_decoding': pval_decoding,
    'sig_decoding': sig_decoding
}

save_dir = os.path.join(args.emd_dir, 'results', 'data_quality_check', 'eeg',
    'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats_channels-all.npy'

np.save(os.path.join(save_dir, file_name), results)