"""Aggregate the EEG-fMRI encoding fusion correlation results, average them
across vertices from the same ROIs, and compute the confidence intervals.

Parameters
----------
eeg_subjects : list
    List of EEG subject numbers (EMD).
tot_eeg_time_splits : int
    The total number of splits in which the EEG time points are divided.
fmri_subjects : list
    List of fMRI subject numbers (BMD).
fmri_hemis : list
    List of fMRI hemispheres.
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
parser.add_argument('--eeg_subjects', default=[1, 2, 3, 4, 5, 6], type=list)
parser.add_argument('--tot_eeg_time_splits', default=5, type=int)
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--fmri_hemis', default=['left', 'right'], type=list)
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
# Load the correlation results
# =============================================================================
result_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion',
    'correlation')

# Loop across EEG and fMRI subjects
lh_correlation = []
rh_correlation = []
for eeg_sub in args.eeg_subjects:
    for fmri_sub in args.fmri_subjects:

        lh_corr_fmri_sub = []
        rh_corr_fmri_sub = []

        # Loop across fMRI hemispheres and EEG time points
        for h, fmri_hemi in enumerate(args.fmri_hemis):
            for eeg_time_split in range(args.tot_eeg_time_splits):

                # Load the correlation results
                file_name = (f'correlation_eeg_sub-{eeg_sub:02}_'
                    f'time_split-{eeg_time_split:02}_fmri_sub-{fmri_sub:02}_'
                    f'hemi-{fmri_hemi}.npy')
                corr = np.load(os.path.join(result_dir, file_name))

                # Append the results across EEG time splits
                if eeg_time_split == 0:
                    correlation = corr
                else:
                    correlation = np.append((correlation, corr), 1)
                del corr

            # Append the results across fMRI subjects
            if h == 0:
                lh_corr_fmri_sub.append(correlation)
            else:
                rh_corr_fmri_sub.append(correlation)
            del correlation
    
    # Append the results across EEG subjects
    lh_correlation.append(np.array(lh_corr_fmri_sub))
    rh_correlation.append(np.array(rh_corr_fmri_sub))
    del lh_corr_fmri_sub, rh_corr_fmri_sub

# Format to numpy arrays
lh_correlation = np.array(lh_correlation)
rh_correlation = np.array(rh_correlation)


# =============================================================================
# ?????????????????? # !!!
# =============================================================================
MAP ROI MASKS FROM VOLUME TO SURFACE SPACE

AVERAGE RESULTS ACROSS VERTICES FROM SAME ROI, USING ONLY VERTICES WITH NOISE CEILING ABOVE THRESHOLD;
THEN AVERAGE THESE RESULTS ACROSS FMRI SUBJECTS (SAVE THESE RESULTS)

COMPUTE CONFIDENCE INTERVALS OF ROI RESULTS ACROSS EEG SUBJECTS (SAVE CIs)



# =============================================================================
# Compute the confidence intervals # !!!
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
results = { # !!!
    'lh_correlation': np.mean(lh_correlation, (0, 1)),
    'rh_correlation': np.mean(rh_correlation, (0, 1)),

}

save_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion',
    'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)