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
noise_ceiling_threshold : float
    The threshold on the noise ceiling for vertex selection.
n_iter : int
    Amount of iterations for creating the confidence intervals bootstrapped
    distribution.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).
bmd_dir : str
    Directory of the fMRI Moments Dataset (BMD).

"""

import os
import argparse
import random
import numpy as np
import pickle
from sklearn.utils import resample
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--eeg_subjects', default=[1, 2, 3], type=list) # !!! [1, 2, 3, 4, 5, 6]
parser.add_argument('--tot_eeg_time_splits', default=5, type=int)
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--fmri_hemis', default=['left', 'right'], type=list)
parser.add_argument('--noise_ceiling_threshold', default=20, type=float)
parser.add_argument('--n_iter', default=100000, type=int)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
parser.add_argument('--bmd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset/bold_moments_dataset', type=str)
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
            if fmri_hemi == 'left':
                lh_corr_fmri_sub.append(correlation)
            elif fmri_hemi == 'right':
                rh_corr_fmri_sub.append(correlation)
            del correlation
    
    # Append the results across EEG subjects
    lh_correlation.append(np.array(lh_corr_fmri_sub))
    rh_correlation.append(np.array(rh_corr_fmri_sub))
    del lh_corr_fmri_sub, rh_corr_fmri_sub

# Format to numpy arrays
lh_correlation = np.array(lh_correlation)
rh_correlation = np.array(rh_correlation)

# Inserts the results into a dictionary for easier access at later stages
correlation = {
    'hemi-left': lh_correlation,
    'hemi-right': rh_correlation
}
del lh_correlation, rh_correlation


# =============================================================================
# Load the noise ceiling and ROI masks of the fMRI subjects
# =============================================================================
# Loop across fMRI subjects
noise_ceiling = []
roi_masks = []
for s, fmri_sub in enumerate(args.fmri_subjects):

    # Load the noise ceiling
    noise_ceiling_sub = {}
    data_dir_nc = os.path.join(args.bmd_dir, 'derivatives', 'versionB',
        'fsaverage', 'GLM', f'sub-{fmri_sub:02}', 'prepared_betas')
    for hemi in args.fmri_hemis:
        file_name_nc = (f'sub-{fmri_sub:02}_noiseceiling_space-fsaverage_'
            f'task-test_hemi-{hemi}_n-10.pkl')
        f = open(os.path.join(data_dir_nc, file_name_nc), 'rb')
        nc = pickle.load(f)[1].astype(np.float32)
        if hemi == 'left':
            noise_ceiling_sub['hemi-left'] = nc
        elif hemi == 'right':
            noise_ceiling_sub['hemi-right'] = nc
        del f, nc
    noise_ceiling.append(noise_ceiling_sub)
    del noise_ceiling_sub

    # Load the ROI masks
    data_dir_rois = os.path.join(args.emd_dir, 'results',
        'eeg_fmri_encoding_fusion', 'bmd_fsaverage_rois')
    file_name_roi = f'fsaverage_rois_sub-{fmri_sub:02}.npy'
    roi_mask = np.load(os.path.join(data_dir_rois, file_name_roi),
        allow_pickle=True).item()
    roi_masks.append(roi_mask)
    del roi_mask


# =============================================================================
# Average the results across vertices from the same ROIs and fMRI subjects
# =============================================================================
roi_correlation = {}

# Loop across ROIs, fMRI subjects, and hemispehres
rois = ['7AL', 'BA2', 'EBA', 'FFA', 'IPS0', 'IPS1-2-3', 'LOC', 'MT', 'OFA',
    'PFop', 'PFt', 'PPA', 'RSC', 'STS', 'TOS', 'V1', 'V2', 'V3', 'V3ab','hV4']
for roi in rois:
    corr = []
    for fs in range(len(args.fmri_subjects)):
        for h, hemi in enumerate(args.fmri_hemis):

            # Get the ROI indices for the fMRI subject and hemisphere
            if roi in ['V1', 'V2', 'V3']:
                idx_roi = np.append(
                    np.where(roi_masks[fs][f'hemi-{hemi}'][f'{roi}d'] == 1)[0],
                    np.where(roi_masks[fs][f'hemi-{hemi}'][f'{roi}v'] == 1)[0])
                idx_roi.sort()
            else:
                roi_idx = np.where(roi_masks[fs][f'hemi-{hemi}'][roi] == 1)[0]

            # Threshold the ROI indices based on the noise ceiling
            nc = noise_ceiling[fs][f'hemi-{hemi}']
            roi_idx = roi_idx[nc[roi_idx] >= args.noise_ceiling_threshold]

            # Append the ROI correlation results across hemispheres
            if h == 0:
                corr_sub = correlation[f'hemi-{hemi}'][:,fs,roi_idx]
            else:
                corr_sub = np.append(corr_sub,
                    correlation[f'hemi-{hemi}'][:,fs,roi_idx], 1)

        # Average the correlation results across ROI vertices, and append them
        # across fMRI subjects
        corr.append(np.mean(corr_sub, 1))
        del corr_sub

    # Average the correlation results of each ROI across fMRI subjects, and
    # store them in a dictionary
    roi_correlation[roi] = np.mean(corr, 0)
    del corr


# =============================================================================
# Compute the confidence intervals
# =============================================================================
ci_roi_correlation = {}
ci_dist = {}
n_times = correlation['hemi-left'].shape[3]
for roi in rois:
    ci_roi_correlation[roi] = np.zeros((2, n_times), dtype=np.float32)
    ci_dist[roi] = np.zeros((args.n_iter, n_times), dtype=np.float32)

for i in tqdm(range(args.n_iter)):
    idx = resample(np.arange(len(args.eeg_subjects)))
    for roi in rois:
        ci_dist[roi][i] = np.mean(ci_roi_correlation[roi][idx], 0)

for roi in rois:
    ci_roi_correlation[roi][0] = np.percentile(ci_dist[roi], 2.5, axis=0)
    ci_roi_correlation[roi][1] = np.percentile(ci_dist[roi], 97.5, axis=0)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'lh_correlation': np.mean(correlation['hemi-left'], (0, 1)),
    'rh_correlation': np.mean(correlation['hemi-right'], (0, 1)),
    'roi_correlation': roi_correlation,
    'ci_roi_correlation': ci_roi_correlation
}

save_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion',
    'stats')
os.makedirs(save_dir, exist_ok=True)

file_name = 'stats.npy'

np.save(os.path.join(save_dir, file_name), results)