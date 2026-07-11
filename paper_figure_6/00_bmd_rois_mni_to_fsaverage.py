"""Map the BOLD Moments Dataset (BMD) ROIs from MNI-152 space to fsaverage
space.

Parameters
----------
subjects : list
    List of BMD subject numbers.
fmri_hemis : list
    List of fMRI hemispheres.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).
bmd_dir : str
    Directory of the fMRI Moments Dataset (BMD).

"""

import argparse
import os
import numpy as np
import pickle
from tqdm import tqdm
import nibabel as nib
from neuromaps import transforms


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=list)
parser.add_argument('--fmri_hemis', default=['l', 'r'], type=list)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
parser.add_argument('--bmd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset/bold_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Map BMD ROIs <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Create the save directory
# =============================================================================
save_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion',
    'bmd_fsaverage_rois')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Transform the ROIs from MNI space to fsaverage space
# =============================================================================
# Get the ROIs
rois = ['7AL', 'BA2', 'EBA', 'FFA', 'IPS0', 'IPS1-2-3', 'LOC', 'MT', 'OFA',
    'PFop', 'PFt', 'PPA', 'RSC', 'STS', 'TOS', 'V1d', 'V1v', 'V2d', 'V2v',
    'V3ab', 'V3d', 'V3v', 'hV4']

# Loop over subjects
for sub in tqdm(args.fmri_subjects):

    # Get the volume shape, affine, and header
    file_dir = os.path.join(args.bmd_dir, 'derivatives', 'versionB', 'MNI152',
        'GLM', f'sub-{sub:02d}', 'ROIs',
        f'ROI-{args.fmri_hemis[0]}{rois[0]}_indices.pkl')
    file = open(file_dir, 'rb')
    vol_img = pickle.load(file)[2]
    vol_shape = vol_img.shape
    vol_affine = vol_img.affine
    vol_header = vol_img.header

    # Loop over ROIs
    lh_fsaverage_rois = {}
    rh_fsaverage_rois = {}
    for roi in rois:

        # Loop over hemispheres
        for h, hemi in enumerate(args.fmri_hemis):

            # Load the ROI voxels in MNI space
            file_dir = os.path.join(args.bmd_dir, 'derivatives', 'versionB',
                'MNI152', 'GLM', f'sub-{sub:02d}', 'ROIs',
                f'ROI-{hemi}{roi}_indices.pkl')
            file = open(file_dir, 'rb')
            if h == 0:
                roi_voxels = pickle.load(file)[0]
            else:
                roi_voxels = np.append(roi_voxels, pickle.load(file)[0], 0)

        # Create the ROI mask nibabel image
        vol_roi = np.zeros(vol_shape)
        for v in range(len(roi_voxels)):
            vol_roi[roi_voxels[v,0],roi_voxels[v,1],roi_voxels[v,2]] = 1
        vol_img_roi = nib.Nifti1Image(vol_roi, vol_affine, vol_header)
        del roi_voxels, vol_roi

        # Map the ROI mask from MNI space to fsaverage surface space
        lh, rh = transforms.mni152_to_fsaverage(vol_img_roi,
            fsavg_density='164k')
        del vol_img_roi

        # Threshold the ROI mask to create binary masks
        threshold = 0.5
        lh_data = lh.darrays[0].data  # Shape: (n_vertices,)
        rh_data = rh.darrays[0].data
        lh_mask = (lh_data >= threshold).astype(np.uint8)
        rh_mask = (rh_data >= threshold).astype(np.uint8)
        del lh, rh, lh_data, rh_data

        # Store the ROI masks in a dictionary
        lh_fsaverage_rois[roi] = lh_mask
        rh_fsaverage_rois[roi] = rh_mask


# =============================================================================
# Save the ROI masks
# =============================================================================
    fsaverage_rois = {
        'hemi-left': lh_fsaverage_rois,
        'hemi-right': rh_fsaverage_rois
    }
    file_name = f'fsaverage_rois_sub-{sub:02d}.npy'
    np.save(os.path.join(save_dir, file_name), fsaverage_rois)
    del fsaverage_rois, lh_fsaverage_rois, rh_fsaverage_rois