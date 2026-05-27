"""Create the ROI RDMs using the Algonauts 2021 fMRI test data, all voxels.

Parameters
----------
sub : int
	Used subject.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr as corr
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/'
	'eeg_videos', type=str)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/'
	#'eeg_videos', type=str)
args = parser.parse_args()

print('>>> Create fMRI RDMs - All voxels data<<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# ROIs and TRs lists
# =============================================================================
rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA', 'OFA',
	'STS', 'LOC', 'PPA', 'RSC', 'TOS', 'V3ab', 'IPS0', 'IPS1-2-3']
if args.sub == 6:
	rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA', 'OFA',
		'STS', 'LOC', 'PPA', 'RSC', 'V3ab', 'IPS0', 'IPS1-2-3']
elif args.sub == 7:
	rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA', 'OFA',
		'STS', 'LOC', 'PPA', 'V3ab', 'IPS0', 'IPS1-2-3']
trs = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Save directory
save_dir = os.path.join(args.project_dir, 'results', 'fmri_rdms_all_voxels')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)


# =============================================================================
# Load the fMRI data
# =============================================================================
#parent_dir = '/home/ale/aaa_stuff/PhD/projects/eeg_videos/code/'+\
	#'02_data_quality_check/04_rsa_fusion/'
parent_dir = '/scratch/giffordale95/projects/algonauts_2021/fmri_time_2/'+\
	'fmri_dataset/preprocessed_data'

for r in tqdm(rois):
	for t in trs:
		data_dir = os.path.join(parent_dir, 'sub'+format(args.sub, '02'), 'TR'+\
			format(t, '02'), 'test_fmri_z=1_'+r+'.npy')
		data = np.load(data_dir)
		# Average the fMRI data across repetitions
		data = np.mean(data, 1)
		# Standardize the fMRI data
		sc = StandardScaler()
		data = sc.fit_transform(data)


# =============================================================================
# Build the RDM
# =============================================================================
		rdm = np.zeros((data.shape[0],data.shape[0]))
		for i1 in range(data.shape[0]):
			for i2 in range(data.shape[0]):
				if i1 < i2:
					rdm[i1,i2] = 1 - corr(data[i1], data[i2])[0]
					rdm[i2,i1] = rdm[i1,i2]


# =============================================================================
# Save the RDMs
# =============================================================================
		file_name = 'test_rdm_sub'+ format(args.sub, '02') + '_' + r + '_TR' + \
			format(t, '02')
		np.save(os.path.join(save_dir, file_name), rdm)