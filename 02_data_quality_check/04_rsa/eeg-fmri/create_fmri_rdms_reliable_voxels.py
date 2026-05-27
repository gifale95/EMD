"""Create the ROI RDMs using the Algonauts 2021 fMRI test data, reliable voxels.

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
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr as corr
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=5, type=int)
#parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/'
	#'eeg_videos', type=str)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/'
	'eeg_videos', type=str)
args = parser.parse_args()

print('>>> Create fMRI RDMs - Reliable voxels <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# ROIs and TRs lists
# =============================================================================
rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA', 'OFA',
	'STS', 'LOC', 'PPA', 'RSC', 'TOS', 'V3ab', 'IPS0', 'IPS1-2-3', '7AL', 'BA2',
	'PFt', 'PFop']
if args.sub == 6:
	rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA',
		'OFA', 'STS', 'LOC', 'PPA', 'RSC', 'V3ab', 'IPS0', 'IPS1-2-3', '7AL',
		'BA2', 'PFt', 'PFop']
elif args.sub == 7:
	rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA',
		'OFA', 'STS', 'LOC', 'PPA', 'V3ab', 'IPS0', 'IPS1-2-3', '7AL', 'BA2',
		'PFt', 'PFop']
trs = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Save directory
save_dir = os.path.join(args.project_dir, 'results',
	'fmri_rdms_reliable_voxels')
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)


# =============================================================================
# Load the fMRI data
# =============================================================================
parent_dir = '/media/ale/Elements/PhD/projects/finished/002_algonauts_2021/' + \
	'fmri_dataset'
#parent_dir = '/scratch/giffordale95/projects/algonauts_2021/fmri_time_2/'+\
	#'fmri_dataset/'

def load_dict(filename_):
	with open(filename_, 'rb') as f:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		ret_di = u.load()
	return ret_di

for r in tqdm(rois):
	for t in trs:

		### Load the fMRI reliable training data ###
		file_dir = os.path.join(parent_dir, 'reliable_preprocessed_data',
			'sub'+format(args.sub,'02'), 'testing',
			r+'_TRavg'+str(t)+'_testing.pkl')
		data = load_dict(file_dir)['test_data']

		### Load the original data if no voxels ###
		if data.shape[2] < 2:
			file_dir = os.path.join(parent_dir, 'preprocessed_data',
				'sub'+format(args.sub,'02'), 'TR'+format(t,'02'),
				'test_fmri_z=1_'+r+'.npy')
			data = np.load(file_dir)

		### Average the fMRI training data across repetitions ###
		data = np.mean(data, 1)

		### Standardize the fMRI data ###
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
