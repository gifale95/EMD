"""Aggregate the the eye-tracking data pairwise decoding RDMs across time points
splits, and average them across pairwise comparisons.

Parameters
----------
sub : int
	Used subject.
tot_units : int
	Total of available data units.
mean_centering : str
	Whether to mean center the eye-tracking data using the 500ms prior to video
	onset ['baseline'], the entire epoch ['epoch'], or to not center the data
	al all ['none'].
sfreq : int
	Downsampling frequency.
zscore : int
	Whether to z-score [1] or not [0] the data.
features : str
	Whether to decode using the X- and Y-gaze features ['gaze'], the X- and
	Y-gaze features without the variance linearly explained by the pupil size
	['gaze_independent'], the pupil size ['pupil'], all features ['all'], or
	all deatures with linearly independent gaze and pupil size
	['all_independent'].
data_split : str
	Whether to decode the 'test' or 'train' split.
pseudo_trials : int
	If 1, create pseudo-trials.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int) # [1, 2]
parser.add_argument('--tot_units', default=1, type=int)
parser.add_argument('--mean_centering', default='baseline', type=str)
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--features', default='all', type=str) # ['gaze' 'gaze_independent' 'pupil' 'all' 'all_independent']
parser.add_argument('--data_split', default='train', type=str) # ['train', 'test']
parser.add_argument('--pseudo_trials', default=1, type=int)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()

print('>>> Merge pairwise decoding RDMs - Eye data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
time_splits = 10

for t in range(time_splits):
	data_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eye',
		'pairwise_decoding', 'sub-'+format(args.sub,'02'), 'sfreq-'+
		format(args.sfreq,'04'), 'data_split-'+args.data_split, 'mean_centering-'+
		args.mean_centering, 'features-'+args.features, 'zscore-'+
		format(args.zscore, '02'), 'pseudo_trials-'+
		format(args.pseudo_trials,'02'), 'pairwise_decoding_time_split-'+
		format(t+1, '02')+'.npy')
	data = np.load(data_dir, allow_pickle=True).item()
	decoding = data['pairwise_decoding']
	idx = np.tril_indices(len(decoding), -1)
	decoding = np.nanmean(decoding[idx], 0)
	if t == 0:
		times = data['times']
		pairwise_decoding = decoding
	else:
		pairwise_decoding = np.append(pairwise_decoding, decoding)
	del data


# =============================================================================
# Save the results
# =============================================================================
results_dict = {
	'pairwise_decoding': pairwise_decoding,
	'times': times
}

save_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eye',
	'pairwise_decoding_merged', 'sub-'+format(args.sub,'02'), 'sfreq-'+
	format(args.sfreq,'04'), 'data_split-'+args.data_split, 'mean_centering-'+
	args.mean_centering, 'features-'+args.features, 'zscore-'+
	format(args.zscore, '02'), 'pseudo_trials-'+
	format(args.pseudo_trials,'02'))
file_name = 'pairwise_decoding'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), results_dict)