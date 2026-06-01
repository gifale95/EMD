"""Average the EEG data pairwise decoding RDMs across pairwise comparisons.

Parameters
----------
sub : int
	Used subject.
sfreq : int
	Downsampling frequency.
channels : str
	Whether to retain occipital ['O'], posterior ['P'], temporal ['T'],
	central ['C'], frontal ['F'] occipital/parital ['OP'] or all ['all']
	channels.
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
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--channels', default='all', type=str) # ['O', 'P', 'T', 'C', 'F', 'all']
parser.add_argument('--data_split', default='test', type=str) # ['train', 'test']
parser.add_argument('--pseudo_trials', default=1, type=int)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()

print('>>> Pairwise decoding RDM merging <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
time_splits = 10

for t in range(time_splits):
	data_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eeg',
		'pairwise_decoding', 'sub-'+format(args.sub,'02'), 'sfreq-'+
		format(args.sfreq,'04'), 'data_split-'+args.data_split, 'channels-'+
		args.channels, 'pseudo_trials-'+format(args.pseudo_trials,'02'),
		'pairwise_decoding_time_split-'+format(t+1, '02')+'.npy')
	data = np.load(data_dir, allow_pickle=True).item()
	decoding = data['pairwise_decoding']
	idx = np.tril_indices(len(decoding), -1)
	decoding = np.mean(decoding[idx], 0)
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

save_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eeg',
	'pairwise_decoding_merged', 'sub-'+format(args.sub,'02'), 'sfreq-'+
	format(args.sfreq,'04'), 'data_split-'+args.data_split, 'channels-'+
	args.channels, 'pseudo_trials-'+format(args.pseudo_trials,'02'))
file_name = 'pairwise_decoding'
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), results_dict)
