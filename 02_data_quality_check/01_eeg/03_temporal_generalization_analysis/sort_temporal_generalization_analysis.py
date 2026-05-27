"""Sort the pairwise decoding results.

Parameters
----------
sub : int
	Used subject.
baseline_correction : int
	Whether to baseline correct [1] or not [0] the data.
mvnn : str
	Type of MVNN applied to preprocess the data ['none', 'time'].
sfreq : int
	Downsampling frequency.
lowpass : float
	Lowpass filter frequency.
highpass : float
	Highpass filter frequency.
zscore : int
	Whether to z-score [1] or not [0] the data.
channels : str
	Whether to retain occipital ['O'], posterior ['P'], temporal ['T'],
	central ['C'], frontal ['F'] occipital/parital ['OP'] or all ['all']
	channels.
data_split : str
	Whether to decode the 'test' or 'train' split.
tot_time_splits : int
	Integer indicating the time points split used.
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
parser.add_argument('--sub', default=1, type=int) # ['1' '2']
parser.add_argument('--baseline_correction', default=1, type=int)
parser.add_argument('--mvnn', default='time', type=str) # ['none' 'time']
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--lowpass', default=100, type=float)
parser.add_argument('--highpass', default=0.01, type=float)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--channels', default='all', type=str) # ['OP' 'all']
parser.add_argument('--data_split', default='test', type=str)
parser.add_argument('--tot_time_splits', default=185, type=int)
parser.add_argument('--pseudo_trials', default=1, type=int)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()


# =============================================================================
# Load and format the temporal generalization analysis results
# =============================================================================
if args.data_split == 'test':
	videos = 102
if args.data_split == 'train':
	videos = 1000


data_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eeg',
	'temporal_generalization_analysis', 'sub-'+format(args.sub,'02'), 'mvnn-'+
	args.mvnn, 'baseline_correction-'+format(args.baseline_correction, '02'),
	'highpass-'+format(args.highpass)+'_lowpass-'+format(args.lowpass),
	'sfreq-'+format(args.sfreq,'04'), 'data_split-'+args.data_split,
	'channels-'+args.channels, 'zscore-'+format(args.zscore, '02'),
	'pseudo_trials-'+format(args.pseudo_trials,'02'))

for t in range(args.tot_time_splits):
	# Load the data
	file_dir = 'temporal_generalization_analysis_time_split-' + \
		format(t+1, '02') + '.npy'
	data_dict = np.load(os.path.join(data_dir, file_dir),
		allow_pickle=True).item()
	# Average across pairwise comparisons
	dec_res = np.mean(np.mean(data_dict['temporal_generalization'], 0), 0)

	if t == 0:
		temp_gen_matrix = dec_res
		times = data_dict['times']
		ch_names = data_dict['ch_names']
		info = data_dict['info']
	else:
		temp_gen_matrix = np.append(temp_gen_matrix, dec_res, 0)


# =============================================================================
# Save the results
# =============================================================================
results_dict = {
	'temp_gen_matrix': temp_gen_matrix,
	'times': times,
	'ch_names': ch_names,
	'info': info
}

save_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eeg',
	'temporal_generalization_analysis_merged', 'sub-'+format(args.sub,'02'), 'mvnn-'+
	args.mvnn, 'baseline_correction-'+format(args.baseline_correction, '02'),
	'highpass-'+format(args.highpass)+'_lowpass-'+format(args.lowpass),
	'sfreq-'+format(args.sfreq,'04'), 'data_split-'+args.data_split,
	'channels-'+args.channels, 'zscore-'+format(args.zscore, '02'),
	'pseudo_trials-'+format(args.pseudo_trials,'02'))
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
file_name = 'temporal_generalization_analysis'
np.save(os.path.join(save_dir, file_name), results_dict)