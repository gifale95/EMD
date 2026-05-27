"""RSA fusion between the EEG and eye-tracking data RDMs.

Parameters
----------
sub : int
	Used subject.
tot_units : int
	Total of available data units.
baseline_correction : int
	Whether to baseline correct [1] or not [0] the EEG data.
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
trials : str
	Whether to decode the data of the 'first' sequence, 'second' sequence, or
	'all' trials.
amount_channels : int
	Amount of used EEG channel groups. If '1' use 32 channels. If '2' use 64
	channels. If '3' use 96 channels. If '4' use 128 channels.
channels : str
	Whether to retain occipital ['O'], posterior ['P'], temporal ['T'],
	central ['C'], frontal ['F'] occipital/parital ['OP'] or all ['all']
	channels.
data_split : str
	Whether to decode the 'test' or 'train' split.
mean_centering : str
	Whether to mean center the eye-tracking data using the 500ms prior to video
	onset ['baseline'], the entire epoch ['epoch'], or to not center the data
	al all ['none'].
features : str
	Whether to decode 'gaze', 'pupil' or 'all' eye-tracking data features.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr as corr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int) # ['1' '2']
parser.add_argument('--tot_units', default=1, type=int)
parser.add_argument('--baseline_correction', default=1, type=int)
parser.add_argument('--mvnn', default='time', type=str)
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--lowpass', default=40, type=float)
parser.add_argument('--highpass', default=.5, type=float)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--trials', default='all', type=str)
parser.add_argument('--amount_channels', default=4, type=int)
parser.add_argument('--channels', default='all', type=str) # ['O' 'P' 'T' 'C' 'F' 'all']
parser.add_argument('--data_split', default='test', type=str) # ['train' 'test']
parser.add_argument('--pseudo_trials', default=1, type=int)
parser.add_argument('--mean_centering', default='baseline', type=str)
parser.add_argument('--features', default='all', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
args = parser.parse_args()

print('>>> RSA EEG-eye <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the EEG RDMs
# =============================================================================
if args.data_split == 'train':
	pseudo_trials = 0
elif args.data_split == 'test':
	pseudo_trials = 1

time_splits = 10

for t in range(time_splits):
	if args.data_split == 'train':
		data_dir = os.path.join(args.project_dir, 'results', 'eeg_data_quality_check',
			'pairwise_decoding', 'sub-'+format(args.sub,'02'), 'mvnn-'+args.mvnn,
			'baseline_correction-'+format(args.baseline_correction, '02'),
			'highpass-'+format(args.highpass)+'_lowpass-'+format(args.lowpass),
			'sfreq-'+format(args.sfreq,'04'), 'data_split-'+args.data_split,
			'sequence_trials-'+args.trials, 'channels-'+args.channels, 'zscore-'+
			format(args.zscore, '02'), 'pseudo_trials-'+
			format(pseudo_trials,'02'), 'pairwise_decoding_time_split-'+
			format(t+1, '02')+'chans-'+format(args.amount_channels,'02')+'.npy')
	elif args.data_split == 'test':
		data_dir = os.path.join(args.project_dir, 'results', 'eeg_data_quality_check',
			'pairwise_decoding', 'sub-'+format(args.sub,'02'), 'mvnn-'+args.mvnn,
			'baseline_correction-'+format(args.baseline_correction, '02'),
			'highpass-'+format(args.highpass)+'_lowpass-'+format(args.lowpass),
			'sfreq-'+format(args.sfreq,'04'), 'data_split-'+args.data_split,
			'sequence_trials-'+args.trials, 'channels-'+args.channels, 'zscore-'+
			format(args.zscore, '02'), 'pseudo_trials-'+
			format(pseudo_trials,'02'), 'pairwise_decoding_time_split-'+
			format(t+1, '02')+'_chans-'+format(args.amount_channels,'02')+'.npy')
	data = np.load(data_dir, allow_pickle=True).item()
	decoding = data['pairwise_decoding']
	idx = np.tril_indices(len(decoding), -1)
	decoding = decoding[idx]
	if t == 0:
		times = data['times']
		eeg_rdm = decoding
	else:
		eeg_rdm = np.append(eeg_rdm, decoding, 1)
	del data


# =============================================================================
# Load the eye-tracking RDMs
# =============================================================================
time_splits = 10

for t in range(time_splits):
	data_dir = os.path.join(args.project_dir, 'results',
		'eye_data_quality_check', 'pairwise_decoding', 'sub-'+
		format(args.sub,'02'), 'sfreq-'+format(args.sfreq,'04'), 'data_split-'+
		args.data_split, 'mean_centering-'+args.mean_centering,
		'sequence_trials-'+args.trials, 'features-'+args.features, 'zscore-'+
		format(args.zscore, '02'), 'pseudo_trials-'+
		format(pseudo_trials,'02'), 'pairwise_decoding_time_split-'+
		format(t+1, '02')+'.npy')
	data = np.load(data_dir, allow_pickle=True).item()
	decoding = data['pairwise_decoding']
	idx = np.tril_indices(len(decoding), -1)
	decoding = decoding[idx]
	if t == 0:
		times = data['times']
		eye_rdm = decoding
	else:
		eye_rdm = np.append(eye_rdm, decoding, 1)
	del data


# =============================================================================
# Perform RSA
# =============================================================================
# Empty RSA 2D array of shape: (EEG times × Eye-tracking times)
rsa = np.zeros((eeg_rdm.shape[1],eye_rdm.shape[1]))

for t1 in tqdm(range(eeg_rdm.shape[1])):
	for t2 in range(eye_rdm.shape[1]):
		if sum(np.isnan(eye_rdm[:,t2])) == 0:
			rsa[t1,t2] = corr(eeg_rdm[:,t1], eye_rdm[:,t2])[0]
		else:
			rsa[t1,t2] = np.nan


# =============================================================================
# Save the results
# =============================================================================
results_dict = {
	'rsa': rsa,
	'times': times,
}

save_dir = os.path.join(args.project_dir, 'results',
	'combined_data_quality_check', 'rsa_eeg-eye', 'sub-'+format(args.sub,'02'),
	'data_split-'+args.data_split, 'channels-'+args.channels)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
file_name = 'rsa_eeg-eye'
np.save(os.path.join(save_dir, file_name), results_dict)
