"""Plot the temporal generalization analysis results.

Parameters
----------
used_subs : list
		List of used subjects.
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
pseudo_trials : int
	If 1, create pseudo-trials.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--used_subs', default=[1, 2], type=list)
parser.add_argument('--baseline_correction', default=1, type=int)
parser.add_argument('--mvnn', default='time', type=str) # ['none' 'time']
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--lowpass', default=100, type=float)
parser.add_argument('--highpass', default=0.01, type=float)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--channels', default='OP', type=str) # ['OP' 'all']
parser.add_argument('--data_split', default='test', type=str)
parser.add_argument('--pseudo_trials', default=1, type=int)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
args = parser.parse_args()


# =============================================================================
# Load the temporal generalization analysis results
# =============================================================================
for s, sub in enumerate(args.used_subs):
	data_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eeg',
		'temporal_generalization_analysis_merged', 'sub-'+format(sub,'02'),
		'mvnn-'+args.mvnn, 'baseline_correction-'+format(args.baseline_correction, '02'),
		'highpass-'+format(args.highpass)+'_lowpass-'+format(args.lowpass),
		'sfreq-'+format(args.sfreq,'04'), 'data_split-'+args.data_split,
		'channels-'+args.channels, 'zscore-'+format(args.zscore, '02'),
		'pseudo_trials-'+format(args.pseudo_trials,'02'),
		'temporal_generalization_analysis.npy')
	# Load the data
	data_dict = np.load(data_dir, allow_pickle=True).item()
	if s == 0:
		temp_gen_matrix = np.expand_dims(
			data_dict['temp_gen_matrix'], 0)
		times = data_dict['times']
	else:
		temp_gen_matrix = np.append(temp_gen_matrix, np.expand_dims(
			data_dict['temp_gen_matrix'], 0), 0)

temp_gen_matrix = np.mean(np.flip(temp_gen_matrix, 1), 0)
#temp_gen_matrix = np.flip(temp_gen_matrix[0], 0)


# =============================================================================
# Plot parameters
# =============================================================================
# Setting the plot parameters
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 30
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
matplotlib.rcParams['axes.linewidth'] = 3
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False


# =============================================================================
# Plot the temporal generalization results
# =============================================================================
# Plot the data
plt.figure()
#img = plt.imshow(temp_gen_matrix, aspect='auto')
img = plt.imshow(temp_gen_matrix)
# Plot parameters
plt.xlabel('Test time (s)', fontsize=30)
plt.ylabel('Train time (s)', fontsize=30)
xticks = np.asarray((10, 60, 110, 160))
xlabels = [0, 1, 2, 3]
yticks = abs(xticks - len(times))
ylabels = [0, 1, 2, 3]
plt.xticks(ticks=xticks, labels=xlabels)
plt.yticks(ticks=yticks, labels=ylabels)
plt.title('Subject average', fontsize=30)
plt.colorbar(img, label='Decoding accuracy (%)', fraction=0.02)

#plt.savefig('temp_gen_analysis_dataset-02_dsplit-test_mvnn-none_chan-all_subj-avg', dpi=100)
#plt.savefig('temp_gen_analysis_dataset-02_dsplit-test_mvnn-none_chan-OP_subj-avg', dpi=100)
#plt.savefig('temp_gen_analysis_dataset-02_dsplit-test_mvnn-time_chan-all_subj-avg', dpi=100)
#plt.savefig('temp_gen_analysis_dataset-02_dsplit-test_mvnn-time_chan-OP_subj-avg', dpi=100)