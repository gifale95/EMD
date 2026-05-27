"""Plot the eyetrack pairwise decoding results.

Parameters
----------
used_subs : list
	List of used subjects.
mean_centering : str
	Whether to mean center the eye-tracking data using the 500ms prior to video
	onset ['baseline'], the entire epoch ['epoch'], or to not center the data
	al all ['none'].
sfreq : int
	Downsampling frequency.
zscore : int
	Whether to z-score [1] or not [0] the data.
features : list
	List with used eye-tracking features during the decoding.
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
parser.add_argument('--mean_centering', default='baseline', type=str)
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--trials', default='all', type=str)
parser.add_argument('--features', default=['gaze', 'gaze_independent', 'pupil',
	'all', 'all_independent'], type=list)
parser.add_argument('--data_split', default='test', type=str) # ['test', 'train']
parser.add_argument('--pseudo_trials', default=1, type=int)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
args = parser.parse_args()


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
pairwise_decoding = {}
for s in args.used_subs:
	pair_dec = []
	for f in args.features:
		data_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eye',
			'pairwise_decoding_merged', 'sub-'+format(s,'02'), 'sfreq-'+
			format(args.sfreq,'04'), 'data_split-'+args.data_split, 'mean_centering-'+
			args.mean_centering, 'features-'+f, 'zscore-'+
			format(args.zscore, '02'), 'pseudo_trials-'+
			format(args.pseudo_trials,'02'), 'pairwise_decoding.npy')
		# Load the data
		data_dict = np.load(data_dir, allow_pickle=True).item()
		pair_dec.append(data_dict['pairwise_decoding'])
		times = data_dict['times']
	pairwise_decoding[s] = pair_dec
	del pair_dec, data_dict


# =============================================================================
# Plot parameters
# =============================================================================
# Setting the plot parameters
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 20
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
matplotlib.rcParams['axes.linewidth'] = 4
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.grid'] = True


# =============================================================================
# Plot the decoding results of single subjects
# =============================================================================
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))
for s, sub in enumerate(args.used_subs):
	# Plot the chance level daashed lines
	axs[s].plot([-10, 10], [.5, .5], 'k--', [0, 0], [100, -100], 'k--',
		[3, 3], [100, -100], 'k--', label='_nolegend_', linewidth=4)
	# Plot the pairwise decoding results
	for f in range(len(args.features)):
		axs[s].plot(times, pairwise_decoding[sub][f], linewidth=4)
	# Other plot parameters
	if s == 0:
		axs[s].legend(args.features, fontsize=20, loc=0, ncol=3, frameon=True)
	if s in [1]:
		axs[s].set_xlabel('Time (s)', fontsize=20)
		xticks = np.arange(-.2, 3.51, .2)
		xlabels = np.round(np.arange(-.2, 3.51, .2), 1)
		plt.xticks(ticks=xticks, labels=xlabels, fontsize=20)
	if s in [0, 1]:
		axs[s].set_ylabel('Decoding\naccuracy (%)', fontsize=20)
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=.47, top=.75)
	title = 'Subject ' + str(sub) + ', data_split-' + args.data_split
	axs[s].set_title(title, fontsize=20)

#plt.savefig('eye_decoding_mean_centering-baseline_data_split-test', dpi=100)
#plt.savefig('eye_decoding_mean_centering-baseline_data_split-train', dpi=100)