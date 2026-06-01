"""Plot the pairwise decoding results.

Parameters
----------
subjects : list
	List of used subjects.
sfreq : int
	Downsampling frequency.
channels : list
	List of all channel types used for decoding.
data_split : str
	Whether to decode the 'test' or 'train' split.
pseudo_trials : int
	If 1, and data_split=='test', create pseudo-trials.
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
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6], type=list)
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--channels', default=['O', 'P', 'T', 'C', 'F', 'all'], type=list)
parser.add_argument('--data_split', default='test', type=str)
parser.add_argument('--pseudo_trials', default=1, type=int)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
parser.add_argument('--project_dir', default='/home/ale/scratch/projects/eeg_videos', type=str)
args = parser.parse_args()


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
pairwise_decoding = []
for s, sub in enumerate(args.subjects):
	deocoding_sub = []
	for c, chan in enumerate(args.channels):
		data_dir = os.path.join(args.project_dir, 'results', 'dataset_02',
			'eeg', 'pairwise_decoding_merged', 'sub-'+format(sub,'02'),
			'sfreq-'+format(args.sfreq,'04'), 'data_split-'+args.data_split,
			'channels-'+chan, 'pseudo_trials-'+
			format(args.pseudo_trials,'02'), 'pairwise_decoding.npy')
		# Load the data
		data_dict = np.load(data_dir, allow_pickle=True).item()
		if s == 0 and c == 0:
			times = data_dict['times']
		deocoding_sub.append(data_dict['pairwise_decoding']*100)
	pairwise_decoding.append(deocoding_sub)
del deocoding_sub
pairwise_decoding = np.asarray(pairwise_decoding)


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 15
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 3
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
channel_types_names = ['Occipital', 'Parietal', 'Temporal', 'Central',
	'Frontal', 'All']
fig, axs = plt.subplots(3, 2, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))
for s, sub in enumerate(args.subjects):
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [50, 50], 'k--', [0, 0], [100, -100], 'k--',
		[3, 3], [100, -100], 'k--', label='_nolegend_', linewidth=3)
	# Plot the decoding results
	for c, chan in enumerate(args.channels):
		axs[s].plot(times, pairwise_decoding[s,c], linewidth=3)
	# Other plot parameters
	if s in [4, 5]:
		axs[s].set_xlabel('Time (s)', fontsize=fontsize)
		xticks = np.arange(-.2, 3.51, .2)
		xlabels = np.round(np.arange(-.2, 3.51, .2), 1)
		plt.xticks(ticks=xticks, labels=xlabels, fontsize=fontsize)
	if s in [0, 2, 4]:
		axs[s].set_ylabel('Decoding\naccuracy (%)', fontsize=fontsize)
	if s in [0]:
		axs[s].legend(channel_types_names, fontsize=fontsize, loc=9, ncol=7,
			frameon=True, bbox_to_anchor=(1, -2.6))
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=45, top=90)
	tit = 'Sub-' + format(sub) + ', data_split-' + args.data_split
# 	tit = 'Sub-' + format(sub) + ', data_split-' + args.data_split + \
# 		', MVNN-' + args.mvnn + ', baseline_correction-' + \
# 		str(args.baseline_correction)
	axs[s].set_title(tit, fontsize=fontsize)

#plt.savefig('eeg_moments_pairwise_decoding_dsplit-test', dpi=100)
#plt.savefig('eeg_moments_pairwise_decoding_dsplit-train', dpi=100)