"""Plot the pairwise decoding results.

Parameters
----------
subjects : list
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
parser.add_argument('--subjects', default=[1, 2], type=list)
parser.add_argument('--baseline_correction', default=[0, 1], type=list)
parser.add_argument('--mvnn', default=['none', 'time'], type=list)
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--lowpass', default=100, type=float)
parser.add_argument('--highpass', default=[0.001, 0.01, 0.1, .5], type=list)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--channels', default=['O', 'P', 'T', 'C', 'F', 'all'], type=list)
parser.add_argument('--data_split', default='test', type=str) # ['train', 'test']
parser.add_argument('--pseudo_trials', default=1, type=int)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
args = parser.parse_args()


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
pairwise_decoding = {}
for s, sub in enumerate(args.subjects):
	for b, base in enumerate(args.baseline_correction):
		for n, mvnn in enumerate(args.mvnn):
			for h, high in enumerate(args.highpass):
				for c, chan in enumerate(args.channels):
					data_dir = os.path.join(args.project_dir, 'results', 'dataset_02', 'eeg',
						'pairwise_decoding_merged', 'sub-'+format(sub,'02'), 'mvnn-'+mvnn,
						'baseline_correction-'+format(base, '02'),
						'highpass-'+format(high)+'_lowpass-'+format(args.lowpass),
						'sfreq-'+format(args.sfreq,'04'), 'data_split-'+args.data_split,
						'channels-'+chan, 'zscore-'+format(args.zscore, '02'),
						'pseudo_trials-'+format(args.pseudo_trials,'02'), 'pairwise_decoding.npy')
					data_dict = np.load(data_dir, allow_pickle=True).item()
					pairwise_decoding['sub-'+str(sub)+'_base-'+str(base)+'_mvnn-'+mvnn+'_high-'+format(high)+'_chan-'+chan] = \
						data_dict['pairwise_decoding']
					times = data_dict['times']


# =============================================================================
# Plot parameters
# =============================================================================
# Setting the plot parameters
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 20
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
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
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))
for s, sub in enumerate(args.used_subs):
	# Plot the chance and stimulus onset dashed lines
	axs[s].plot([-10, 10], [.5, .5], 'k--', [0, 0], [100, -100], 'k--',
		[3, 3], [100, -100], 'k--', label='_nolegend_', linewidth=4)
	# Plot the decoding results
	for c, chan in enumerate(args.channels):
		axs[s].plot(times, pairwise_decoding[s,c], linewidth=4)
	# Other plot parameters
	if s in [1]:
		axs[s].set_xlabel('Time (s)', fontsize=20)
		xticks = np.arange(-2, 6.1, .25)
		xlabels = np.arange(-2, 6.1, .25)
		plt.xticks(ticks=xticks, labels=xlabels, fontsize=20)
	if s in [0, 1]:
		axs[s].set_ylabel('Decoding\naccuracy (%)', fontsize=20)
	if s in [0]:
		axs[s].legend(channel_types_names, fontsize=20, loc=9, ncol=7,
			frameon=True, bbox_to_anchor=(0.5, -1.35))
	axs[s].set_xlim(left=min(times), right=max(times))
	axs[s].set_ylim(bottom=.4, top=.8)
	tit = 'Sub-' + format(sub) + ', data_split-' + args.data_split + \
		', MVNN-' + args.mvnn + ', baseline_correction-' + \
		str(args.baseline_correction) + ', zscore-' + format(args.zscore) + \
		', pseudo_trials-' + format(args.pseudo_trials) + \
		', sequence_trials-' + args.trials
	axs[s].set_title(tit, fontsize=20)

#plt.savefig('decoding_dsplit-test_mvnn-time_baselineCorr-01_zscore-01_pseudoTrials-01_sequenceTrials-all', dpi=100)
#plt.savefig('decoding_dsplit-train_mvnn-time_baselineCorr-01_zscore-01_pseudoTrials-01_sequenceTrials-all', dpi=100)