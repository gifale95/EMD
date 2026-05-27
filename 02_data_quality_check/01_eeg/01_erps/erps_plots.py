"""ERPs of the EEG responses to videos.

Parameters
----------
subjects : list
	Used subjects.
baseline_correction : int
	Whether to baseline correct [1] or not [0] the data.
mvnn : str
	Type of MVNN applied to preprocess the data.
sfreq : int
	Downsampling frequency.
lowpass : float
	Lowpass filter frequency.
highpass : float
	Highpass filter frequency.
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
parser.add_argument('--baseline_correction', default=0, type=int) # ['0' '1']
parser.add_argument('--mvnn', default='none', type=str) # ['none' 'time']
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--lowpass', default=100, type=float)
parser.add_argument('--highpass', default=0.01, type=float)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
args = parser.parse_args()

# =============================================================================
# Load the biological EEG data
# =============================================================================
preprocessed_data = []
stimuli_presentation_order = []
for s in args.subjects:
	data_dir = os.path.join(args.project_dir, 'dataset', 'preprocessed_data',
		'dataset_02', 'eeg', 'sub-'+format(s,'02'), 'mvnn-'+args.mvnn,
		'baseline_correction-'+format(args.baseline_correction, '02'), 'highpass-'+
		format(args.highpass)+'_lowpass-'+format(args.lowpass), 'sfreq-'+
		format(args.sfreq,'04'), 'preprocessed_data.npy')
	data_dict = np.load(data_dir, allow_pickle=True).item()
	times = data_dict['times']
	ch_names = data_dict['ch_names']
	info = data_dict['info']
	preprocessed_data.append(data_dict['eeg_data'])
	stimuli_presentation_order.append(data_dict['stimuli_presentation_order'])
	del data_dict


# =============================================================================
# Select and average the EEG trials
# =============================================================================
eeg_data = []
for sub in range(len(preprocessed_data)):
	for s in range(len(preprocessed_data[sub])):
		if s == 0:
			eeg_data_sub = preprocessed_data[sub][s]
		else:
			eeg_data_sub = np.append(eeg_data_sub, preprocessed_data[sub][s], 0)
	eeg_data.append(np.mean(np.asarray(eeg_data_sub), 0))
	del eeg_data_sub
del preprocessed_data
eeg_data = np.asarray(eeg_data)


# =============================================================================
# EEG channels selection
# =============================================================================
channel_types = ['O', 'P', 'T', 'C', 'FA', 'all']
channel_types_names = ['Occipital', 'Parietal', 'Temporal', 'Central',
	'Frontal/Anterior', 'All']
idx_ch = []

# 'O', 'P', 'T', 'C' channel types
for ch_type in channel_types:
	idx = []
	if ch_type != 'FA' and ch_type != 'all':
		for c, chan in enumerate(ch_names):
			if ch_type in chan:
				idx.append(c)
		idx_ch.append(np.asarray(idx))

# 'FA' channel types
idx = []
for c, chan in enumerate(ch_names):
	if 'F' in chan or 'A' in chan:
		idx.append(c)
idx_ch.append(np.asarray(idx))

# 'all' channel types
idx_ch.append(np.arange(0, len(ch_names)))


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
# Plot the ERPs
# =============================================================================
fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
axs = np.reshape(axs, (-1))
for s, sub in enumerate(args.subjects):
	# Plot baseline and stimulus onset/offset dashed lines
	axs[s].plot([-5, 5], [0, 0], 'k--', [0, 0], [10, -10], 'k--', [3, 3], [10, -10],
		'k--', label='_nolegend_', linewidth=3)
	# Plot the ERPs
	min_val = 0
	max_val = 0
	for c in idx_ch:
		axs[s].plot(times, np.mean(eeg_data[s,c], 0), linewidth=2)
		if min(np.mean(eeg_data[s,c], 0)) < min_val:
			min_val = min(np.mean(eeg_data[s,c], 0))
		if max(np.mean(eeg_data[s,c], 0)) > max_val:
			max_val = max(np.mean(eeg_data[s,c], 0))
	# Plot parameters
	axs[s].set_ylabel('EEG voltage', fontsize=20)
	axs[s].set_ylim(bottom=min_val, top=max_val)
	title = 'Sub-' + format(sub) + ', MVNN-' + args.mvnn + \
		', baseline_correction-' + str(args.baseline_correction) + ', highpass-' + \
		str(args.highpass)
	axs[s].set_title(title, fontsize=20)
	if s == 1:
		axs[s].set_xlabel('Time (s)', fontsize=20)
		xticks = np.arange(-.2, 3.51, .2)
		xlabels = np.round(np.arange(-.2, 3.51, .2), 1)
		plt.xticks(ticks=xticks, labels=xlabels, fontsize=20)
		plt.xlim(left=min(times), right=max(times))
		plt.legend(channel_types_names, fontsize=20, loc=8, ncol=6,
			frameon=True, bbox_to_anchor=(0.5, -.3))

#plt.savefig('plot_erps_mvnn-none_baseline_corr-00_highpass-0.01', dpi=100)
#plt.savefig('plot_erps_mvnn-none_baseline_corr-01_highpass-0.01', dpi=100)
#plt.savefig('plot_erps_mvnn-time_baseline_corr-00_highpass-0.01', dpi=100)
#plt.savefig('plot_erps_mvnn-time_baseline_corr-01_highpass-0.01', dpi=100)