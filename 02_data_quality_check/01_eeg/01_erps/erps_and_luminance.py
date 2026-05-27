"""Plot the correlation between EEG ERPs and luminance.

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
from scipy.stats import pearsonr as corr
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2], type=list)
parser.add_argument('--baseline_correction', default=1, type=int) # ['0' '1']
parser.add_argument('--mvnn', default='time', type=str) # ['none' 'time']
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--lowpass', default=100, type=float)
parser.add_argument('--highpass', default=0.01, type=float)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
args = parser.parse_args()

#plt.savefig('erps_and_luminance_mvnn-none_baseline_corr-00', dpi=100)
#plt.savefig('erps_and_luminance_mvnn-none_baseline_corr-01', dpi=100)
#plt.savefig('erps_and_luminance_mvnn-time_baseline_corr-00', dpi=100)
#plt.savefig('erps_and_luminance_mvnn-time_baseline_corr-01', dpi=100)


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
# Retain only the video presentation time points
idx_start = np.where(times == 0)[0][0]
idx_end = np.where(times == 3)[0][0]
times = times[idx_start:idx_end]

eeg_data = []
presentation_order = []
for sub in range(len(preprocessed_data)):
	for s in range(len(preprocessed_data[sub])):
		if s == 0:
			eeg_data_sub = preprocessed_data[sub][s][:,:,idx_start:idx_end]
			pres_ord_sub = stimuli_presentation_order[sub][s]
		else:
			eeg_data_sub = np.append(eeg_data_sub,
				preprocessed_data[sub][s][:,:,idx_start:idx_end], 0)
			pres_ord_sub = np.append(pres_ord_sub,
				stimuli_presentation_order[sub][s], 0)
	eeg_data.append(np.asarray(eeg_data_sub))
	presentation_order.append(np.asarray(pres_ord_sub))
	del eeg_data_sub, pres_ord_sub
del preprocessed_data

# Average the EEG data across trials of the same video conditions
video_conditions = np.unique(presentation_order[0])
avg_eeg = np.zeros((len(args.subjects), len(video_conditions), len(ch_names),
	len(times)))
for s in range(len(eeg_data)):
	for v, video in enumerate(video_conditions):
		idx = np.where(presentation_order[s] == video)[0]
		avg_eeg[s,v] = np.mean(eeg_data[s][idx], 0)
del eeg_data


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
# Load the videos luminance and perceived lightness
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'video_frames_stats',
	'perceived_lightness', 'luminance_perceived-lightness.npy')
data_dict = np.load(data_dir, allow_pickle=True).item()

# Normalize the video frames to the eyetracking times
luminance = np.zeros((len(data_dict['luminance']), len(times)))
perceived_lightness = np.zeros((luminance.shape))
luminance_gray_screen = data_dict['luminance_gray_screen']
perceived_lightness_gray_screen = data_dict['perceived_lightness_gray_screen']
for v in range(luminance.shape[0]):
	t_frames = np.zeros(len(data_dict['luminance'][v+1]))
	for f in range(len(data_dict['luminance'][v+1])):
		t_frames[f] = f / len(data_dict['luminance'][v+1])
	for t in range(luminance.shape[1]):
		t_pupil = t / luminance.shape[1]
		idx = (np.abs(t_frames - t_pupil)).argmin()
		luminance[v,t] = data_dict['luminance'][v+1][idx]
		perceived_lightness[v,t] = data_dict['perceived_lightness'][v+1][idx]


# =============================================================================
# Correlate the EEG ERPs with luminance
# =============================================================================
eeg_lum_corr = np.zeros((avg_eeg.shape[0], avg_eeg.shape[2], avg_eeg.shape[3]))
for s in range(avg_eeg.shape[0]):
	for c in range(avg_eeg.shape[2]):
		for t in range(avg_eeg.shape[3]):
			eeg_lum_corr[s,c,t] = corr(avg_eeg[s,:,c,t], luminance[:,t])[0]


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
# Plot the data
# =============================================================================
fig, axs = plt.subplots(2, 1, sharex=True)
axs = np.reshape(axs, (-1))
for s in range(len(axs)):
	# Plot baseline dashed lines
	axs[s].plot([0, 3], [0, 0], 'k--', label='_nolegend_', linewidth=4)
	# Plot the correlation between pupil size and luminance
	for c in idx_ch:
		axs[s].plot(times, np.mean(eeg_lum_corr[s,c], 0), linewidth=2)
	# Other plot parameters
	if s == 1:
		axs[s].set_xlabel('Time (s)', fontsize=20)
		axs[s].set_xlim(left=min(times), right=max(times))
		xticks = np.arange(0, 3.01, .1)
		xlabels = np.round(np.arange(0, 3.01, .1), 1)
		plt.xticks(ticks=xticks, labels=xlabels)
		plt.legend(channel_types_names, fontsize=20, loc=8, ncol=6,
			frameon=True, bbox_to_anchor=(0.5, -.3))
	axs[s].set_ylabel('Pearson\'s $r$', fontsize=20)
	title = 'Correlation between EEG ERPs and luminance, subject ' + str(s+1)
	axs[s].set_title(title, fontsize=20)

#plt.savefig('erps_and_luminance_mvnn-none_baseline_corr-00', dpi=100)
#plt.savefig('erps_and_luminance_mvnn-none_baseline_corr-01', dpi=100)
#plt.savefig('erps_and_luminance_mvnn-time_baseline_corr-00', dpi=100)
#plt.savefig('erps_and_luminance_mvnn-time_baseline_corr-01', dpi=100)