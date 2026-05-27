"""Plot the correlation between pupil size and luminance.

Parameters
----------
used_subj : list of int
	List of used subjects.
mean_centering : str
	Whether to mean center the eye-tracking data using the 500ms prior to video
	onset ['baseline'], the entire epoch ['epoch'], or to not center the data
	al all ['none'].
sfreq : int
	Downsampling frequency.
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
parser.add_argument('--used_subj', default=[1, 2], type=list)
parser.add_argument('--mean_centering', default='baseline', type=str)
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
args = parser.parse_args()


# =============================================================================
# Load the eyetrack data
# =============================================================================
eyetrack_data = {}
stim_pres_order = {}
for s in args.used_subj:
	data_dir = os.path.join(args.project_dir, 'dataset', 'preprocessed_data',
		'dataset_02', 'eye', 'sub-'+format(s,'02'), 'mean_centering-'+
		args.mean_centering, 'sfreq-'+format(args.sfreq,'04'),
		'preprocessed_data.npy')
	data_dict = np.load(data_dir, allow_pickle=True).item()
	eyetrack_times = data_dict['eyetrack_times']
	eyetrack_data[s] = data_dict['eyetrack_data']
	stim_pres_order[s] = data_dict['stimuli_presentation_order']
	del data_dict


# =============================================================================
# Select and average the EEG trials
# =============================================================================
eye_data_all = {}
stim_pres_order_all = {}
for s in args.used_subj:
	for ses in range(len(eyetrack_data[s])):
		if ses == 0:
			eye_data_all[s] = eyetrack_data[s][ses]
			stim_pres_order_all[s] = stim_pres_order[s][ses]
		else:
			eye_data_all[s] = np.append(eye_data_all[s], eyetrack_data[s][ses],
				0)
			stim_pres_order_all[s] = np.append(stim_pres_order_all[s],
				stim_pres_order[s][ses], 0)
del eyetrack_data, stim_pres_order

# Retain only the video presentation time points
idx_start = np.where(eyetrack_times == 0)[0][0]
idx_end = np.where(eyetrack_times == 3)[0][0]
eyetrack_times = eyetrack_times[idx_start:idx_end]

# Average the trials across video conditions
video_cond = np.unique((stim_pres_order_all[args.used_subj[0]]))
eye_data = np.zeros((len(args.used_subj), len(video_cond),
	len(eyetrack_times)))
for s, sub in enumerate(args.used_subj):
	for v, vid in enumerate(video_cond):
		idx = np.where(stim_pres_order_all[sub] == vid)[0]
		eye_data[s,v] = np.nanmean(
			eye_data_all[sub][idx,idx_start:idx_end,2], 0)


# =============================================================================
# Load the videos luminance and perceived lightness
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'video_frames_stats',
	'perceived_lightness', 'luminance_perceived-lightness.npy')
data_dict = np.load(data_dir, allow_pickle=True).item()

# Normalize the video frames to the eyetracking times
luminance = np.zeros((len(data_dict['luminance']), len(eyetrack_times)))
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
# Correlate pupil size with luminance
# =============================================================================
pupil_lum_corr = np.zeros((eye_data.shape[0], eye_data.shape[2]))
for s in range(eye_data.shape[0]):
	for t in range(eye_data.shape[2]):
		pupil_lum_corr[s,t] = corr(eye_data[s,:,t], luminance[:,t])[0]


# =============================================================================
# Correlate pupil size change with luminance change
# =============================================================================
# Compute the pupil size change and luminance change
eye_data_delta = np.zeros(eye_data.shape)
luminance_delta = np.zeros(luminance.shape)

for t in range(eye_data.shape[2]):
	if t == 0:
		luminance_delta[:,t] = luminance[:,t]
	else:
		luminance_delta[:,t] = luminance[:,t] - luminance[:,t-1]
	for s in range(eye_data.shape[0]):
		if t == 0:
			eye_data_delta[s,:,t] = eye_data[s,:,t]
		else:
			eye_data_delta[s,:,t] = eye_data[s,:,t] - eye_data[s,:,t-1]

# Correlate the pupil size change with the luminance change
pupil_delta_lum_corr = np.zeros((eye_data_delta.shape[0],
	eye_data_delta.shape[2]))
pupil_delta_lum_delta_corr = np.zeros((eye_data_delta.shape[0],
	eye_data_delta.shape[2]))
pupil_lum_delta_corr = np.zeros((eye_data_delta.shape[0],
	eye_data_delta.shape[2]))
for s in range(eye_data_delta.shape[0]):
	for t in range(eye_data_delta.shape[2]):
		pupil_delta_lum_corr[s,t] = corr(eye_data_delta[s,:,t],
			luminance[:,t])[0]
		pupil_delta_lum_delta_corr[s,t] = corr(eye_data_delta[s,:,t],
			luminance_delta[:,t])[0]
		pupil_lum_delta_corr[s,t] = corr(eye_data[s,:,t],
			luminance_delta[:,t])[0]


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
# Plot the data
# =============================================================================
fig, axs = plt.subplots(4, 1, sharex=True)
axs = np.reshape(axs, (-1))
for d in range(len(axs)):
	# Plot baseline dashed lines
	axs[d].plot([0, 3], [0, 0], 'k--', label='_nolegend_', linewidth=4)
	if d == 0:
		# Plot the correlation between pupil size and luminance
		for s in range(len(args.used_subj)):
			axs[d].plot(eyetrack_times, pupil_lum_corr[s], linewidth=4)
		# Other plot parameters
		axs[d].set_ylabel('Pearson\'s $r$', fontsize=20)
		title = 'Correlation between pupil size and luminance'
		axs[d].set_title(title, fontsize=20)
		legend = ['Subject 1', 'Subject 2']
		axs[d].legend(legend, fontsize=20, loc=0, ncol=1, frameon=True)
	elif d == 1:
		# Plot the correlation between pupil size change and luminance change
		for s in range(len(args.used_subj)):
			axs[d].plot(eyetrack_times, pupil_delta_lum_delta_corr[s],
				linewidth=4)
		# Other plot parameters
		axs[d].set_ylabel('Pearson\'s $r$', fontsize=20)
		title = 'Correlation between pupil size $\Delta$ and luminance $\Delta$'
		axs[d].set_title(title, fontsize=20)
	elif d == 2:
		# Plot the correlation between pupil size and luminance change
		for s in range(len(args.used_subj)):
			axs[d].plot(eyetrack_times, pupil_lum_delta_corr[s], linewidth=4)
		# Other plot parameters
		axs[d].set_ylabel('Pearson\'s $r$', fontsize=20)
		title = 'Correlation between pupil size and luminance $\Delta$'
		axs[d].set_title(title, fontsize=20)
	elif d == 3:
		# Plot the correlation between pupil size change and luminance
		for s in range(len(args.used_subj)):
			axs[d].plot(eyetrack_times, pupil_delta_lum_corr[s], linewidth=4)
		# Other plot parameters
		axs[d].set_ylabel('Pearson\'s $r$', fontsize=20)
		title = 'Correlation between pupil size $\Delta$ and luminance'
		axs[d].set_title(title, fontsize=20)
		axs[d].set_xlabel('Time (s)', fontsize=20)
		axs[d].set_xlim(left=min(eyetrack_times), right=max(eyetrack_times))
		xticks = np.arange(0, 3.01, .1)
		xlabels = np.round(np.arange(0, 3.01, .1), 1)
		plt.xticks(ticks=xticks, labels=xlabels)

#plt.savefig('pupil_size_and_luminance_dataset-02', dpi=100)


# =============================================================================
# Plot the data (old)
# =============================================================================
# fig, axs = plt.subplots(3, 1, sharex=True)
# axs = np.reshape(axs, (-1))
# for d in range(len(axs)):
# 	# Plot baseline dashed lines
# 	if d != 1:
# 		axs[d].plot([0, 3], [0, 0], 'k--', label='_nolegend_', linewidth=4)
# 	if d == 0:
# 		# Plot the mean pupil size
# 		for s in range(len(args.used_subj)):
# 			axs[d].plot(eyetrack_times, np.mean(eye_data[s], 0), linewidth=4)
# 		# Other plot parameters
# 		axs[d].set_ylabel('Centered\npupil size (px)', fontsize=25)
# 		title = 'Mean pupil size'
# 		axs[d].set_title(title, fontsize=30)
# 		legend = ['Subject 1', 'Subject 2']
# 		axs[d].legend(legend, fontsize=25, loc=0, ncol=1, frameon=True)
# 	elif d == 1:
# 		# Plot the mean luminance
# 		axs[d].plot(eyetrack_times, np.mean(luminance, 0), linewidth=4)
# 		# Other plot parameters
# 		axs[d].set_ylabel('cd/m²', fontsize=25)
# 		title = 'Mean luminance'
# 		axs[d].set_title(title, fontsize=30)
# 		legend = ['Luminance']
# 		axs[d].legend(legend, fontsize=25, loc=0, ncol=1, frameon=True)
# 	elif d == 2:
# 		# Plot the correlation between pupil size and luminance
# 		for s in range(len(args.used_subj)):
# 			axs[d].plot(eyetrack_times, pupil_lum_corr[s], linewidth=4)
# 		# Other plot parameters
# 		axs[d].set_ylabel('Pearson\'s $r$', fontsize=25)
# 		title = 'Correlation between pupil size and luminance'
# 		axs[d].set_title(title, fontsize=25)
# 		axs[d].set_xlabel('Time (s)', fontsize=25)
# 		axs[d].set_xlim(left=min(eyetrack_times), right=max(eyetrack_times))
# 		legend = ['Subject 1', 'Subject 2']
# 		axs[d].legend(legend, fontsize=25, loc=0, ncol=1, frameon=True)
# 		xticks = np.arange(0, 3.01, .1)
# 		xlabels = np.round(np.arange(0, 3.01, .1), 1)
# 		plt.xticks(ticks=xticks, labels=xlabels)
