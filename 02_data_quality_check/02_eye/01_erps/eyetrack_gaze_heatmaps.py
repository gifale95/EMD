"""Plot the gaze heatmaps of the eye-tracking data.

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
# Format the eye-tracking data
# =============================================================================
eye_data_all = {}
for s in args.used_subj:
	for ses in range(len(eyetrack_data[s])):
		if ses == 0:
			eye_data_all[s] = eyetrack_data[s][ses]
		else:
			eye_data_all[s] = np.append(eye_data_all[s], eyetrack_data[s][ses],
				0)
del eyetrack_data

# Retain only the video presentation time points
idx_start = np.where(eyetrack_times == 0)[0][0]
idx_end = np.where(eyetrack_times == 3)[0][0]
for s in args.used_subj:
	eye_data_all[s] = eye_data_all[s][:,idx_start:idx_end,:2]


# =============================================================================
# Extract the gaze positions into a 2D grid
# =============================================================================
max_vis_deg = 5
points_per_degree = 20
deg_per_point = 1 / points_per_degree # resolution of: 1 / 20 = 0.05° (~2 pixels)
size_heatmaps = int(max_vis_deg * points_per_degree * 2)

gaze_heatmaps = np.zeros((len(args.used_subj), size_heatmaps, size_heatmaps))
for s, sub in enumerate(args.used_subj):
	for v in range(eye_data_all[sub].shape[0]):
		for t in range(eye_data_all[sub].shape[1]):
			if np.isnan(eye_data_all[sub][v,t,0]) == False:
				x_coord = int(np.round((size_heatmaps / 2) + \
					(eye_data_all[sub][v,t,0] / deg_per_point)))
				y_coord = int(np.round((size_heatmaps / 2) + \
					(eye_data_all[sub][v,t,1] / deg_per_point)))
				gaze_heatmaps[s,x_coord,y_coord] += 1


# =============================================================================
# Plot parameters
# =============================================================================
# Setting the plot parameters
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 30
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
matplotlib.rcParams['axes.linewidth'] = 4
matplotlib.rcParams['xtick.major.width'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 3
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = True
matplotlib.rcParams['axes.spines.top'] = True
matplotlib.rcParams['axes.grid'] = True


# =============================================================================
# Plot the eye gaze heatmaps
# =============================================================================
fig, axs = plt.subplots(1, len(args.used_subj), sharex=True, sharey=True)
axs = np.reshape(axs, (-1))
for s, sub in enumerate(args.used_subj):
	# Plot the heatmap
	axs[s].imshow(gaze_heatmaps[s], cmap='inferno', aspect='equal')
	# Plot the video presentation box
	axs[s].plot([50, 150], [50, 50], 'w', [50, 150], [150, 150], 'w',
		[50, 50], [50, 150], 'w', [150, 150], [50, 150], 'w', linewidth=8)
	# Plot parameters
	if s == 0:
		axs[s].set_ylabel('Y°', fontsize=30)
	axs[s].set_xlabel('X°', fontsize=30)
	xticks = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 199]
	xlabels = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
	plt.xticks(ticks=xticks, labels=xlabels)
	yticks = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 199]
	ylabels = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]
	plt.yticks(ticks=yticks, labels=ylabels)
	title = 'Sub-' + format(sub)
	axs[s].set_title(title, fontsize=30)

#plt.savefig('eye_gaze_heatmaps_dataset-02', dpi=100)