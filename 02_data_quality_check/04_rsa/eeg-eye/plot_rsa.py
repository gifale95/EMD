"""Plot the RSA fusion between EEG and fMRI data.

Parameters
----------
used_subs : list
	List of used subjects.
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
channels : list
	List of used EEG channels
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
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--used_subs', default=[1, 2], type=list)
parser.add_argument('--baseline_correction', default=1, type=int)
parser.add_argument('--mvnn', default='time', type=str)
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--lowpass', default=40, type=float)
parser.add_argument('--highpass', default=.5, type=float)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--trials', default='all', type=str)
parser.add_argument('--amount_channels', default=4, type=int)
parser.add_argument('--channels', default='all', type=str)
parser.add_argument('--data_split', default='test', type=str) # ['train' 'test']
parser.add_argument('--pseudo_trials', default=1, type=int) # [0 1]
parser.add_argument('--mean_centering', default='baseline', type=str)
parser.add_argument('--features', default='all', type=str)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
args = parser.parse_args()


# =============================================================================
# Load RSA results
# =============================================================================
for s, sub in enumerate(args.used_subs):
	data_dir = os.path.join(args.project_dir, 'results',
		'combined_data_quality_check', 'rsa_eeg-eye', 'sub-'+format(sub,'02'),
		'data_split-'+args.data_split, 'channels-'+args.channels, 'rsa_eeg-eye.npy')
	# Load the data
	data_dict = np.load(data_dir, allow_pickle=True).item()
	if s == 0:
		rsa = np.expand_dims(np.transpose(data_dict['rsa']), 0)
		times = data_dict['times']
	else:
		rsa = np.append(rsa, np.expand_dims(np.transpose(data_dict['rsa']), 0), 0)
	del data_dict


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


# =============================================================================
# Plot the RSA results of single subjects
# =============================================================================
fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))
for s in range(2):
	axs[s].plot([0, 250], [50, 50], '--k', [0, 250], [200, 200], '--k',
		[50, 50], [0, 250], '--k', [200, 200], [0, 250], '--k', linewidth=4)
	img = axs[s].imshow(np.flip(rsa[s], 0), aspect='auto')
	title = 'Sub-' + str(s+1)
	axs[s].set_title(title, fontsize=20)
	# Plot parameters
	if s in [0, 1]:
		axs[s].set_xlabel('EEG time (s)', fontsize=20)
		xticks = [0, 50, 100, 150, 200, 250]
		xlabels = [-1, 0, 1, 2, 3, 4]
		plt.xticks(ticks=xticks, labels=xlabels)
	if s in [0]:
		axs[s].set_ylabel('Eye-tracking time (s)', fontsize=20)
		yticks = [0, 50, 100, 150, 200, 250]
		ylabels = [4, 3, 2, 1, 0, -1]
		plt.yticks(ticks=yticks, labels=ylabels)
plt.colorbar(img, label='Pearson\'s $r$', fraction=0.2, ax=axs[s])

#plt.savefig('rsa_eeg-eye_split-test_sub-single', dpi=100)
#plt.savefig('rsa_eeg-eye_split-train_sub-single', dpi=100)
#plt.savefig('rsa_eeg-eye_split-test_sub-single_legend', dpi=100)
#plt.savefig('rsa_eeg-eye_split-train_sub-single_legend', dpi=100)


# =============================================================================
# Plot the RSA results of averaged subjects
# =============================================================================
plt.figure()
plt.plot([0, 250], [50, 50], '--k', [0, 250], [200, 200], '--k',
	[50, 50], [0, 250], '--k', [200, 200], [0, 250], '--k', linewidth=4)
img = plt.imshow(np.flip(np.mean(rsa, 0), 0), aspect='auto')
# Plot parameters
plt.xlabel('EEG time (s)', fontsize=20)
xticks = [0, 50, 100, 150, 200, 250]
xlabels = [-1, 0, 1, 2, 3, 4]
plt.xticks(ticks=xticks, labels=xlabels)
plt.ylabel('Eye-tracking time (s)', fontsize=20)
yticks = [0, 50, 100, 150, 200, 250]
ylabels = [4, 3, 2, 1, 0, -1]
plt.yticks(ticks=yticks, labels=ylabels)
title = 'Sub-average'
plt.title(title, fontsize=20)
plt.colorbar(img, label='Pearson\'s $r$', fraction=.1)


#plt.savefig('rsa_eeg-eye_split-test_sub-average', dpi=100)
#plt.savefig('rsa_eeg-eye_split-train_sub-average', dpi=100)
