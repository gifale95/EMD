"""Plot the RSA fusion between EEG and fMRI data.

Parameters
----------
used_subs : list
		List of used subjects.
mvnn : str
	Type of MVNN applied to preprocess the data ['time', 'epochs', 'baseline',
	'none'].
channels : str
	Whether to retain occipital ['O'], posterior ['P'], temporal ['T'],
	central ['C'], frontal ['F'] occipital/parital ['OP'] or all ['all']
	channels.
pseudo_trials : int
	If 1, create pseudo-trials.
fmri_voxels : str
	Whether to use fMRI RDMs built using 'all' or 'reliable' voxels.
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
parser.add_argument('--mvnn', default='time', type=str)
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--channels', default='OP', type=str)
parser.add_argument('--pseudo_trials', default=1, type=int)
parser.add_argument('--fmri_voxels', default='reliable', type=str)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/'
	'projects/eeg_videos', type=str)
args = parser.parse_args()


# =============================================================================
# Load the pairwise decoding results of single subjects EEG RDMs
# =============================================================================
for s, sub in enumerate(args.used_subs):
	data_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(sub,'02'), 'data_quality_check', 'mvnn-'+args.mvnn,'sfreq-'+
		format(args.sfreq,'04'),
		'rsa_fusion', 'channels-'+args.channels, 'pseudo_trials-'+
		format(args.pseudo_trials,'02'), 'rsa_fusion_chans-'+format(4,'02')+
		'_voxels-'+args.fmri_voxels+'_eeg_subj-'+format(sub,'02')+'.npy')
	# Load the data
	data_dict = np.load(data_dir, allow_pickle=True).item()
	if s == 0:
		rsa_single = np.expand_dims(data_dict['rsa_fusion'], 0)
		times = data_dict['times']
		rois = data_dict['rois']
		trs = data_dict['trs']
	else:
		rsa_single = np.append(rsa_single, np.expand_dims(
			data_dict['rsa_fusion'], 0), 0)


# =============================================================================
# Load the pairwise decoding results of EEG RDMs averaged across subjects
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'sub-'+
	format(1,'02'), 'data_quality_check', 'mvnn-'+args.mvnn, 'sfreq-'+
	format(args.sfreq,'04'),
	'rsa_fusion', 'channels-'+args.channels, 'pseudo_trials-'+
	format(args.pseudo_trials,'02'), 'rsa_fusion_chans-'+format(4,'02')+
		'_voxels-'+args.fmri_voxels+'_eeg_subj-all'+'.npy')
# Load the data
data_dict = np.load(data_dir, allow_pickle=True).item()
rsa_average = data_dict['rsa_fusion']
times = data_dict['times']
rois = data_dict['rois']
trs = data_dict['trs']


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
color_noise_ceiling = (150/255, 150/255, 150/255)
colors = [(31/255, 119/255, 180/255), (255/255, 127/255, 14/255),
	(44/255, 160/255, 44/255), (214/255, 39/255, 40/255)]


# =============================================================================
# Plot the decoding results of single subject EEG RDMs
# =============================================================================
fig, axs = plt.subplots(6, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for r in range(len(axs)):
	if r < len(rois):
		img = axs[r].imshow(rsa_single[1,r], aspect='auto')
		axs[r].set_title(rois[r], fontsize=20)
	# Plot parameters
	if r in [20, 21, 22, 23]:
		axs[r].set_xlabel('EEG time (s)', fontsize=20)
		xticks = [25, 75, 125, 175, 225, 275]
		xlabels = [0, 1, 2, 3, 4, 5]
		plt.xticks(ticks=xticks, labels=xlabels)
	if r in [0, 4, 8, 12, 16, 20]:
		axs[r].set_ylabel('fMRI TRs', fontsize=20)
		yticks = np.arange(0, len(trs))
		ylabels = trs
		plt.yticks(ticks=yticks, labels=ylabels)
plt.colorbar(img, label='Pearson\'s $r$', fraction=0.2, ax=axs[r])

#plt.savefig('decoding_mvnn-time_chan-all_trials-normal', dpi=100)


# =============================================================================
# Plot the decoding results of EEG RDMs averaged across subjects
# =============================================================================
fig, axs = plt.subplots(6, 4, 'all', 'all')
axs = np.reshape(axs, (-1))
for r in range(len(axs)):
	if r < len(rois):
		img = axs[r].imshow(rsa_average[r], aspect='auto')
		axs[r].set_title(rois[r], fontsize=20)
	# Plot parameters
	if r in [20, 21, 22, 23]:
		axs[r].set_xlabel('EEG time (s)', fontsize=20)
		xticks = [25, 75, 125, 175, 225, 275]
		xlabels = [0, 1, 2, 3, 4, 5]
		plt.xticks(ticks=xticks, labels=xlabels)
	if r in [0, 4, 8, 12, 16, 20]:
		axs[r].set_ylabel('fMRI TRs', fontsize=20)
		yticks = np.arange(0, len(trs))
		ylabels = trs
		plt.yticks(ticks=yticks, labels=ylabels)
plt.colorbar(img, label='Pearson\'s $r$', fraction=.1, ax=axs[r])


#plt.savefig('rsa_mvnn-time_chan-OP_trials-pseudo_subj-avg_corr-pears', dpi=100)
#plt.savefig('colorbar', dpi=100)

