"""Plot the encoding accuracy.

Parameters
----------
fmri_subjects : list
	List of used BOLM Moments fMRI subjects.
tot_fmri_splits : int
	Total amount of fMRI splits.
eeg_channel_policy : str
	If 'average', the EEG responses are averaged across subjects, and then
	transformed with PCA. If 'append', the EEG responses are appended across
	subjects across the channel dimension, and transformed with PCA.
sfreq : int
	EEG downsampling frequency.
modality : str
	Whether to transform stimulus 'visual' features from video DNNs, or
	stimulus 'semantic' features from LLMs.
model_name : str
	Name of the model used for feature extraction.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from copy import copy
import cortex
import cortex.polyutils
import matplotlib
import matplotlib.pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subjects', type=list, default=[1])
parser.add_argument('--tot_fmri_splits', type=int, default=21)
parser.add_argument('--eeg_channel_policy', type=str, default='append')
parser.add_argument('--sfreq', type=int, default=100)
parser.add_argument('--modality', default='visual', type=str)
parser.add_argument('--model_name', default='s3d', type=str)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/science/projects/eeg_moments', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments', type=str)
args = parser.parse_args()

print('>>> EEG-fMRI fusion, Plot encoding accuracy <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the encoding accuracies
# =============================================================================
lh_correlation = []
rh_correlation = []
lh_noise_ceiling = []
rh_noise_ceiling = []
lh_explained_variance = []
rh_explained_variance = []

for sub in args.fmri_subjects:
	for hemi in ['left', 'right']:
		for split in range(args.tot_fmri_splits):
			data_dir = os.path.join(args.project_dir, 'results',
				'eeg_fmri_model_fusion_even_odd_eeg_split', 'encoding_accuracy',
				'sfreq-'+format(args.sfreq, '04'), 'eeg_channel_policy-'+
				args.eeg_channel_policy, 'fmri_hemi-'+hemi, 'modality-'+
				args.modality, 'model-'+args.model_name,
				'encoding_accuracy_fmri_split-'+format(split+1, '03')+'.npy')
			results = np.load(data_dir, allow_pickle=True).item()
			if split == 0:
				corr = results['correlation']
				nc = results['noise_ceiling']
				exp_var = results['explained_variance']
				times = results['times']
			else:
				corr = np.append(corr, results['correlation'], 0)
				nc = np.append(nc, results['noise_ceiling'], 0)
				exp_var = np.append(exp_var, results['explained_variance'], 0)
		if hemi == 'left':
			lh_correlation.append(corr.astype(np.float32))
			lh_noise_ceiling.append(nc.astype(np.float32))
			lh_explained_variance.append(exp_var.astype(np.float32))
		elif hemi == 'right':
			rh_correlation.append(corr.astype(np.float32))
			rh_noise_ceiling.append(nc.astype(np.float32))
			rh_explained_variance.append(exp_var.astype(np.float32))
		del corr, exp_var

lh_correlation = np.asarray(lh_correlation)
rh_correlation = np.asarray(rh_correlation)
lh_noise_ceiling = np.asarray(lh_noise_ceiling)
rh_noise_ceiling = np.asarray(rh_noise_ceiling)
lh_explained_variance = np.asarray(lh_explained_variance)
rh_explained_variance = np.asarray(rh_explained_variance)


# =============================================================================
# Plot parameters for colorbar
# =============================================================================
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage_nsd'


# =============================================================================
# Save directory
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results',
	'eeg_fmri_model_fusion_even_odd_eeg_split', 'encoding_accuracy_plots',
	'sfreq-'+format(args.sfreq, '04'), 'eeg_channel_policy-'+
	args.eeg_channel_policy, 'modality-'+args.modality, 'model-'+
	args.model_name)

if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)


# =============================================================================
# Plot the results, averaged across subjects
# =============================================================================
# Plot the correlation results
for t, time in enumerate(tqdm(times)):
	# Select the data to plot
	corr = np.append(np.mean(lh_correlation[:,:,t], 0),
		np.mean(rh_correlation[:,:,t], 0))
	# Plot the data on a brain surface
	vertex_data = cortex.Vertex(corr, subject, cmap='hot', vmin=0, vmax=1,
		with_colorbar=True)
	fig = cortex.quickshow(vertex_data,
	#	height=500, # Increase resolution of map and ROI contours
		with_curvature=True,
		curvature_brightness=0.5,
		with_rois=True,
		with_labels=False,
		linewidth=5,
		linecolor=(1, 1, 1),
		with_colorbar=True
		)
	# Figure title
	title = str(np.round(time, 3))+' s'
	plt.title(title, fontsize=40)
	plt.show()
	# Save the surface plot
	file_name = os.path.join(save_dir,
		'model_fusion_correlation_eeg_channel_policy-'+args.eeg_channel_policy+
		'_sub-average_time_'+format(t, '04')+'.png')
	fig.savefig(file_name, dpi=100, bbox_inches='tight', transparent=False,
		format='png')
	plt.close()

# Plot the explained variance results
for t, time in enumerate(tqdm(times)):
	# Select the data to plot, and remove vertices with noise ceiling values
	# below a threshold, since they cannot be interpreted in terms of modeling
	expl_var = []
	for s in range(len(args.fmri_subjects)):
		lh_data = copy(lh_explained_variance[s,:,t])
		rh_data = copy(rh_explained_variance[s,:,t])
		lh_idx = lh_noise_ceiling[s] < 10
		rh_idx = rh_noise_ceiling[s] < 10
		lh_data[lh_idx] = np.nan
		rh_data[rh_idx] = np.nan
		expl_var.append(np.append(lh_data, rh_data))
	expl_var = np.mean(expl_var, 0)
	# Plot the data on a brain surface
	vertex_data = cortex.Vertex(expl_var, subject, cmap='hot', vmin=0, vmax=1,
		with_colorbar=True)
	fig = cortex.quickshow(vertex_data,
	#	height=500, # Increase resolution of map and ROI contours
		with_curvature=True,
		curvature_brightness=0.5,
		with_rois=True,
		with_labels=False,
		linewidth=5,
		linecolor=(1, 1, 1),
		with_colorbar=True
		)
	# Figure title
	title = str(np.round(time, 3))+' s'
	plt.title(title, fontsize=40)
	plt.show()
	# Save the surface plot
	file_name = os.path.join(save_dir,
		'model_fusion_explained_variance_eeg_channel_policy-'+
		args.eeg_channel_policy+'_sub-average_time_'+format(t, '04')+'.png')
	fig.savefig(file_name, dpi=100, bbox_inches='tight', transparent=False,
		format='png')
	plt.close()


# =============================================================================
# Plot the results, for single subjects # !!!
# =============================================================================



