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
trials : str
	Whether to decode the data of the 'first' sequence, 'second' sequence, or
	'all' trials.
features : str
	Whether to decode 'gaze', 'pupil' or 'all' features.
data_split : str
	Whether to decode the 'test' or 'train' split.
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
parser.add_argument('--mean_centering', default='none', type=str) # ['baseline', 'epoch', 'none']
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--zscore', default=1, type=int)
parser.add_argument('--trials', default='all', type=str)
parser.add_argument('--features', default='all', type=str) # ['gaze', 'pupil', 'all']
parser.add_argument('--data_split', default='train', type=str) # ['train', 'test']
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
args = parser.parse_args()


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
if args.data_split == 'test':
	pseudo_trials = 1
elif args.data_split == 'train':
	pseudo_trials = 0

pairwise_decoding = []
time_splits = 10

for s in args.used_subs:
	for t in range(time_splits):
		data_dir = os.path.join(args.project_dir, 'results',
			'eye_data_quality_check', 'pairwise_decoding', 'sub-'+
			format(s,'02'), 'sfreq-'+format(args.sfreq,'04'), 'data_split-'+
			args.data_split, 'mean_centering-'+args.mean_centering,
			'sequence_trials-'+args.trials, 'features-'+args.features, 'zscore-'+
			format(args.zscore, '02'), 'pseudo_trials-'+
			format(pseudo_trials,'02'), 'pairwise_decoding_time_split-'+
			format(t+1, '02')+'.npy')
		data = np.load(data_dir, allow_pickle=True).item()
		decoding = data['pairwise_decoding']
		if t == 0:
			times = data['times']
			pair_dec_sub = decoding
		else:
			pair_dec_sub = np.append(pair_dec_sub, decoding, 2)
		del data
	pairwise_decoding.append(pair_dec_sub)
pairwise_decoding = np.asarray(pairwise_decoding)


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
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.grid'] = True


# =============================================================================
# Plot the decoding RDMs (single subjects)
# =============================================================================
time_idx = np.where(times == 0)[0][0]

fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
axs = np.reshape(axs, (-1))
for s, sub in enumerate(args.used_subs):
	# Plot the decoding results
	axs[s].imshow(pairwise_decoding[s,:,:,time_idx], aspect='auto')
	# Other plot parameters
	axs[s].set_xlabel('Video conditions', fontsize=30)
	if s in [0]:
		axs[s].set_ylabel('Video conditions', fontsize=30)
	tit = 'Sub-' + format(sub) + ', ' + str(times[time_idx]) + 's, ' + \
		'data_split-'+args.data_split + ', mean_centering-' + args.mean_centering
	axs[s].set_title(tit, fontsize=30)

#plt.savefig('eye_decoding_rdm_sub-sing_dtype-test_centering-baseline_features-all', dpi=100)
#plt.savefig('eye_decoding_rdm_sub-sing_dtype-train_centering-baseline_features-all', dpi=100)

#plt.savefig('eye_decoding_rdm_sub-sing_dtype-test_centering-epoch_features-all', dpi=100)
#plt.savefig('eye_decoding_rdm_sub-sing_dtype-train_centering-epoch_features-all', dpi=100)

#plt.savefig('eye_decoding_rdm_sub-sing_dtype-test_centering-none_features-all', dpi=100)
#plt.savefig('eye_decoding_rdm_sub-sing_dtype-train_centering-none_features-all', dpi=100)


# =============================================================================
# Plot the decoding RDMs (averaged subjects)
# =============================================================================
# Plot the decoding results
plt.figure()
plt.imshow(np.mean(pairwise_decoding[:,:,:,time_idx], 0), aspect='auto')
# Other plot parameters
plt.xlabel('Video conditions', fontsize=30)
plt.ylabel('Video conditions', fontsize=30)
tit = 'Sub-mean, ' + str(times[time_idx]) + 's, ' + 'data_split-' + \
	args.data_split + ', mean_centering-' + args.mean_centering
plt.title(tit, fontsize=30)

#plt.savefig('eye_decoding_rdm_sub-avg_dtype-test_centering-baseline_features-all', dpi=100)
#plt.savefig('eye_decoding_rdm_sub-avg_dtype-train_centering-baseline_features-all', dpi=100)

#plt.savefig('eye_decoding_rdm_sub-avg_dtype-test_centering-epoch_features-all', dpi=100)
#plt.savefig('eye_decoding_rdm_sub-avg_dtype-train_centering-epoch_features-all', dpi=100)

#plt.savefig('eye_decoding_rdm_sub-avg_dtype-test_centering-none_features-all', dpi=100)
#plt.savefig('eye_decoding_rdm_sub-avg_dtype-train_centering-none_features-all', dpi=100)