"""RSA fusion between the EEG and fMRI test RDMs.

Parameters
----------
subjects : str
	Whether to use the RDMs of single subjects ['single'], or averaged across
	subjects ['all'].
sub : int
	Used subject, if "subjects==single".
tot_units : int
	Total of available data units.
mvnn : str
	Type of MVNN applied to preprocess the data ['time', 'epochs', 'baseline',
	'none'].
sfreq : int
	Downsampling frequency.
amount_channels : int
	Amount of used EEG channel groups. If '1' use 32 channels. If '2' use 64
	channels. If '3' use 96 channels. If '4' use 128 channels.
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
from tqdm import tqdm
from scipy.stats import pearsonr as corr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
# ('all', 'single')
parser.add_argument('--subjects', default='all', type=str)
# (1, 2)
parser.add_argument('--sub', default=1, type=int)
parser.add_argument('--tot_units', default=1, type=int)
parser.add_argument('--mvnn', default='time', type=str)
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--amount_channels', default=4, type=int)
parser.add_argument('--channels', default='OP', type=str)
parser.add_argument('--pseudo_trials', default=1, type=int)
parser.add_argument('--fmri_voxels', default='all', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
args = parser.parse_args()

print('>>> RSA fusion <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the EEG test RDMs
# =============================================================================
list_subs = [1, 2]

for s, sub in enumerate(list_subs):
	data_dir = os.path.join(args.project_dir, 'results', 'sub-'+
		format(sub,'02'), 'data_quality_check', 'mvnn-'+args.mvnn, 'sfreq-'+
		format(args.sfreq,'04'), 'pairwise_decoding', 'channels-'+args.channels, 'pseudo_trials-'+
		format(args.pseudo_trials,'02'), 'pairwise_decoding_chans-'+
		format(args.amount_channels,'02')+'.npy')
	data = np.load(data_dir, allow_pickle=True).item()
	eeg_rdm = data['pairwise_decoding']
	if s == 0:
		idx = np.tril_indices(eeg_rdm.shape[0], -1)
		eeg_rdm_flat = np.expand_dims(eeg_rdm[idx], 0)
		ch_names = data['ch_names']
		times = data['times']
		info = data['info']
	else:
		eeg_rdm_flat = np.append(eeg_rdm_flat, np.expand_dims(eeg_rdm[idx], 0), 0)
eeg_rdm = eeg_rdm_flat
del eeg_rdm_flat

# Subjects selection/averaging
if args.subjects == 'all':
	eeg_rdm = np.mean(eeg_rdm, 0)
else:
	eeg_rdm = eeg_rdm[args.sub-1]


# =============================================================================
# fMRI data parameters
# =============================================================================
if args.fmri_voxels == 'all':
	rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA',
		'OFA', 'STS', 'LOC', 'PPA', 'RSC', 'TOS', 'V3ab', 'IPS0', 'IPS1-2-3']
elif args.fmri_voxels == 'reliable':
	rois = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FFA',
		'OFA', 'STS', 'LOC', 'PPA', 'RSC', 'TOS', 'V3ab', 'IPS0', 'IPS1-2-3',
		'7AL', 'BA2', 'PFt', 'PFop']

trs = [1, 2, 3, 4, 5, 6, 7, 8, 9]


# =============================================================================
# Perform RSA
# =============================================================================
rsa_fusion = np.zeros((len(rois),len(trs),eeg_rdm.shape[1]))

for r, roi in tqdm(enumerate(rois)):
	for t, tr in enumerate(trs):
		# Average the fMRI RDMs across subjects
		subs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
		if roi == 'TOS':
			subs = [1, 2, 3, 4, 5, 8, 9, 10]
		elif roi == 'RSC':
			subs = [1, 2, 3, 4, 5, 6, 8, 9, 10]
		fmri_rdm = []
		for s in subs:
			data_dir = os.path.join(args.project_dir, 'results',
				'fmri_rdms_'+args.fmri_voxels+'_voxels', 'test_rdm_sub'+
				format(s, '02')+'_'+roi+'_TR'+format(tr, '02')+'.npy')
			fmri_rdm.append(np.load(data_dir))
		fmri_rdm = np.mean(np.asarray(fmri_rdm), 0)
		fmri_rdm = fmri_rdm[idx]
		# Correlate the EEG and fMRI RDMs
		for et in range(eeg_rdm.shape[1]):
			rsa_fusion[r,t,et] = corr(fmri_rdm, eeg_rdm[:,et])[0]


# =============================================================================
# Save the results
# =============================================================================
results_dict = {
	'rsa_fusion': rsa_fusion,
	'times': times,
	'ch_names': ch_names,
	'info': info,
	'rois': rois,
	'trs': trs
}

save_dir = os.path.join(args.project_dir, 'results', 'sub-'+\
	format(args.sub,'02'), 'data_quality_check', 'mvnn-'+args.mvnn, 'sfreq-'+
	format(args.sfreq,'04'), 'rsa_fusion', 'channels-'+args.channels, 'pseudo_trials-'+
	format(args.pseudo_trials,'02'))
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)
if args.subjects == 'all':
	file_name = 'rsa_fusion_chans-'+format(args.amount_channels,'02')+\
		'_voxels-'+args.fmri_voxels+'_eeg_subj-all'
else:
	file_name = 'rsa_fusion_chans-'+format(args.amount_channels,'02')+\
		'_voxels-'+args.fmri_voxels+'_eeg_subj-'+format(args.sub,'02')
np.save(os.path.join(save_dir, file_name), results_dict)