"""Train encoding models that map model features onto t-fMRI responses for the
1000 training videos.

Parameters
----------
fmri_subject : int
	Number of used BOLM Moments fMRI subject.
fmri_hemi : str
	Whether to use the 'left' or 'right' fMRI hemisphere.
fmri_split : int
	fMRI split used.
tot_fmri_splits : int
	Total amount of fMRI splits.
eeg_channel_policy : str
	If 'average', the EEG responses are averaged across subjects, and then
	transformed with PCA. If 'append', the EEG responses are appended across
	subjects across the channel dimension, and transformed with PCA.
max_eeg_pcs : int
	Max number of EEG PCs used.
sfreq : int
	EEG downsampling frequency.
modality : str
	Whether to transform stimulus 'visual' features from video DNNs, or
	stimulus 'semantic' features from LLMs.
model_name : str
	Name of the model used for feature extraction.
n_components : int
	Number of model PCs retained.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LinearRegression


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--fmri_subject', type=int, default=1)
parser.add_argument('--fmri_hemi', type=str, default='left')
parser.add_argument('--fmri_split', type=int, default=1)
parser.add_argument('--tot_fmri_splits', type=int, default=21)
parser.add_argument('--eeg_channel_policy', type=str, default='append')
parser.add_argument('--max_eeg_pcs', type=int, default=500)
parser.add_argument('--sfreq', type=int, default=100)
parser.add_argument('--modality', default='visual', type=str)
parser.add_argument('--model_name', default='s3d', type=str)
parser.add_argument('--n_components', default=100, type=int)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/science/projects/eeg_moments', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments', type=str)
args = parser.parse_args()

print('>>> EEG-fMRI model fusion, train encoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Load the PCA-transformed training model features
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
	'pca_model_features', 'modality-'+args.modality,
	'pca_stimulus_features_train_model-'+args.model_name+'.npy')

model_features = np.load(data_dir)[:,:args.n_components]


# =============================================================================
# Load the training EEG responses
# =============================================================================
# Train the DNN-to-tfMRI encoding models on t-fMRI responses generated using EEG
# responses from the even repeats
data_dir = os.path.join(args.project_dir, 'results',
	'eeg_fmri_fusion_even_odd_eeg_split', 'eeg_pca',
	'eeg_pca_train_even_channel_policy-'+args.eeg_channel_policy+'_sfreq-'+
	format(args.sfreq, '04')+'.npy')

eeg_dict = np.load(data_dir, allow_pickle=True).item()
eeg = eeg_dict['eeg_pca'][:,:args.max_eeg_pcs].astype(np.float32)
ch_names = eeg_dict['ch_names']
times = eeg_dict['times']
del eeg_dict


# =============================================================================
# Load the trained EEG-fMRI fusion models
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results',
	'eeg_fmri_fusion_even_odd_eeg_split', 'regression_weights', 'sfreq-'+
	format(args.sfreq, '04'), 'eeg_channel_policy-'+args.eeg_channel_policy,
	'fmri_hemi-'+args.fmri_hemi, 'regression_weights_fmri_split-'+
	format(args.fmri_split, '03')+'.npy')
regression_weights = np.load(data_dir, allow_pickle=True).item()


# =============================================================================
# Train the regressions that map model features onto t-fMRI responses
# =============================================================================
# Empty results dictionaries
results = {}
results['ch_names'] = ch_names
results['times'] = times
results['reg_param'] = []

# Loop over EEG time points
for t in tqdm(range(len(times))):

	# Create the linear regression object
	reg = LinearRegression()
	reg.coef_ = regression_weights['reg_param'][t]['coef_']
	reg.intercept_ = regression_weights['reg_param'][t]['intercept_']
	reg.n_features_in_ = regression_weights['reg_param'][t]['n_features_in_']

	# Generate the fMRI responses for the train videos
	pred_fmri = reg.predict(eeg[:,:,t]).astype(np.float32)

	# Train regressions that map model features onto t-fMRI responses
	reg = LinearRegression().fit(model_features, pred_fmri)

	# Store the regression parameters
	param = {
		'coef_': reg.coef_,
		'intercept_': reg.intercept_,
		'n_features_in_': reg.n_features_in_
		}
	results['reg_param'].append(param)
	del reg, param


# =============================================================================
# Save the regression weights
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results',
	'eeg_fmri_model_fusion_even_odd_eeg_split', 'regression_weights', 'sfreq-'+
	format(args.sfreq, '04'), 'eeg_channel_policy-'+ args.eeg_channel_policy,
	'fmri_hemi-'+args.fmri_hemi, 'modality-'+args.modality, 'model-'+
	args.model_name)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name = 'regression_weights_fmri_split-' + format(args.fmri_split, '03') + \
	'.npy'

np.save(os.path.join(save_dir, file_name), results)
