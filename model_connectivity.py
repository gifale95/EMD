"""Link EEG and fMRI responses through model connectivity.

fMRI and EEG responses are linked via similarity of their encoding models
weights. For this, it is important that both fMRI and EEG encoding models
consist of the same feature space (i.e., same vision/language model, same
number of PCs).

Each fMRI vertex is indepentendly linked to each EEG time point. Therefore,
the results matrix will be of dimensions:
(370 EEG time points x 327,684 fMRI vertices)

For each vertex-time combination, the fMRI vertex weights are sequentially
compared with all the EEG chanels weights of the time point of interest.
Since there are 127 EEG channels, this will result in 127 similarity scores,
one for each channel. You will then keep either the best similarity score
of the best channel, or the mean similarity score across all channels, and
store this value in the results matrix.

You can perform this analysis independently for each fMRI hemisphere.

Parameters
----------
subjects: str
	If 'average', perform model connectivity using encoding models linear
	regression weights averaged across all EEG subjects, and across all fMRI
	subjects. If 'single', perform model connectivity using single subjects.
eeg_sub : int
	EEG subect used if subjects == 'single'.
fmri_sub : int
	fMRI subect used if subjects == 'single'.
fmri_hemi : str
	Whether to use the left ('lh') or right ('rh') fMRI hemishpere.
model: str
	Used vision/language model.
n_pcs: int
	Number of principal components used for the encoding models linear
	regression. Since model connectivity compares EEG and fMRI based on their
	encoding models linear regression weights, the number of principal
	components correponds to the number of samples in the model connectivity
	analysis.
similarity: str
	Whether to quantify the similarity of the linear regression weights using
	a Pearson's correlation ('pearson') or MSE error ('mse').
channel_policy: str
	During model connectivity, the ecoding weights of each fMRI vertex are
	compared with the encoding weights of all EEG channels, thus resulting
	in 127 similarity scores. If channel_policy == 'best', the best
	similarity score is kept (i.e., highest correlation, or lower MSE). If
	channel_policy == 'mean', the average similarity score is kept.
project_dir : str
	Project directory.

"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from joblib import load
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=str, default='average')
parser.add_argument('--eeg_sub', type=int, default=1)
parser.add_argument('--fmri_sub', type=int, default=1)
parser.add_argument('--fmri_hemi', type=str, default='lh')
parser.add_argument('--model', default='best_1', type=str) # ['best_1', 'best_2']
parser.add_argument('--n_pcs', default=100, type=int) # [best, 100, 200]
parser.add_argument('--similarity', default='pearson', type=str) # ['pearson', 'mse']
parser.add_argument('--channel_policy', default='best', type=str) # ['best', 'mean']
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

print('>>> Model connectivity <<<')
print('\nInput parameters:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Test (now)
# =============================================================================
# subjects:
	# 'average'.

# model:
	# best_1, best_2

# n_pcs:
	# Andreea: best PCs of each model, across both EEG and fMRI; 100; 200.
	# Lorenzo: best PCs of each model, across both EEG and fMRI; 100; 200.

# similarity:
	# 'pearson'; 'mse'.

# channel_policy:
	# 'best'; 'mean'.

# This will result in 24 different hperparameter combinations:
# (1 subjects x 2 model x 3 n_pcs x 2 similarity x 2 channel_policy).
# Therefore, you will eventually have 24 final plots: 24 pycortex brain
# surface movies, with the similarity scores indicating how similar are each
# vertex' weights with the EEG weights across all time points (i.e., one plot
# movie frame for each EEG time point).
# This movie should contain a colorbar, and also an iterable displaying the
# EEG time point (in second or milliseconds) corresponding to each movie frame.


# =============================================================================
# Test (later)
# =============================================================================
# n_pcs
	# Andreea: 400 (only with s3d)
	# Lorenzo: 400 (only with gte-Qwen1.5-7B-instruct)

# subjects:
	# 'single'.

# model:
	# best_2, best_3


# =============================================================================
# Load the fMRI encoding models weights
# =============================================================================
if args.subjects == 'average':

	fmri_weights = []
	fmri_subs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

	for s in fmri_subs:

		# You need to edit the 'weights_dir' so that it loads the right linear
		# regression weights: correct fMRI subject, hemisphere, model, and PC
		# number.
		# These are the linear regression weights you previously saved when
		# training the linear regression. See line 105 of this script for how
		# to save regression weights:
		# https://github.com/gifale95/NED/blob/main/ned_creation_code/01_train_encoding_models/train_dataset-things_eeg_2/model-vit_b_32/utils.py
		weights_dir = os.path.join(args.project_dir, '/.../',
			'LinearRegression_param.joblib')
		weights = load(weights_dir)
		# The weights are of shape (163842 fMRI vertices x N PCs)
		weights = weights.coef_
		# Store the weights
		fmri_weights.append(weights)
		del weights

	# Average the fMRI weights across subject
	fmri_weights = np.mean(fmri_weights, 0)


# =============================================================================
# Load the EEG encoding models weights
# =============================================================================
n_eeg_chans = 127
n_eeg_times = 370

if args.subjects == 'average':

	eeg_weights = []
	eeg_subs = [1, 2, 3, 4, 5, 6]

	for s in eeg_subs:

		# You need to edit the 'weights_dir' so that it loads the right linear
		# regression weights: correct EEG subject, model, and PC number.
		weights_dir = os.path.join(args.project_dir, '/.../',
			'LinearRegression_param.joblib')
		weights = load(weights_dir)
		# The weights are of shape (??? EEG features x N PCs)
		weights = weights.coef_
		# You first need to reshape the weights to:
		# (127 channels x 370 time points x N PCs)
		weights = np.reshape(weights, (n_eeg_chans, n_eeg_times, -1))
		# Store the weights
		eeg_weights.append(weights)
		del weights

	# Average the EEG weights across subject
	eeg_weights = np.mean(eeg_weights, 0)



		# EEG (127 channels x 50 PCs) --> fMRI (1 vertex x 50 PCs)


# =============================================================================
# Model connectivity
# =============================================================================
n_fmri_vertices = len(fmri_weights)
results_matrix = np.zeros((n_fmri_vertices, n_eeg_times))

for v in tqdm(range(n_fmri_vertices)):
	for t in range(n_eeg_times):

		# For each EEG time point, correlate the encoding weights of each fMRI
		# vertex with the weights of all EEG channels.
		chan_scores = np.zeros(n_eeg_chans)
		for c in range(n_eeg_chans):
			if args.similarity == 'pearson':
				chan_scores[c] = pearsonr(fmri_weights[v], eeg_weights[c,t])[0]
			elif args.similarity == 'mse':
				chan_scores[c] = mean_squared_error(
					fmri_weights[v], eeg_weights[c,t])

		# Keep the max correlation score
		if args.channel_policy == 'best':
			if args.similarity == 'pearson':
				# For Pearson's r, the higher the better
				results_matrix[v,t] = max(chan_scores)
			elif args.similarity == 'mse':
				# For MSE, the lower the better
				results_matrix[v,t] = min(chan_scores)
		elif args.channel_policy == 'mean':
			results_matrix[v,t] = np.mean(chan_scores)
		del chan_scores


# =============================================================================
# Save the result matrix
# =============================================================================

