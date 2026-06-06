"""Transform the stimulus features for using PCA.

Parameters
----------
modality : str
	Whether to transform stimulus 'visual' features from video DNNs, or
	stimulus 'semantic' features from LLMs.
model_name : str
	Name of the model used for feature extraction.
n_components : int
	Number of model PCA components retained.
project_dir : str
	Directory of the project folder.

"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--modality', default='visual', type=str)
parser.add_argument('--model_name', default='s3d', type=str)
parser.add_argument('--n_components', default=1000, type=int)
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/science/projects/eeg_moments', type=str)
#parser.add_argument('--project_dir', default='/home/ale/scratch/projects/eeg_moments', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments', type=str)
args = parser.parse_args()

print('>>> Stimulus feature PCA <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Reshape the model features to (Samples x Features)
# =============================================================================
# Get the feature files list
features_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
	'full_model_features', 'modality-'+args.modality, 'model-'+args.model_name)
features_list = os.listdir(features_dir)
features_list.sort()

if args.modality == 'visual':

	features = []
	for v, file in enumerate(tqdm(features_list)):
		f = np.empty(0, dtype=np.float32)
		feature_dict = np.load(os.path.join(features_dir, file),
			allow_pickle=True).item()
		for key, val in feature_dict.items():
			f = np.append(f, np.reshape(val, -1))
		features.append(f)
		del f
	features = np.asarray(features)

elif args.modality == 'semantic':

	pass


# =============================================================================
# Divide the features into training and testing
# =============================================================================
features_train = features[:1000]
features_test = features[1000:]
del features


# =============================================================================
# Apply PCA
# =============================================================================
# Z-score the image features
scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

# Downsample the features with PCA
if features_train.shape[1] < args.n_components:
	n_components = features_train.shape[1]
else:
	n_components = args.n_components
pca = TruncatedSVD(n_components=n_components, random_state=20200220)
pca.fit(features_train)
features_train = pca.transform(features_train)
features_test = pca.transform(features_test)


# =============================================================================
# Save the PCA-transformed model features
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
	'pca_model_features', 'modality-'+args.modality)
if os.path.isdir(save_dir) == False:
	os.makedirs(save_dir)

file_name_train = 'pca_stimulus_features_train_model-' + args.model_name + \
	'.npy'
file_name_test = 'pca_stimulus_features_test_model-' + args.model_name + \
	'.npy'

np.save(os.path.join(save_dir, file_name_train), features_train)
np.save(os.path.join(save_dir, file_name_test), features_test)
