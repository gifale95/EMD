"""Extract and save language features from the EEG Moments video captions using
MPNet, a transformer-based language model. The language features are then
reduced to N principal components using PCA, where N corresponds to the number
of principal components that explain 95% of the variance.

https://huggingface.co/sentence-transformers/all-mpnet-base-v2

Parameters
----------
model_name : str
    Name of the language model used to extract the stimulus features.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).

"""

import os
import argparse
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='all-mpnet-base-v2', type=str)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Extract language features <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the language model
# =============================================================================
model = SentenceTransformer(f'sentence-transformers/{args.model_name}')


# =============================================================================
# Load the stimulus annotations
# =============================================================================
annotations_dir = os.path.join(args.emd_dir, 'derivatives',
    'stimuli_metadata', 'annotations.json')

annotations = json.load(open(annotations_dir))


# =============================================================================
# Extract the language features
# =============================================================================
# Extract the language features using both text (5 captions per video) and
# spoken (1 caption per video) descriptions, for a total of 6 text embeddings
# per video.
embeddings = []

# Loop across the 1102 videos
for i, (key, val) in enumerate(tqdm(annotations.items())):
    text_descriptions = val['text_descriptions']
    spoken_transcription = val['spoken_transcription']
    captions = text_descriptions + [spoken_transcription]
    embeddings.append(model.encode(captions))

# Shape: (102 videos, 6 captions per video, embedding dimension)
embeddings = np.array(embeddings).astype(np.float32)

# Divide the 1102 videos into the 102 test videos and 1000 training videos
embeddings_train = embeddings[:1000]
embeddings_test = embeddings[1000:]

# Reshape to (samples, features)
embeddings_train_shape = embeddings_train.shape
embeddings_test_shape = embeddings_test.shape
embeddings_train = embeddings_train.reshape(-1, embeddings_train_shape[-1])
embeddings_test = embeddings_test.reshape(-1, embeddings_test_shape[-1])


# =============================================================================
# Downsample the language features using PCA
# =============================================================================
# Z-score the features
scaler = StandardScaler()
scaler.fit(embeddings_train)
embeddings_train = scaler.transform(embeddings_train)
embeddings_test = scaler.transform(embeddings_test)

# Downsample the features with PCA
pca = PCA(random_state=20200220)
pca.fit(embeddings_train)
embeddings_train = pca.transform(embeddings_train)
embeddings_test = pca.transform(embeddings_test)

# Only retain the N principal components that explain 95% of the variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
embeddings_train = embeddings_train[:,:n_components_95]
embeddings_test = embeddings_test[:,:n_components_95]

# Reshape to the original shape: (videos, captions per video, embedding dimension)
embeddings_train = np.reshape(embeddings_train,
    (embeddings_train_shape[0], embeddings_train_shape[1], n_components_95))
embeddings_test = np.reshape(embeddings_test,
    (embeddings_test_shape[0], embeddings_test_shape[1], n_components_95))


# =============================================================================
# Save the language features
# =============================================================================
save_dir = os.path.join(args.emd_dir, 'results', 'stimulus_features',
    'language_features', args.model_name)
os.makedirs(save_dir, exist_ok=True)

file_name_train = 'language_features_train.npy'
file_name_test = 'language_features_test.npy'

np.save(os.path.join(save_dir, file_name_train), embeddings_train)
np.save(os.path.join(save_dir, file_name_test), embeddings_test)