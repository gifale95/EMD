"""Extract visual features.

The feature maps come from a CLIP vision transformer, and are downsampled to 250
principal components using PCA.

https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html

Parameters
----------
things_eeg_2_dir : str
    Directory of the THINGS EEG2 dataset.
    https://osf.io/3jk45/
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG

"""

import argparse
import torch
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
import torchvision
from torchvision import transforms as trn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names
from sklearn.preprocessing import StandardScaler
from scipy.linalg import eigh


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--things_eeg_2_dir', default='/scratch/ccn_datasets/things_eeg_2', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/encoding_things_eeg2_zitong', type=str)
args, unknown = parser.parse_known_args()

print('>>> Train encoding models <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Define the image preprocessing
# =============================================================================
transform = trn.Compose([
    trn.Lambda(lambda img: trn.CenterCrop(min(img.size))(img)),
    trn.Resize((224,224)),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# =============================================================================
# Vision model
# =============================================================================
# Load the model
model = torchvision.models.vit_b_32(weights='DEFAULT')
model.to(device)
model.eval()

# Select the used layers for feature extraction
#nodes, _ = get_graph_node_names(model)
model_layers = ['encoder.layers.encoder_layer_0.add_1',
                'encoder.layers.encoder_layer_1.add_1',
                'encoder.layers.encoder_layer_2.add_1',
                'encoder.layers.encoder_layer_3.add_1',
                'encoder.layers.encoder_layer_4.add_1',
                'encoder.layers.encoder_layer_5.add_1',
                'encoder.layers.encoder_layer_6.add_1',
                'encoder.layers.encoder_layer_7.add_1',
                'encoder.layers.encoder_layer_8.add_1',
                'encoder.layers.encoder_layer_9.add_1',
                'encoder.layers.encoder_layer_10.add_1',
                'encoder.layers.encoder_layer_11.add_1']
feature_extractor = create_feature_extractor(model, return_nodes=model_layers)


# =============================================================================
# Extract the THINGS EEG2 training image features
# =============================================================================
# Image directories
img_dir = os.path.join(args.things_eeg_2_dir, 'image_set', 'training_images')
image_list = []
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".JPEG"):
            image_list.append(os.path.join(root,file))
image_list.sort()

features_train = []
with torch.no_grad():
    for i, img_dir in enumerate(tqdm(image_list, leave=False)):
        # Load the images
        img = Image.open(img_dir).convert('RGB')
        img = transform(img).unsqueeze(0)
        img = img.to(device)
        # Extract the features
        ft = feature_extractor(img)
        # Flatten the features
        ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
        features_train.append(np.squeeze(ft.detach().cpu().numpy()))
        del ft
features_train = np.asarray(features_train)


# =============================================================================
# Extract the THINGS EEG2 testing image features
# =============================================================================
# Image directories
img_dir = os.path.join(args.things_eeg_2_dir, 'image_set', 'test_images')
image_list = []
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".JPEG"):
            image_list.append(os.path.join(root,file))
image_list.sort()

features_test = []
for i, img_dir in enumerate(tqdm(image_list, leave=False)):
    # Load the images
    img = Image.open(img_dir).convert('RGB')
    img = transform(img).unsqueeze(0)
    img = img.to(device)
    # Extract the features
    ft = feature_extractor(img)
    # Flatten the features
    ft = torch.hstack([torch.flatten(l, start_dim=1) for l in ft.values()])
    features_test.append(np.squeeze(ft.detach().cpu().numpy()))
    del ft
features_test = np.asarray(features_test)


# =============================================================================
# PCA functions
# =============================================================================
def fit_pca_float32(X_train):
    """Fit PCA in sample space on float32 data.
    Assumes X_train is already z-scored so no centering needed."""

    # Covariance in sample space: (n_samples, n_samples)
    cov = (X_train @ X_train.T) / (X_train.shape[0] - 1)     # (1102, 1102), float32

    # Eigen-decomposition — eigh returns ascending order
    eigenvalues, eigenvectors = eigh(cov)

    # Reverse to descending order
    eigenvalues  = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    # Clip tiny negative eigenvalues caused by float32 numerical noise
    eigenvalues = np.maximum(eigenvalues, 0)

    # Principal axes in feature space: (n_features, n_samples)
    principal_axes = X_train.T @ eigenvectors                  # (n_features, n_samples)
    principal_axes /= np.linalg.norm(principal_axes, axis=0)   # normalize → V

    # Explained variance ratio (after clipping)
    explained_variance_ratio = eigenvalues / eigenvalues.sum()

    return principal_axes, explained_variance_ratio


def transform_pca_float32(X, principal_axes, n_components):
    """Project data onto the top n_components principal axes."""
    return X @ principal_axes[:, :n_components]               # (n_samples, n_components)


# =============================================================================
# Downsample the vision features using PCA
# =============================================================================
# Z-score the features
scaler = StandardScaler()
scaler.fit(features_train)
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

# Fit the PCA on the train data
principal_axes, explained_variance_ratio = fit_pca_float32(features_train)

# Find the number of components explaining 95% variance
cumulative_explained_variance = np.cumsum(explained_variance_ratio)
n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
print(f"Components explaining 95% variance: {n_components_95}")

# Transform train and test
features_train = transform_pca_float32(features_train, principal_axes,
    n_components_95)
features_test  = transform_pca_float32(features_test,  principal_axes,
    n_components_95)


# =============================================================================
# Save the visual features
# =============================================================================
acc = {
    'fmaps_train': features_train,
    'fmaps_test': features_test
    }

save_dir = os.path.join(args.project_dir, 'visual_features')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)

file_name = 'visual_features.npy'

np.save(os.path.join(save_dir, file_name), acc)