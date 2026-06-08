"""Extract and save visual features from the EEG Moments videos using S3D, a
3D CNN trained on video classification. The vision features are then reduced to
N principal components using PCA, where N corresponds to the number of
principal components that explain 95% of the variance.

https://docs.pytorch.org/vision/main/models/generated/torchvision.models.video.s3d.html

Parameters
----------
model_name : str
    Name of the vision model used to extract the stimulus features.
batch_size : int
    Stimulus batch size for model feature extraction.
project_dir : str
    Directory of the project folder.

"""

import os
import argparse
import numpy as np
from enum import Enum
import torch
import torchvision
from torchvision.io.video import read_video
from torch.utils.data import Dataset, DataLoader
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.linalg import eigh


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='s3d', type=str)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Extract vision features <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# GPU and workers
# =============================================================================
# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# Load the S3D model and video transform
# =============================================================================
model = torchvision.models.video.s3d(weights='KINETICS400_V1')
transform = torchvision.models.video.S3D_Weights.KINETICS400_V1.transforms()


# =============================================================================
# Define the layers from which to extract features
# =============================================================================
# For each model, we specify the layers from which to save the features. This
# layer selection is needed because the networks are too large to save all the
# features of all layers. The saved layers are spread across early, middle and
# late layers of the networks.

class Nodes_for_FE(Enum):
    # Define the layers from which to extract features
    VideoResNet_mc3_18 = {
        'stem.2': 'stem',
        'layer1.1.relu': 'layer1',
        'layer2.1.relu': 'layer2',
        'layer3.1.relu': 'layer3',
        'layer4.1.relu': 'layer4',
        'avgpool': 'avgpool'
        }
    VideoResNet_r3d_18 = {
        'stem.2': 'stem',
        'layer1.1.relu': 'layer1',
        'layer2.1.relu': 'layer2',
        'layer3.1.relu': 'layer3',
        'layer4.1.relu': 'layer4',
        'avgpool': 'avgpool'
        }
    VideoResNet_r2plus1d_18 = {
        'stem.5': 'stem',
        'layer1.1.relu': 'layer1',
        'layer2.1.relu': 'layer2',
        'layer3.1.relu': 'layer3',
        'layer4.1.relu': 'layer4',
        'avgpool': 'avgpool'
    }
    s3d = {
        'features.1': 'layer1',
        'features.2': 'layer2',
        'features.3.1': 'layer3',
        'features.4': 'layer4',
        'features.5.cat': 'layer5',
        'features.6.cat': 'layer6',
        'features.7': 'layer7',
        'features.8.cat': 'layer8',
        'features.9.cat': 'layer9',
        'features.10.cat': 'layer10',
        'features.11.cat': 'layer11',
        'features.12.cat': 'layer12',
        'features.13': 'layer13',
        'features.14.cat': 'layer14',
        'features.15.cat': 'layer15',
        'avgpool': 'avgpool',
        'classifier.1': 'classifier'
        }
    MViT_v2_s = {
        'pos_encoding.cat' : 'stem',
        'blocks.2.mlp': 'early_block',
        'blocks.9.mlp': 'middle_block',
        'blocks.15.mlp': 'late_block'
    }
    MViT_v1_b = {
        'pos_encoding.cat' : 'stem',
        'blocks.2.mlp': 'early_block',
        'blocks.9.mlp': 'middle_block',
        'blocks.15.mlp': 'late_block'
    }
    x3d_xs = {
        'blocks.0.activation': 'stem',
        'blocks.1.res_blocks.2.activation': 'layer1',
        'blocks.2.res_blocks.4.activation': 'layer2',
        'blocks.3.res_blocks.10.activation': 'layer3',
        'blocks.4.res_blocks.6.activation': 'layer4',
        'blocks.5.proj': 'layer5'
    }
    ResNet_3D = {
        'blocks.0.pool': 'stem',
        'blocks.2.res_blocks.3.activation': 'layer2',
        'blocks.4.res_blocks.2.activation': 'layer4',
        'blocks.5.proj': 'projection'
    }


# =============================================================================
# Select the video frame number
# =============================================================================
# Each model has a specific number of frames that need to be selected at once
# from the input video.

class Num_frames(Enum):
    VideoResNet_mc3_18 = 8
    VideoResNet_r3d_18 = 8
    VideoResNet_r2plus1d_18 = 8
    s3d = 14
    MViT_v2_s = 16
    MViT_v1_b = 16
    x3d_xs = 4
    ResNet_3D = 8


# =============================================================================
# Video dataset class
# =============================================================================
class VideoDataset(Dataset):
    def __init__(self, video_dir, num_samples, device, transform=None):
        self.video_dir = video_dir
        self.video_files = sorted([os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(('.mp4'))])
        assert len(self.video_files) == 1102
        #self.video_files = videos_first_N
        self.num_samples = num_samples
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def sample_frames(self, video_frames, num_samples):
        num_frames = video_frames.shape[0]
        if num_samples > num_frames:
            raise ValueError("The number of samples requested is greater than the number of frames in the video.")
        indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
        sampled_frames = video_frames[indices]
        return sampled_frames

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        video_frames, _, _ = read_video(video_path, pts_unit='sec',
            output_format='TCHW')
        try:
            sampled_frames = self.sample_frames(video_frames, self.num_samples)
        except ValueError:
            last_frame = video_frames[-1].unsqueeze(0).repeat(
                self.num_samples - video_frames.shape[0], 1, 1, 1)
            sampled_frames = torch.cat([video_frames, last_frame], dim=0)
        if self.transform:
            sampled_frames = self.transform(sampled_frames)
            sampled_frames = sampled_frames.to(device)
        return idx, sampled_frames


# =============================================================================
# Model specific parameters
# =============================================================================
# Define the video frame sample number
if args.model_name == 'mc3_18':
    num_samples = Num_frames.VideoResNet_mc3_18.value
    fe_nodes = Nodes_for_FE.VideoResNet_mc3_18.value
elif args.model_name == 'r3d_18':
    num_samples = Num_frames.VideoResNet_r3d_18.value
    fe_nodes = Nodes_for_FE.VideoResNet_r3d_18.value
elif args.model_name == 'r2plus1d_18':
    num_samples = Num_frames.VideoResNet_r2plus1d_18.value
    fe_nodes = Nodes_for_FE.VideoResNet_r2plus1d_18.value
elif args.model_name == 's3d':
    num_samples = Num_frames.s3d.value
    fe_nodes = Nodes_for_FE.s3d.value
elif args.model_name == 'mvit_v2_s':
    num_samples = Num_frames.MViT_v2_s.value
    fe_nodes = Nodes_for_FE.MViT_v2_s.value
elif args.model_name == 'mvit_v1_b':
    num_samples = Num_frames.MViT_v1_b.value
    fe_nodes = Nodes_for_FE.MViT_v1_b.value
elif args.model_name == 'x3d_xs':
    num_samples = Num_frames.x3d_xs.value
    fe_nodes = Nodes_for_FE.x3d_xs.value
elif args.model_name == 'slow_r50':
    num_samples = Num_frames.ResNet_3D.value
    fe_nodes = Nodes_for_FE.ResNet_3D.value


# =============================================================================
# Create the dataset and dataloader
# =============================================================================
video_dir = os.path.join(args.project_dir, 'stimuli')

# Create the dataset
dataset = VideoDataset(video_dir=video_dir, num_samples=num_samples,
    device=device, transform=transform)

# Create a DataLoader without shuffling
dataloader = None
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
    pin_memory=False) # num_workers=num_workers


# =============================================================================
# Create the feature extractor
# =============================================================================
feature_extractor = create_feature_extractor(model, fe_nodes)

# Set the model in evaluation mode, on the current device
feature_extractor.eval()
feature_extractor.to(device)


# =============================================================================
# Extract the vision features
# =============================================================================
# Extract the vision features
features = []
with torch.no_grad():
    for indices, batch in tqdm(dataloader):
        ft_batch = feature_extractor(batch)
        for idx in range(len(indices)):
            ft = np.empty(0, dtype=np.float32)
            for key in fe_nodes.values():
                ft = np.append(ft, ft_batch[key][idx].flatten())
            features.append(ft.astype(np.float32))
            del ft
        del ft_batch
        print('batch {} - {} / 1101 done'.format(indices[0].item(), indices[-1].item()))
features = np.array(features).astype(np.float32)

# Divide the 1102 videos into the 102 test videos and 1000 training videos
features_train = features[:1000].astype(np.float32)
features_test = features[1000:].astype(np.float32)
del features


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

# # Downsample the features with PCA # !!! OLD DELETE
# pca = PCA(n_components=0.95, svd_solver="covariance_eigh",
#     random_state=20200220)
# pca.fit(features_train)
# features_train = pca.transform(features_train)
# features_test = pca.transform(features_test)

# # Only retain the N principal components that explain 95% of the variance # !!! OLD DELETE
# explained_variance_ratio = pca.explained_variance_ratio_
# cumulative_explained_variance = np.cumsum(explained_variance_ratio)
# n_components_95 = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
# features_train = features_train[:,:n_components_95]
# features_test = features_test[:,:n_components_95]


# =============================================================================
# Save the vision features
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'stimulus_features',
    'vision_features', args.model_name)
os.makedirs(save_dir, exist_ok=True)

file_name_train = 'vision_features_train.npy'
file_name_test = 'vision_features_test.npy'

np.save(os.path.join(save_dir, file_name_train), features_train)
np.save(os.path.join(save_dir, file_name_test), features_test)