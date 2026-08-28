"""Compute the luminance of each video frame:
https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color


Parameters
----------
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image
from decord import VideoReader, cpu


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

# Printing the arguments
print('\n\n\n>>> Compute luminance <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Loop across splits, videos, and video frames
# =============================================================================
# Empty luminance dictionaries
luminance = {}


# Loop across train/test splits
splits = ['train', 'test']
for split in splits:

    # Loop across videos
    video_dir = os.path.join(args.emd_dir, 'stimuli', 'stimuli', split)
    video_list = os.listdir(video_dir)
    video_list.sort()
    for v, video in enumerate(tqdm(video_list)):
        luminance[video[:4]] = []

        # Read the video
        vr = VideoReader(os.path.join(video_dir, video), ctx=cpu(0))

        # Loop across frames
        for f, frame in enumerate(vr):

            # Load the frame image
            img = np.asarray(Image.fromarray(
                frame.asnumpy()).convert('RGB')) / 255.0


# =============================================================================
# Compute the luminance
# =============================================================================
            # Convert sRGB to linear
            linear = np.where(
                img <= 0.04045,
                img / 12.92,
                ((img + 0.055) / 1.055) ** 2.4
            )

            # Compute luminance
            rgb_to_y = np.asarray([0.2126, 0.7152, 0.0722])
            lum = linear @ rgb_to_y  

            # Store the luminance
            luminance[video[:4]].append(np.mean(lum))

        # Format the luminance to numpy arrays
        luminance[video[:4]] = np.array(luminance[video[:4]])


# =============================================================================
# Save the luminance
# =============================================================================
save_dir = os.path.join(args.emd_dir, 'results', 'data_quality_check',
    'eyetracking', 'image_luminance')
os.makedirs(save_dir, exist_ok=True)

file_name = 'image_luminance.npy'

np.save(os.path.join(save_dir, file_name), luminance)