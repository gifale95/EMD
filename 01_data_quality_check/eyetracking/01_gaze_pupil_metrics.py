"""Run a pairwise decoding analysis on the eyetracking data for the 102 test
videos.

Parameters
----------
subject : int
    Used subject.
project_dir : str
    Directory of the project folder.

"""

import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subject', default=1, type=int)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Eyetracking metrics <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the eyetracking data for all videos
# =============================================================================
# Loop across eyetracking recording sessions
data_dir = os.path.join(args.project_dir, 'derivatives', 'eyetracking',
    f'sub-{args.subject:02}')
n_sessions = 8
eye = []
for ses in range(1, n_sessions+1):
    file_name = f'sub-{args.subject:02}_ses-{ses:02}_preprocessed_eyetracking.h5'
    eye.append(h5py.File(
        os.path.join(data_dir, file_name), 'r')['eyetracking'][:])
eye = np.concatenate(eye, 0)


# =============================================================================
# Get the time points
# =============================================================================
tmin = -0.1
tmax = 3.5
times = np.round(np.linspace(tmin, tmax, eye.shape[2]), 3)


# =============================================================================
# Gaze heatmaps
# =============================================================================
# Select samples within the video presentation interval (0 to 3s)
mask = (times >= 0) & (times <= 3)
eye_video = eye[:,:,mask]

# Compute the 2D histograms of gaze position
max_vis_deg = 5 # degrees of visual angle from the center of the screen
points_per_degree = 20
deg_per_point = 1 / points_per_degree # resolution of: 1 / 20 = 0.05° (~2 pixels)
size_heatmaps = int(max_vis_deg * points_per_degree * 2)
gaze_heatmap = np.zeros((size_heatmaps, size_heatmaps))
for v in tqdm(range(eye_video.shape[0])):
    for t in range(eye_video.shape[2]):
        if np.isnan(eye_video[v,0,t]) == False:
            x_coord = int(np.round((size_heatmaps / 2) + \
                (eye_video[v,0,t] / deg_per_point)))
            y_coord = int(np.round((size_heatmaps / 2) + \
                (eye_video[v,1,t] / deg_per_point)))
            gaze_heatmap[x_coord,y_coord] += 1


# =============================================================================
# Percentage of fixations as a function of visual angle
# =============================================================================
# Compute the Euclidean distance of the gaze coordinates from the center of the
# screen
euclDist = np.sqrt(
    np.power(eye_video[:,0], 2) + np.power(eye_video[:,1], 2)).flatten()

# X coordinates where you want to evaluate the CDF
visual_angles = np.linspace(0, 2, 1000)

# Compute the percentage of fixations as a function of visual angle
cdf_fixations = np.array([(euclDist <= va).mean() for va in visual_angles])


# =============================================================================
# Trial-average pupil size across time
# =============================================================================
avg_pupil_size = np.nanmean(eye[:,2,:], 0)


# =============================================================================
# Save the results
# =============================================================================
results = {
    'times': times,
    'gaze_heatmap': {
        'gaze_heatmap': gaze_heatmap,
        'max_vis_deg': max_vis_deg,
        'points_per_degree': points_per_degree,
        'deg_per_point': deg_per_point
    },
    'cdf_fixations': {
        'cdf_fixations': cdf_fixations,
        'visual_angles': visual_angles
    },
    'avg_pupil_size': avg_pupil_size
}

save_dir = os.path.join(args.project_dir, 'results', 'data_quality_check',
    'eyetracking', 'gaze_pupil_metrics')
os.makedirs(save_dir, exist_ok=True)

file_name = f'gaze_pupil_metrics_sub-{args.subject:02d}.npy'

np.save(os.path.join(save_dir, file_name), results)