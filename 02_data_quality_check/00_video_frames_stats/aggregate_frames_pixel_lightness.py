"""Aggregate the frames pixel lightness results into one file.

Parameters
----------
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
#parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> EEG videos - Aggregate frames perceived lightness <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get the videos directories
# =============================================================================
video_dir = os.path.join(args.project_dir, 'stimuli_videos')
videos_list = os.listdir(video_dir)
videos_list.sort()

data_dir = os.path.join(args.project_dir, 'results', 'video_frames_stats',
	'perceived_lightness')


# =============================================================================
# Aggregate the data
# =============================================================================
luminance = {}
perceived_lightness = {}

for v in range(1102):
	file_name = 'luminance_perceived-lightness_video-' + format(v+1, '05') + \
		'.npy'
	stats = np.load(os.path.join(data_dir, file_name), allow_pickle=True).item()
	# For each frame, average the luminance / lightness across pixels
	luminance[v+1] = np.mean(np.reshape(stats['luminance'],
		(len(stats['luminance']), -1)), 1)
	perceived_lightness[v+1] = np.mean(np.reshape(stats['perceived_lightness'],
		(len(stats['perceived_lightness']), -1)), 1)


# =============================================================================
# Compute gray-screen luminance / perceived_lightness
# =============================================================================
gray_screen = np.asarray((159, 162, 166))

# Convert all sRGB 8 bit integer values to decimal 0.0-1.0
gray_screen = gray_screen / 255

chan_stats = np.zeros(len(gray_screen), dtype=np.float32)
for c in range(len(gray_screen)):
	# Convert a gamma encoded RGB to a linear value
	if gray_screen[c] <= 0.04045:
		chan_stats[c] = gray_screen[c] / 12.92
	else:
		chan_stats[c] = np.power(((
			gray_screen[c] + 0.055) / 1.055), 2.4)
# To find luminance apply the standard coefficients for sRGB
luminance_gray_screen = (0.2126 * chan_stats[0] + 0.7152 * chan_stats[1] + \
	0.0722 * chan_stats[2])
# Compute the perceived lighness from the luminance
if luminance_gray_screen <= (216 / 24389):
	perceived_lightness_gray_screen = luminance_gray_screen * \
		(24389 / 27)
else:
	perceived_lightness_gray_screen = np.power(
		luminance_gray_screen, 1/3) * 100


# =============================================================================
# Save the luminance and perceived_lightness
# =============================================================================
video_stats = {
	'luminance': luminance,
	'perceived_lightness': perceived_lightness,
	'luminance_gray_screen': luminance_gray_screen,
	'perceived_lightness_gray_screen': perceived_lightness_gray_screen
	}
file_name = 'luminance_perceived-lightness'
np.save(os.path.join(data_dir, file_name), video_stats)
