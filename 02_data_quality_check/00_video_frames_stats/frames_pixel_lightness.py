"""Compute the perceived lightness of each video frame:
https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color


Parameters
----------
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from decord import VideoReader
from decord import cpu
from PIL import Image


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
#parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> EEG videos - Frames perceived lightness <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get the videos directories
# =============================================================================
video_dir = os.path.join(args.project_dir, 'stimuli_videos')
videos_list = os.listdir(video_dir)
videos_list.sort()

save_dir = os.path.join(args.project_dir, 'results', 'video_frames_stats',
	'perceived_lightness')
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)


# =============================================================================
# Compute the perceived lightness
# =============================================================================
for v, video in enumerate(tqdm(videos_list)):

	# Extract the video frames
	frames = []
	vr = VideoReader(os.path.join(video_dir, video), ctx=cpu(0))
	total_frames = len(vr)
	frames_idx = np.arange(0, total_frames, dtype=int)
	for idx in frames_idx:
		frames.append(np.asarray(Image.fromarray(
			vr[idx].asnumpy())))
	frames = np.asarray(frames)

	# Luminance [0 1] and perceived lightness [0 100] empty arrays
	luminance = np.zeros((frames.shape[0], frames.shape[1],
		frames.shape[2]), dtype=np.float32)
	perceived_lightness = np.zeros((frames.shape[0], frames.shape[1],
		frames.shape[2]), dtype=np.float32)
	# Convert all sRGB 8 bit integer values to decimal 0.0-1.0
	frames = frames / 255
	for f in tqdm(range(frames.shape[0])):
		for h in range(frames.shape[1]):
			for w in range(frames.shape[2]):
				chan_stats = np.zeros(frames.shape[3], dtype=np.float32)
				for c in range(frames.shape[3]):
					# Convert a gamma encoded RGB to a linear value
					if frames[f,h,w,c] <= 0.04045:
						chan_stats[c] = frames[f,h,w,c] / 12.92
					else:
						chan_stats[c] = np.power(((
							frames[f,h,w,c] + 0.055) / 1.055), 2.4)
				# To find luminance apply the standard coefficients for sRGB
				luminance[f,h,w] = (0.2126 * chan_stats[0] + \
					0.7152 * chan_stats[1] + 0.0722 * chan_stats[2])
				# Compute the perceived lighness from the luminance
				if luminance[f,h,w] <= (216 / 24389):
					perceived_lightness[f,h,w] = luminance[f,h,w] * \
						(24389 / 27)
				else:
					perceived_lightness[f,h,w] = np.power(
						luminance[f,h,w], 1/3) * 100

	# Save the luminance and perceived_lightness
	video_stats = {
		'luminance': luminance,
		'perceived_lightness': perceived_lightness
		}
	file_name = 'luminance_perceived-lightness_video-' + format(v+1, '05')
	np.save(os.path.join(save_dir, file_name), video_stats)
