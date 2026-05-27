"""Compute the within-videos similarity by correlating their pixel value
frames.

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
from scipy.stats import pearsonr as corr


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
#parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()

# Printing the arguments
print('\n\n\n>>> EEG videos - Frames pixel similarity <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))


# =============================================================================
# Get the videos directories
# =============================================================================
video_dir = os.path.join(args.project_dir, 'stimuli_videos')
videos_list = os.listdir(video_dir)
videos_list.sort()

save_dir = os.path.join(args.project_dir, 'results', 'video_frames_similarity',
	'pixel_similarity')
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)


# =============================================================================
# Correlate the video frames
# =============================================================================
for v, video in enumerate(tqdm(videos_list)):
	# Extract the video frames
	frames = []
	vr = VideoReader(os.path.join(video_dir, video), ctx=cpu(0))
	total_frames = len(vr)
	frames_idx = np.arange(0, total_frames, dtype=int)
	for idx in frames_idx:
		frames.append(np.reshape(np.asarray(Image.fromarray(
			vr[idx].asnumpy())), -1))
	frames = np.asarray(frames)
	# Correlate the video frames
	frame_corr = np.ones((total_frames, total_frames))
	for v1 in range(total_frames):
		for v2 in range(v1):
			frame_corr[v1,v2] = corr(frames[v1], frames[v2])[0]
			frame_corr[v2,v1] = frame_corr[v1,v2]
	# Save the frames correlations
	file_name = 'pixel_similarity_video-' + format(v+1, '05')
	np.save(os.path.join(save_dir, file_name), frame_corr)
