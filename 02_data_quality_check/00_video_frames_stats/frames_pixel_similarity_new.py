"""Compute the within-videos similarity by correlating their pixel value
frames.

Parameters
----------
video : int
	Used video condition.
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
parser.add_argument('--video', default=766, type=int) # [0, 1101]
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
#parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()

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
video_file = videos_list[args.video]


# =============================================================================
# Extract the video frames
# =============================================================================
frames = []
vr = VideoReader(os.path.join(video_dir, video_file), ctx=cpu(0))
total_frames = len(vr)
frames_idx = np.arange(0, total_frames, dtype=int)

for idx in frames_idx:
	frames.append(np.asarray(Image.fromarray(
		vr[idx].asnumpy())))
frames = np.asarray(frames)


# =============================================================================
# Correlate the video frames
# =============================================================================
frame_corr = np.ones((total_frames, total_frames, frames.shape[1],
	frames.shape[2]))
for d1 in tqdm(range(frames.shape[1])): # 49
	for d2 in range(frames.shape[2]): # 12
		for f1 in range(total_frames): # 89
			for f2 in range(f1): # 87
				print(f2)
				frame_corr[f1,f2,d1,d2] = corr(frames[f1,d1,d2],
					frames[f2,d1,d2])[0]
				frame_corr[f2,f1,d1,d2] = frame_corr[f1,f2,d1,d2]
frame_corr = np.nanmean(np.nanmean(frame_corr, 2), 2)


# =============================================================================
# Save the frames correlations
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'video_frames_similarity',
	'pixel_similarity_new')
if not os.path.isdir(save_dir):
	os.makedirs(save_dir)
file_name = 'pixel_similarity_video-' + format(args.video+1, '05')
np.save(os.path.join(save_dir, file_name), frame_corr)
