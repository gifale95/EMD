"""Rank videos based on their frames pixel similarity.

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
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
#parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_videos', type=str)
args = parser.parse_args()


# =============================================================================
# Load the frame similarity RDMs
# =============================================================================
data_dir = os.path.join(args.project_dir, 'results', 'video_frames_similarity',
	'pixel_similarity')
videos_list = os.listdir(data_dir)
videos_list.sort()

frames_similarity = []
for v in videos_list:
	frames_similarity.append(np.load(os.path.join(data_dir, v)))


# =============================================================================
# Rank videos based on their frames similarity
# =============================================================================
# Rank videos in which second 2 is highly similar/dissimilar from second 1
score_1 = np.zeros(len(frames_similarity))
for v, video in enumerate(frames_similarity):
	s2_start = int(0 + np.round(len(video) / 3))
	s3_start = int(s2_start + np.round(len(video) / 3))
	video = video[s2_start:s3_start,:s2_start]
	score_1[v] = np.mean(video.flatten())
rank_1 = np.argsort(score_1)

# Rank videos in which second 3 is highly similar/dissimilar from seconds 1
# and 2
score_2 = np.zeros(len(frames_similarity))
for v, video in enumerate(frames_similarity):
	s2_start = int(0 + np.round(len(video) / 3))
	s3_start = int(s2_start + np.round(len(video) / 3))
	video = video[s3_start:,:s3_start]
	score_2[v] = np.mean(video.flatten())
rank_2 = np.argsort(score_2)




# Rank videos in which all 3 seconds are highly similar/dissimilar between
# each other



# !!! This ranking seems accurate at times but inaccurate other times.
	# I should also hand-pick most similar/dissimilar video!








