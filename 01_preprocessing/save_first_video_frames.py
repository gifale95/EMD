import os
from scipy import io
from decord import VideoReader
from decord import cpu
from PIL import Image
from tqdm import tqdm

project_dir = '/home/ale/aaa_stuff/PhD/projects/eeg_videos'
output_dir = '/home/ale/Downloads/eeg_videos/first_video_frames'
subjects = [1]
sessions = [6]
runs = [1, 2, 15, 16]

video_dir = os.path.join(project_dir, 'stimuli_videos')
videos_list = os.listdir(video_dir)
videos_list.sort()

for sub in subjects:
	for ses in sessions:
		stim_order = io.loadmat(os.path.join(project_dir, 'dataset',
			'source_data', 'dataset_02', 'sub-'+format(sub, '02'), 'ses-'+
			format(ses, '02'), 'stim_order_sub-'+format(sub, '02')+
			'_sess-'+format(ses, '02')+'.mat'))['stim_order']
		for run in runs:
			stim_order_run = stim_order[:,run-1]
			save_dir = os.path.join(output_dir, 'sub-'+format(sub, '02'),
				'ses-'+format(ses, '02'), 'run-'+format(run, '02'))
			if not os.path.isdir(save_dir):
				os.makedirs(save_dir)
			for v, video in enumerate(stim_order_run):
				video_file = videos_list[video-1]
				vr = VideoReader(os.path.join(video_dir, video_file),
					ctx=cpu(0))
				img = Image.fromarray(vr[0].asnumpy())
				img_name = 'trial-' + format(v+1, '02') + '_video-' + \
					format(video) + '.jpg'
				img = img.save(os.path.join(save_dir, img_name))







