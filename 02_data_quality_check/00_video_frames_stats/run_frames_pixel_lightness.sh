#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=frames_pixel_lightness
#SBATCH --mail-type=end
#SBATCH --mem=500
#SBATCH --time=40:00:00
#SBATCH --qos=prio

# Creating the parameters combinations
declare -a video_all
index=0

for v in `seq 0 1101`
do
	video_all[$index]=$v
	((index=index+1))
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
video=${video_all[$SLURM_ARRAY_TASK_ID]}
echo Video: $video

# wait a bit so it doesn't crash
sleep 8

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_videos/code/02_data_quality_check/00_video_frames_stats

# Activating the Anaconda environment
source /scratch/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate eeg_encoding

# Running the job
python frames_pixel_lightness.py
