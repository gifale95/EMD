#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=frames_pixel_similarity
#SBATCH --mail-type=end
#SBATCH --mem=500
#SBATCH --time=05:00:00
#SBATCH --qos=hiprio

# Creating the parameters combinations
declare -a sub_all
index=0

for s in `seq 1 1`
do
	sub_all[$index]=$s
	((index=index+1))
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[$SLURM_ARRAY_TASK_ID]}
echo Subject: $sub

# wait a bit so it doesn't crash
sleep 8

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_videos/code/02_data_quality_check/00_videos_correlations

# Activating the Anaconda environment
source /scratch/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate eeg_encoding

# Running the job
python frames_pixel_similarity.py
