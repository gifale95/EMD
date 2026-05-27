#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=rsa_eeg-eye_dsplit-test
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# Creating the parameters combinations
declare -a sub_all
declare -a channels_all
declare -a data_split_all
index=0
for s in `seq 1 2` ; do
	for c in 'O' 'P' 'T' 'C' 'F' 'all' ; do
		for d in 'test' ; do
			sub_all[$index]=$s
			channels_all[$index]=$c
			data_split_all[$index]=$d
			((index=index+1))
		done
	done
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[$SLURM_ARRAY_TASK_ID]}
channels=${channels_all[$SLURM_ARRAY_TASK_ID]}
data_split=${data_split_all[$SLURM_ARRAY_TASK_ID]}
echo Sub: $sub
echo EEG channels: $channels
echo Data split: $data_split

# Wait a bit so it doesn't crash
sleep 8

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_videos/code/02_data_quality_check/04_rsa/eeg-eye

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Running the job
python rsa.py --sub $sub --channels $channels --data_split $data_split
