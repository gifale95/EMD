#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_moments_merge_pairwise_decoding_data_split-test
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:20:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a sub_all
declare -a channels_all
declare -a pseudo_trials_all

index=0

for s in `seq 1 6` ; do
	for c in 'O' 'P' 'T' 'C' 'F' 'all' ; do
		for p in '1' ; do
			sub_all[$index]=$s
			channels_all[$index]=$c
			pseudo_trials_all[$index]=$p
			((index=index+1))
		done
	done
done
# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[$SLURM_ARRAY_TASK_ID]}
channels=${channels_all[$SLURM_ARRAY_TASK_ID]}
pseudo_trials=${pseudo_trials_all[$SLURM_ARRAY_TASK_ID]}
echo Subject: $sub
echo EEG channels: $channels
echo Pseudo trials: $pseudo_trials

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/giffordale95/projects/eeg_videos/code/02_data_quality_check/dataset_02/01_eeg/02_pairwise_decoding

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python merge_pairwise_decoding.py --sub $sub --channels $channels --pseudo_trials $pseudo_trials --data_split 'test'
