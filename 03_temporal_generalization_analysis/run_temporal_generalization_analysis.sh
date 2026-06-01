#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=temp_gen_analysis
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=04:00:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a sub_all
declare -a mvnn_all
declare -a channels_all
declare -a time_split_all
index=0
for s in `seq 1 2` ; do
	for m in 'none' 'time' ; do
		for c in 'OP' 'all' ; do
			for t in `seq 1 185` ; do
				sub_all[$index]=$s
				mvnn_all[$index]=$m
				channels_all[$index]=$c
				time_split_all[$index]=$t
				((index=index+1))
			done
		done
	done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[$SLURM_ARRAY_TASK_ID]}
mvnn=${mvnn_all[$SLURM_ARRAY_TASK_ID]}
channels=${channels_all[$SLURM_ARRAY_TASK_ID]}
time_split=${time_split_all[$SLURM_ARRAY_TASK_ID]}
echo Subject: $sub
echo MVNN: $mvnn
echo Channels: $channels
echo Time split: $time_split

# Wait a bit so it doesn't crash
sleep 8

# Chang to the .py script directory
cd /home/giffordale95/projects/eeg_videos/code/02_data_quality_check/dataset_02/01_eeg/03_temporal_generalization_analysis

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python temporal_generalization_analysis.py --sub $sub --mvnn $mvnn --channels $channels --time_split $time_split
