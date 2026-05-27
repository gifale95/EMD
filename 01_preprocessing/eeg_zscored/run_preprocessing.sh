#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=preprocessing_eeg_videos_zscored
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=20:00:00
#SBATCH --qos=prio

# Creating the parameters combinations
declare -a sub_all
declare -a sfreq_all
declare -a mvnn_all
index=0

for s in '1' '2' '3' '4' '5' '6' ; do
	for f in '100' ; do
		for m in 'time' ; do
			sub_all[$index]=$s
			sfreq_all[$index]=$f
			mvnn_all[$index]=$m
			((index=index+1))
		done
	done
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[$SLURM_ARRAY_TASK_ID]}
sfreq=${sfreq_all[$SLURM_ARRAY_TASK_ID]}
mvnn=${mvnn_all[$SLURM_ARRAY_TASK_ID]}
echo Subject: $sub
echo Frequency: $sfreq
echo MVNN: $mvnn

# wait a bit so it doesn't crash
sleep 8

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_videos/code/01_preprocessing/dataset_02/eeg_zscored

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Running the job
python preprocessing.py --sub $sub --sfreq $sfreq --mvnn $mvnn
