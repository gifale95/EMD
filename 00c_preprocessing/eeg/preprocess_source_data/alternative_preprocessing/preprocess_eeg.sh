#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=preprocessing_eeg_moments
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=10:00:00
#SBATCH --qos=prio

# Creating the parameters combinations
declare -a sub_all
index=0

for s in '1' '2' '3' '4' '5' '6' ; do
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
cd /home/giffordale95/projects/eeg_videos/code/01_preprocessing/dataset_02/eeg

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Running the job
python preprocessing.py --sub $sub

