#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=preprocessing_eye
#SBATCH --mail-type=end
#SBATCH --mem=3000
#SBATCH --time=03:00:00
#SBATCH --qos=hiprio

# Creating the parameters combinations
declare -a sub_all
declare -a mean_centering_all
index=0

for s in '1' '2' '3' ; do
	for c in 'baseline' ; do
		sub_all[$index]=$s
		mean_centering_all[$index]=$c
		((index=index+1))
	done
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[$SLURM_ARRAY_TASK_ID]}
mean_centering=${mean_centering_all[$SLURM_ARRAY_TASK_ID]}
echo Subject: $sub
echo Mean centering: $mean_centering

# wait a bit so it doesn't crash
sleep 8

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_videos/code/01_preprocessing/dataset_02/eye

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Running the job
python preprocessing.py --sub $sub --mean_centering $mean_centering
