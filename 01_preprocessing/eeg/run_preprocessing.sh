#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eeg_moments-eeg_preprocessing
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=20:00:00
#SBATCH --qos=extended

# Creating the parameters combinations
declare -a sub_all
declare -a sfreq_all
declare -a mvnn_all
index=0

for s in `seq 1 6` ; do
	for f in '100' ; do
		for m in 'none' ; do
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

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_moments/code/01_preprocessing/eeg

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Running the job
python preprocessing.py --sub $sub --sfreq $sfreq --mvnn $mvnn
