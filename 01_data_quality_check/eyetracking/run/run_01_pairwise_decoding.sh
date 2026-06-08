#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=EMD-01_data_quality_check-eyetracking-01_pairwise_decoding
#SBATCH --mail-type=end
#SBATCH --mem=500
#SBATCH --time=15:00:00
#SBATCH --qos=extended

# Creating the parameters combinations
declare -a subject_all
index=0

for s in `seq 1 6` ; do
    subject_all[$index]=$s
    ((index=index+1))
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_moments_dataset/github/EMD/01_data_quality_check/eyetracking

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Running the job
python 01_pairwise_decoding.py --subject $subject