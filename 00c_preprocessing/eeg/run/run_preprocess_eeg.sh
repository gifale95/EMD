#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=EMD-00c_preprocessing-preprocess_eeg
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=20:00:00
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
cd /home/giffordale95/projects/eeg_moments_dataset/github/EMD/00c_preprocessing/eeg

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Running the job
python preprocess_eeg.py --subject $subject