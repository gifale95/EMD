#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=EMD-01_data_quality_check-eeg-01_pairwise_decoding_ICA
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=15:00:00
#SBATCH --qos=extended

# Creating the parameters combinations
declare -a subject_all
declare -a channels_all
index=0

for s in `seq 1 6` ; do
    for c in 'O' 'P' 'T' 'C' 'F' ; do
        subject_all[$index]=$s
        channels_all[$index]=$c
        ((index=index+1))
    done
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
channels=${channels_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo channels: $channels

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_moments_dataset/github/EMD/01_data_quality_check/eeg_ICA

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Running the job
python 01_pairwise_decoding.py --subject $subject --channels $channels