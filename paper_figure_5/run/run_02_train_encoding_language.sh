#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=EMD-03_encoding_models-02_train_encoding_language
#SBATCH --mail-type=end
#SBATCH --mem=30000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Creating the parameters combinations
declare -a subject_all
declare -a modality_all
index=0

for s in `seq 1 6` ; do
    for m in 'language' ; do
        subject_all[$index]=$s
        modality_all[$index]=$m
        ((index=index+1))
    done
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
subject=${subject_all[$SLURM_ARRAY_TASK_ID]}
modality=${modality_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $subject
echo modality: $modality

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_moments_dataset/github/EMD/03_encoding_models

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Running the job
python 02_train_encoding.py --subject $subject --modality $modality