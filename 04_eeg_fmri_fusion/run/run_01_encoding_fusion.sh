#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=EMD-04_eeg_fmri_fusion-01_encoding_fusion
#SBATCH --mail-type=end
#SBATCH --mem=20000
#SBATCH --time=12:00:00
#SBATCH --qos=extended

# Creating the parameters combinations
declare -a eeg_subject_all
declare -a eeg_time_split_all
declare -a fmri_subject_all
declare -a fmri_hemi_all
index=0

for es in `seq 1 3` ; do
    for et in `seq 0 4` ; do
        for fs in `seq 1 10` ; do
            for fh in 'left' 'right' ; do
                eeg_subject_all[$index]=$es
                eeg_time_split_all[$index]=$et
                fmri_subject_all[$index]=$fs
                fmri_hemi_all[$index]=$fh
                ((index=index+1))
            done
        done
    done
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
eeg_subject=${eeg_subject_all[$SLURM_ARRAY_TASK_ID]}
eeg_time_split=${eeg_time_split_all[$SLURM_ARRAY_TASK_ID]}
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
fmri_hemi=${fmri_hemi_all[$SLURM_ARRAY_TASK_ID]}
echo eeg_subject: $eeg_subject
echo eeg_time_split: $eeg_time_split
echo fmri_subject: $fmri_subject
echo fmri_hemi: $fmri_hemi

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_moments_dataset/github/EMD/04_eeg_fmri_fusion

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Running the job
python 01_encoding_fusion.py --eeg_subject $eeg_subject --eeg_time_split $eeg_time_split --fmri_subject $fmri_subject --fmri_hemi $fmri_hemi