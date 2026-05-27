#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=rsa_fusion_subj-all
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# Creating the parameters combinations
declare -a all_fmri_voxels
index=0
for v in 'all' 'reliable'
do
	all_fmri_voxels[$index]=$v
	((index=index+1))
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_voxels=${all_fmri_voxels[$SLURM_ARRAY_TASK_ID]}
echo Subjects: All
echo fMRI voxels: $fmri_voxels

# Wait a bit so it doesn't crash
sleep 8

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_videos/code/02_data_quality_check/04_rsa_fusion

# Activating the Anaconda environment
source /scratch/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate eeg_encoding

# Running the job
python rsa_fusion.py --subjects 'all' --fmri_voxels $fmri_voxels
