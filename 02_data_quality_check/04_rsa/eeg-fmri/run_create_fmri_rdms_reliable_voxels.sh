#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=fmri_rdms_reliable_voxels
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:10:00
#SBATCH --qos=prio

# Creating the parameters combinations
declare -a subjects
index=0
for s in `seq 1 10`
do
	subjects[$index]=$s
	((index=index+1))
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${subjects[$SLURM_ARRAY_TASK_ID]}
echo Subject: $sub

# Wait a bit so it doesn't crash
sleep 8

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_videos/code/02_data_quality_check/04_rsa_fusion

# Activating the Anaconda environment
source /scratch/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate eeg_encoding

# Running the job
python create_fmri_rdms_reliable_voxels.py --sub $sub
