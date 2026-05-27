#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=05_eeg_fmri_model_fusion-01_eeg_pca
#SBATCH --mail-type=end
#SBATCH --mem=15000
#SBATCH --time=00:10:00
#SBATCH --qos=hiprio

# Create the parameters combinations
declare -a eeg_channel_policy_all
index=0
for c in 'average' 'append' ; do
	eeg_channel_policy_all[$index]=$c
	((index=index+1))
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
eeg_channel_policy=${eeg_channel_policy_all[$SLURM_ARRAY_TASK_ID]}
echo subject: $eeg_channel_policy

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/giffordale95/projects/eeg_moments/code/05_eeg_fmri_model_fusion

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 01_eeg_pca.py --eeg_channel_policy $eeg_channel_policy

