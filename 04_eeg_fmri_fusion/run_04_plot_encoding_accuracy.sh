#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=04_eeg_fmri_fusion-04_plot_encoding_accuracy
#SBATCH --mail-type=end
#SBATCH --mem=4000
#SBATCH --time=01:40:00
#SBATCH --qos=standard

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
echo eeg_channel_policy: $eeg_channel_policy

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/giffordale95/projects/eeg_moments/code/04_eeg_fmri_fusion

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 04_plot_encoding_accuracy.py --eeg_channel_policy $eeg_channel_policy

