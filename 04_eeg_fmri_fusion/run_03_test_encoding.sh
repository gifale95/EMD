#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=04_eeg_fmri_fusion-03_test_encoding
#SBATCH --mail-type=end
#SBATCH --mem=10000
#SBATCH --time=01:10:00
#SBATCH --qos=standard

# Create the parameters combinations
declare -a fmri_subject_all
declare -a fmri_hemi_all
declare -a eeg_channel_policy_all
declare -a fmri_split_all
index=0
for s in '1' ; do
	for h in 'left' 'right' ; do
		for c in 'average' 'append' ; do
			for f in $(seq 1 21) ; do
				fmri_subject_all[$index]=$s
				fmri_hemi_all[$index]=$h
				eeg_channel_policy_all[$index]=$c
				fmri_split_all[$index]=$f
				((index=index+1))
			done
		done
	done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
fmri_subject=${fmri_subject_all[$SLURM_ARRAY_TASK_ID]}
fmri_hemi=${fmri_hemi_all[$SLURM_ARRAY_TASK_ID]}
eeg_channel_policy=${eeg_channel_policy_all[$SLURM_ARRAY_TASK_ID]}
fmri_split=${fmri_split_all[$SLURM_ARRAY_TASK_ID]}
echo fmri_subject: $fmri_subject
echo fmri_hemi: $fmri_hemi
echo eeg_channel_policy: $eeg_channel_policy
echo fmri_split: $fmri_split

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/giffordale95/projects/eeg_moments/code/04_eeg_fmri_fusion

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 03_test_encoding.py --fmri_subject $fmri_subject --fmri_hemi $fmri_hemi --eeg_channel_policy $eeg_channel_policy --fmri_split $fmri_split

