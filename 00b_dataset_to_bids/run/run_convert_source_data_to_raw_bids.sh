#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=EMD-convert_source_data_to_raw_bids
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=02:00:00
#SBATCH --qos=extended

# Creating the parameters combinations
declare -a sub_all
index=0

for s in `seq 1 6` ; do
    sub_all[$index]=$s
    ((index=index+1))
done

# Extracting the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[$SLURM_ARRAY_TASK_ID]}
echo Subject: $sub

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_moments_dataset/github/EMD/00b_dataset_to_bids

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Running the job
python convert_source_data_to_raw_bids.py --sub $sub
