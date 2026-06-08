#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=EMD-03_encoding_models-04_stats
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=00:15:00
#SBATCH --qos=extended

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_moments_dataset/github/EMD/03_encoding_models

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Running the job
python 04_stats.py