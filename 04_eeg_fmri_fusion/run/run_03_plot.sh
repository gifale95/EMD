#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=EMD-04_eeg_fmri_fusion-03_plot
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=20:00:00
#SBATCH --qos=extended

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_moments_dataset/github/EMD/04_eeg_fmri_fusion

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Running the job
python 03_plot.py