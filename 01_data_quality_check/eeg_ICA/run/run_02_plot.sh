#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=EMD-01_data_quality_check-eeg-02_plot_ICA
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:10:00
#SBATCH --qos=extended

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_moments_dataset/github/EMD/01_data_quality_check/eeg_ICA

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Running the job
python 02_plot.py