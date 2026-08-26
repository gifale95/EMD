#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=EMD-02_representational_dynamics-eeg_model_ct_rsa-03_stats
#SBATCH --mail-type=end
#SBATCH --mem=2000
#SBATCH --time=00:30:00
#SBATCH --qos=extended

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_moments_dataset/github/EMD/paper_figure_4

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate berg

# Running the job
python 03_stats.py