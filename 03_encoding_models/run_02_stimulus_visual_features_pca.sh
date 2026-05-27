#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=03_encoding_models-02_stimulus_visual_features_pca
#SBATCH --mail-type=end
#SBATCH --mem=180000
#SBATCH --time=03:00:00
#SBATCH --qos=prio

# Change to the .py script directory
cd /home/giffordale95/projects/eeg_moments/code/03_encoding_models

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 02_stimulus_visual_features_pca.py

