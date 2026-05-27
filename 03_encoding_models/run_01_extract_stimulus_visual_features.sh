#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=03_encoding_models-01_extract_stimulus_visual_features
#SBATCH --mail-type=end
#SBATCH --mem=5000
#SBATCH --time=00:40:00
#SBATCH --qos=hiprio

# Change to the .py script directory
cd /home/giffordale95/projects/eeg_moments/code/03_encoding_models

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python 01_extract_stimulus_visual_features.py

