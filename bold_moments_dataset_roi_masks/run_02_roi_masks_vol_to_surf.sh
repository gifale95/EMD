#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=02_roi_masks_vol_to_surf
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=03:00:00
#SBATCH --qos=hiprio

# Changing to the .py script directory
cd /home/giffordale95/projects/eeg_moments/code/bold_moments_dataset_roi_masks

# Activating the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Running the job
python3 02_roi_masks_vol_to_surf.py
