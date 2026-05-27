#!/bin/bash
#SBATCH --mail-user=giffordale95@zedat.fu-berlin.de
#SBATCH --job-name=eye_pair_dec_data_split-test
#SBATCH --mail-type=end
#SBATCH --mem=1000
#SBATCH --time=01:00:00
#SBATCH --qos=prio

# Create the parameters combinations
declare -a sub_all
declare -a mean_centering_all
declare -a zscore_all
declare -a features_all
declare -a time_split_all
declare -a pseudo_trials_all

index=0

for s in `seq 1 2` ; do
	for c in 'baseline' ; do
		for z in '1' ; do
			for f in 'gaze' 'gaze_independent' 'pupil' 'all' 'all_independent' ; do
				for t in '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' ; do
					for p in '1' ; do
						sub_all[$index]=$s
						mean_centering_all[$index]=$c
						zscore_all[$index]=$z
						features_all[$index]=$f
						time_split_all[$index]=$t
						pseudo_trials_all[$index]=$p
						((index=index+1))
					done
				done
			done
		done
	done
done

# Extract the parameters
echo SLURM_ARRAY_JOB_ID: $SLURM_ARRAY_TASK_ID
sub=${sub_all[$SLURM_ARRAY_TASK_ID]}
mean_centering=${mean_centering_all[$SLURM_ARRAY_TASK_ID]}
zscore=${zscore_all[$SLURM_ARRAY_TASK_ID]}
features=${features_all[$SLURM_ARRAY_TASK_ID]}
time_split=${time_split_all[$SLURM_ARRAY_TASK_ID]}
pseudo_trials=${pseudo_trials_all[$SLURM_ARRAY_TASK_ID]}
echo Subject: $sub
echo Mean centering: $mean_centering
echo z-score: $zscore
echo Eye features: $features
echo Time split: $time_split
echo Pseudo trials: $pseudo_trials

# Wait a bit so it doesn't crash
sleep 8

# Change to the .py script directory
cd /home/giffordale95/projects/eeg_videos/code/02_data_quality_check/dataset_02/02_eye/02_pairwise_decoding

# Activate the Anaconda environment
source /home/giffordale95/anaconda3/etc/profile.d/conda.sh
conda activate general

# Run the job
python eye_pairwise_decoding.py --sub $sub --mean_centering $mean_centering --zscore $zscore --features $features --time_split $time_split --pseudo_trials $pseudo_trials --data_split 'test' 
