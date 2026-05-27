for s in '01' '02' '03' '04' '05' '06' '07' '08' '09' '10' ; do
	output_dir='/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/MNI152/GLM/sub-'${s}'/ROIs'
#	output_dir='/home/ale/Downloads/data_jeffrey/fmri/bold_moments/derivatives/versionB/MNI152/GLM/sub-'${s}'/ROIs'

	if [ ! -d "$output_dir" ]; then
  		mkdir -p "$output_dir"
	fi
	aws s3 sync --no-sign-request s3://openneuro.org/ds005165/derivatives/versionB/MNI152/GLM/sub-${s}/ROIs /scratch/giffordale95/projects/eeg_moments/bold_moments_dataset/derivatives/versionB/MNI152/GLM/sub-${s}/ROIs
#	aws s3 sync --no-sign-request s3://openneuro.org/ds005165/derivatives/versionB/MNI152/GLM/sub-${s}/ROIs /home/ale/Downloads/data_jeffrey/fmri/bold_moments/derivatives/versionB/MNI152/GLM/sub-${s}/ROIs
	echo Sub-${s}
done
