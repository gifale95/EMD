#!/bin/bash
declare sessions=('01' '06')

declare subjects=('06')
declare data_dir='/home/ale/aaa_stuff/PhD/projects/eeg_videos/dataset/source_data/dataset_02/'

for s in ${subjects[@]} ; do
	for se in ${sessions[@]} ; do
		declare files_dir=$data_dir'sub-'$s'/ses-'$se
		cd $files_dir
		declare filelist=`ls | grep '.edf' | sed s/.edf//`
		for f in ${filelist[@]} ; do
			# Extract sample data
			edf2asc -s $f.edf
			mv $f.asc $f'_samples'.asc
			# Extract events and blinks
			edf2asc -e $f.edf
			cat $f.asc | grep 'TRIAL_START' > $f'_events'.asc
			cat $f.asc | grep 'SBLINK' > $f'_sblink'.asc
			cat $f.asc | grep 'EBLINK' > $f'_eblink'.asc
			rm $f.asc
		done
	done
done
