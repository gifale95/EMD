#!/bin/bash

declare subjects=('01' '02')
declare units=('01')
declare sessions=('01' '02' '03' '04')
declare data_dir_local='/home/ale/aaa_stuff/PhD/projects/eeg_videos/dataset/source_data/'
declare data_dir_curta='/scratch/giffordale95/projects/eeg_videos/dataset/source_data/'

for s in ${subjects[@]} ; do
	for u in ${units[@]} ; do
		for se in ${sessions[@]} ; do
			declare files_dir=$data_dir_local'sub-'$s'/unit-'$u'/ses-'$se
			declare output_dir=$data_dir_curta'sub-'$s'/unit-'$u'/ses-'$se
			cd $files_dir
			declare filelist=`ls | grep '.edf' | sed s/.edf//`
			for f in ${filelist[@]} ; do
				sshpass -p k2gx2r52 scp -r $f'_samples.asc' giffordale95@curta.zedat.fu-berlin.de:$output_dir'/'$f'_samples.asc'
				sshpass -p k2gx2r52 scp -r $f'_events.asc' giffordale95@curta.zedat.fu-berlin.de:$output_dir'/'$f'_events.asc'
				sshpass -p k2gx2r52 scp -r $f'_sblink.asc' giffordale95@curta.zedat.fu-berlin.de:$output_dir'/'$f'_sblink.asc'
				sshpass -p k2gx2r52 scp -r $f'_eblink.asc' giffordale95@curta.zedat.fu-berlin.de:$output_dir'/'$f'_eblink.asc'
				echo Subject $s, unit $u, session $se, file $f
			done
		done
	done
done
