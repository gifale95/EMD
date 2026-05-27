"""ERPs of the EEG responses to videos.

Parameters
----------
sub : int
	Used subject.
baseline_correction : int
	Whether to baseline correct [1] or not [0] the data.
mvnn : str
	Type of MVNN applied to preprocess the data.
sfreq : int
	Downsampling frequency.
lowpass : float
	Lowpass filter frequency.
highpass : float
	Highpass filter frequency.
project_dir : str
	Directory of the project folder.


"""

import argparse
import os
import numpy as np
import mne
import matplotlib


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int) # [1, 2]
parser.add_argument('--baseline_correction', default=1, type=int) # ['0' '1']
parser.add_argument('--mvnn', default='time', type=str) # ['none' 'time']
parser.add_argument('--sfreq', default=50, type=int)
parser.add_argument('--lowpass', default=100, type=float)
parser.add_argument('--highpass', default=0.01, type=float)
parser.add_argument('--project_dir', default='/home/ale/aaa_stuff/PhD/projects/eeg_videos', type=str)
args = parser.parse_args()


# =============================================================================
# Load the biological EEG data
# =============================================================================
data_dir = os.path.join(args.project_dir, 'dataset', 'preprocessed_data',
	'dataset_02', 'eeg', 'sub-'+format(args.sub,'02'), 'mvnn-'+args.mvnn,
	'baseline_correction-'+format(args.baseline_correction, '02'), 'highpass-'+
	format(args.highpass)+'_lowpass-'+format(args.lowpass), 'sfreq-'+
	format(args.sfreq,'04'), 'preprocessed_data.npy')
data_dict = np.load(data_dir, allow_pickle=True).item()
times = data_dict['times']
ch_names = data_dict['ch_names']
info = data_dict['info']
preprocessed_data = data_dict['eeg_data']
stimuli_presentation_order = data_dict['stimuli_presentation_order']
del data_dict


# =============================================================================
# Select and average the EEG trials
# =============================================================================
for s in range(len(preprocessed_data)):
	if s == 0:
		eeg_data = preprocessed_data[s]
	else:
		eeg_data = np.append(eeg_data, preprocessed_data[s], 0)
del preprocessed_data

# Average the EEG data across trials
eeg_data = np.mean(eeg_data, 0)


# =============================================================================
# Convert the EEG data to MNE Evoked Array format
# =============================================================================
evoked_eeg = mne.EvokedArray(eeg_data, info, tmin=min(times))


# =============================================================================
# Plot the activity topomaps
# =============================================================================
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 20
# Interactive topomaps
#evoked_eeg.plot_topomap(times='interactive', show_names=True, time_unit='s',
	#size=10, res=300)

# Defined time points topomaps
plotted_times = np.arange(min(times), 3.25, .05)
title = 'Sub-' + format(args.sub) + ', MVNN-' + args.mvnn + \
	', baseline_correction-' + str(args.baseline_correction) + ', highpass-' + \
	str(args.highpass)
evoked_eeg.plot_topomap(times=plotted_times, time_unit='s', ncols=14,
	nrows='auto')

#plt.savefig('topo_erps_sub-01_mvnn-none_baseline_corr-00_highpass-0.01', dpi=100)
#plt.savefig('topo_erps_sub-01_mvnn-none_baseline_corr-00_highpass-0.01', dpi=100)

#plt.savefig('topo_erps_sub-01_mvnn-none_baseline_corr-01_highpass-0.01', dpi=100)
#plt.savefig('topo_erps_sub-01_mvnn-none_baseline_corr-01_highpass-0.01', dpi=100)

#plt.savefig('topo_erps_sub-01_mvnn-time_baseline_corr-00_highpass-0.01', dpi=100)
#plt.savefig('topo_erps_sub-01_mvnn-time_baseline_corr-00_highpass-0.01', dpi=100)

#plt.savefig('topo_erps_sub-01_mvnn-time_baseline_corr-01_highpass-0.01', dpi=100)
#plt.savefig('topo_erps_sub-01_mvnn-time_baseline_corr-01_highpass-0.01', dpi=100)