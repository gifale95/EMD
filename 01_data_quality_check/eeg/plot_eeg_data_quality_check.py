"""Plot the EEG ERPs, NCSNR, and noise ceiling.

Parameters
----------
subjects : list
    List of used subjects.
project_dir : str
    Directory of the project folder.

"""

Also plot the number of trials dropped (also for eyetracking)

import argparse
import os
import h5py
from tqdm import tqdm
import numpy as np
from scipy.stats import zscore
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6], type=list)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Plot save directory
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'data_quality_check',
    'eeg', 'erps_and_noise_ceiling')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the EEG data, and average it into the ERPs
# =============================================================================
sessions = 8

erps = []
for sub in tqdm(args.subjects):
    data_dir = os.path.join(args.project_dir, 'derivatives', 'eeg',
    f'sub-{sub:02}')
    data_ses = []
    for ses in range(1, sessions+1):
        data_dir_ses = os.path.join(data_dir,
            f'sub-{sub:02}_ses-{ses:02}_preprocessed_eeg.h5')
        data_ses.append(np.mean(h5py.File(data_dir_ses, 'r')['eeg'][:], 0))
    erps.append(np.mean(np.asarray(data_ses), 0))
    del data_ses
erps = np.asarray(erps)


# =============================================================================
# Load the NCSNR and noise ceiling
# =============================================================================
ncsnr = []
noise_ceiling = []
for sub in args.subjects:
    data_dir = os.path.join(args.project_dir, 'derivatives', 'eeg',
    f'sub-{sub:02}', f'sub-{sub:02}_eeg_metadata.npy')
    metadata = np.load(data_dir, allow_pickle=True).item()
    ncsnr.append(metadata['ncsnr'])
    noise_ceiling.append(metadata['noise_ceiling'])
    times = metadata['times']
    ch_names = metadata['ch_names']
ncsnr = np.asarray(ncsnr)
noise_ceiling = np.asarray(noise_ceiling)


# =============================================================================
# EEG channel selection
# =============================================================================
channel_types = ['O', 'P', 'T', 'C', 'FA']
channel_types_names = ['Occipital', 'Parietal', 'Temporal', 'Central',
    'Frontal']

idx_ch = []
for ch_type in channel_types:
    idx = []
    for c, chan in enumerate(ch_names):
        if ch_type != 'FA':
            if ch_type in chan:
                idx.append(c)
        else:
            if 'F' in chan or 'A' in chan:
                idx.append(c)
    idx_ch.append(np.asarray(idx))
    del idx


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 25
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams["font.weight"] = "normal"
matplotlib.rcParams["axes.labelweight"] = "normal"
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['xtick.major.width'] = 0
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.width'] = 0
matplotlib.rcParams['ytick.major.size'] = 5
matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
colors = [
    (120/255, 81/255, 169/255),
    (109/255, 86/255, 180/255),
    (97/255, 92/255, 191/255),
    (84/255, 99/255, 203/255),
    (70/255, 108/255, 215/255),
    (65/255, 105/255, 225/255),
]
colors = [
    (140/255, 90/255, 200/255),
    (125/255, 100/255, 210/255),
    (110/255, 115/255, 220/255),
    (95/255, 130/255, 225/255),
    (80/255, 145/255, 230/255),
    (65/255, 160/255, 235/255)
]

# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors


# =============================================================================
# Plot the ERPs
# =============================================================================
# Create the figure
fig = plt.figure(figsize=(10, 20))

# Plot the stimulus onset and offset dashed lines
# plt.plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--', linewidth=2,
#     alpha=.25, label='_nolegend_')

# Plot the ERPs of each subject
for s, sub in enumerate(args.subjects):
    data = erps[s,idx_ch[4]] - (0.00005 * s)
    plt.plot(times, np.transpose(data), color=colors[s], linewidth=1,
        alpha=0.2)

# x-axis parameters
plt.xlabel('Time (s)', fontsize=fontsize)
xticks = [0, .5, 1, 1.5, 2, 2.5, 3, 3.498]
xlabels = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
# plt.ylabel("Pearson's $r$", fontsize=fontsize)
# yticks = [0, 0.2, 0.4, 0.6, 0.8]
# ylabels = [0, 0.2, 0.4, 0.6, 0.8]
# plt.yticks(ticks=yticks, labels=ylabels)
# plt.ylim(bottom=-10, top=10)

# Save the figure
file_name = os.path.join(save_dir, 'eeg_erps.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)
axs = np.reshape(axs, (-1))
for s, sub in enumerate(args.subjects):
    # Plot baseline and stimulus onset/offset dashed lines
    axs[s].plot([-5, 5], [0, 0], 'k--', [0, 0], [10, -10], 'k--', [3, 3], [10, -10],
        'k--', label='_nolegend_', linewidth=3)
    # Plot the ERPs
    min_val = 0
    max_val = 0
    for c in idx_ch:
        axs[s].plot(times, np.mean(eeg_data[s,c], 0), linewidth=2)
        if min(np.mean(eeg_data[s,c], 0)) < min_val:
            min_val = min(np.mean(eeg_data[s,c], 0))
        if max(np.mean(eeg_data[s,c], 0)) > max_val:
            max_val = max(np.mean(eeg_data[s,c], 0))
    # Plot parameters
    axs[s].set_ylabel('EEG voltage', fontsize=20)
    axs[s].set_ylim(bottom=min_val, top=max_val)
    title = 'Sub-' + format(sub) + ', MVNN-' + args.mvnn + \
        ', baseline_correction-' + str(args.baseline_correction) + ', highpass-' + \
        str(args.highpass)
    axs[s].set_title(title, fontsize=20)
    if s == 1:
        axs[s].set_xlabel('Time (s)', fontsize=20)
        xticks = np.arange(-.2, 3.51, .2)
        xlabels = np.round(np.arange(-.2, 3.51, .2), 1)
        plt.xticks(ticks=xticks, labels=xlabels, fontsize=20)
        plt.xlim(left=min(times), right=max(times))
        plt.legend(channel_types_names, fontsize=20, loc=8, ncol=6,
            frameon=True, bbox_to_anchor=(0.5, -.3))

#plt.savefig('plot_erps_mvnn-none_baseline_corr-00_highpass-0.01', dpi=100)
#plt.savefig('plot_erps_mvnn-none_baseline_corr-01_highpass-0.01', dpi=100)
#plt.savefig('plot_erps_mvnn-time_baseline_corr-00_highpass-0.01', dpi=100)
#plt.savefig('plot_erps_mvnn-time_baseline_corr-01_highpass-0.01', dpi=100)