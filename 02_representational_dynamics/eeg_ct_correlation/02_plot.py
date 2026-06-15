"""Plot the cross-temporal correlation analysis of EEG reponses.

Parameters
----------
subjects : list
    List of used subjects.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).

"""

import argparse
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6], type=list)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Plot save directory
# =============================================================================
save_dir = os.path.join(args.emd_dir, 'results',
    'representational_dynamics', 'eeg_ct_correlation', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the cross-temporal correlation results
# =============================================================================
# Load the cross-temporal correlation results
ct_corr = []
for s, sub in enumerate(args.subjects):
    data_dir = os.path.join(args.emd_dir, 'results',
        'representational_dynamics', 'eeg_ct_correlation', 'ct_correlation',
        f'ct_correlation_sub-{sub:02d}.npy')
    ct_corr.append(np.load(data_dir))
ct_corr = np.array(ct_corr)

# Load the time points
data_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg',
    f'sub-{args.subjects[0]:02}',
    f'sub-{args.subjects[0]:02}_eeg_metadata.npy')
times = np.load(data_dir, allow_pickle=True).item()['times']


# =============================================================================
# Plot parameters
# =============================================================================
fontsize = 20
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
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'


# =============================================================================
# Plot the cross-temporal correlation results
# =============================================================================
# Create the figure
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 10))
axs = np.reshape(axs, (-1))

# Loop across subjects
for s, sub in enumerate(args.subjects):

    # Plot the cross-temporal correlation results of each subject
    axs[s].imshow(ct_corr[s], cmap='Reds', vmin=0, vmax=0.5,
        aspect='equal', origin='lower') # !!! vmax=??? 

    # Plot title
    axs[s].set_title(f'Participant {sub}', fontsize=fontsize)

    # x-axis parameters
    if s in [3, 4, 5]:
        plot_times = np.array([0, 1, 2, 3])
        axs[s].set_xlabel('Time (s)', fontsize=fontsize)
        xticks = np.array([np.where(times == t)[0][0] for t in plot_times])
        axs[s].set_xticks(ticks=xticks, labels=plot_times)
        # axs[s].set_xlim(left=min(xticks), right=max(xticks))

    # y-axis parameters
    if s in [0, 3]:
        plot_times = np.array([0, 1, 2, 3])
        axs[s].set_ylabel("Time (s)", fontsize=fontsize)
        yticks = np.array([np.where(times == t)[0][0] for t in plot_times])
        axs[s].set_yticks(ticks=yticks, labels=plot_times)
        # axs[s].set_ylim(bottom=min(yticks), top=max(yticks))

# Save the figure
file_name = os.path.join(save_dir, 'eeg_ct_correlation.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()