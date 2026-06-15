"""Plot the partial correlation results.

Parameters
----------
subjects : list
    List of used subjects.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).

"""

import argparse
import os
import numpy as np
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
save_dir = os.path.join(args.emd_dir, 'results', 'encoding_models',
    'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the EEG channels and time points
# =============================================================================
data_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg',
    f'sub-{args.subjects[0]:02}', f'sub-{args.subjects[0]:02}_eeg_metadata.npy')
metadata = np.load(data_dir, allow_pickle=True).item()
times = metadata['times']
ch_names = metadata['ch_names']


# =============================================================================
# Load the partial correlation results
# =============================================================================
data_dir = os.path.join(args.emd_dir, 'results', 'encoding_models',
    'stats', 'stats.npy')
data = np.load(data_dir, allow_pickle=True).item()
partial_correlation = data['partial_correlation']
ci = data['ci']

res_types = ['total_variance_vision', 'total_variance_language',
    'unique_variance_vision', 'unique_variance_language']
res_labels = ['Vision', 'Language', 'Vision | Language', 'Language | Vision']


# =============================================================================
# EEG channel selection
# =============================================================================
channel_types = ['O', 'P', 'T', 'C', 'F']
channel_type_names = ['Occipital', 'Parietal', 'Temporal', 'Central',
    'Frontal']


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
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
matplotlib.rcParams['lines.markersize'] = 3
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['grid.linewidth'] = 2
matplotlib.rcParams['grid.alpha'] = .3
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'

# Plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(channel_type_names))


# =============================================================================
# Plot the partial correlation results (for each channel type)
# =============================================================================
# Create the figure
fig, axs = plt.subplots(len(res_types), 1, sharex=True, sharey=True,
    figsize=(20, 20))
axs = np.reshape(axs, (-1))

# Loop across result types
for r, res_type in enumerate(res_types):

    # Plot the stimulus onset/offset and correlation chance dashed lines
    axs[r].plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--',
        [-10, 10], [0, 0], 'k--', linewidth=2, alpha=.25, label='_nolegend_')

    # Plot the partial correlation results of each channel group
    for c, chan in enumerate(channel_types):
        res = np.mean(partial_correlation[f'{res_type}_{chan}'], 0)
        axs[r].plot(times, res, color=colors[c], linewidth=2, alpha=1,
            label=channel_type_names[c])
        # Plot the confidence intervals
        axs[r].fill_between(times, ci[f'{res_type}_{chan}'][0],
            ci[f'{res_type}_{chan}'][1], color=colors[c], alpha=.1)

    # Plot title
    axs[r].set_title(res_labels[r], fontsize=fontsize)

    # x-axis parameters
    if r == len(res_types)-1:
        axs[r].set_xlabel('Time (s)', fontsize=fontsize)
        xticks = [0, .5, 1, 1.5, 2, 2.5, 3, 3.498]
        xlabels = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
        axs[r].set_xticks(ticks=xticks, labels=xlabels)
        axs[r].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    axs[r].set_ylabel("Pearson's $r$", fontsize=fontsize)
    yticks = [0, 0.2, 0.4, 0.6]
    ylabels = [0, 0.2, 0.4, 0.6]
    axs[r].set_yticks(ticks=yticks, labels=ylabels)
    axs[r].set_ylim(bottom=-0.1, top=0.5)

    # Legend
    if r == 0:
        axs[r].legend(loc=0, ncol=len(channel_type_names), fontsize=fontsize,
            frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'partial_correlation.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()