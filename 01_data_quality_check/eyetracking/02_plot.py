"""Plot the EEG ERPs, noise ceiling, pairwise decoding, and save the number of
retained EEG trials.

Parameters
----------
subjects : list
    List of used subjects.
channels : list
    List of all channel types used for decoding.
project_dir : str
    Directory of the project folder.

"""

import argparse
import os
import h5py
from tqdm import tqdm
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import csv


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', default=[1, 2, 3, 4, 5, 6], type=list)
parser.add_argument('--channels', default=['O', 'P', 'T', 'C', 'F'], type=list)
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Plot save directory # !!!
# =============================================================================
save_dir = os.path.join(args.project_dir, 'results', 'data_quality_check',
    'eeg', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the gaze/pupil eyetracking metrics # !!!
# =============================================================================


# =============================================================================
# Load the pairwise decoding results # !!!
# =============================================================================
# Load the pairwise decoding results of all subjects and channel types
decoding = {}
for s, sub in enumerate(args.subjects):
    for c, chan in enumerate(args.channels):
        data_dir = os.path.join(args.project_dir, 'results',
            'data_quality_check', 'eeg', 'pairwise_decoding_rdms',
            f'rdms_sub-{sub:02d}_channels-{chan}.npy')
        rdms = np.load(data_dir)
        if s == 0 and c == 0:
            idx_tril = np.tril_indices(rdms.shape[0], k=-1)
        decoding[(sub, chan)] = np.mean(rdms[idx_tril], 0) * 100


# =============================================================================
# Save the number of retained EEG trials # !!!
# =============================================================================
# Get the retained trials
n_sessions = 8
tot_trials = np.array([8447, 8448, 8448, 8448, 8446, 8448])
retained_trials = [
    ["participant_id", "total_trials", "retained_trials",
    "percentage_retained_trials"]
]
for s, sub in enumerate(args.subjects):
    n_trials = 0
    for ses in range(n_sessions):
        n_trials += len(trial_number[s][f'ses-{ses+1:02}'])
    perc = np.round((n_trials / tot_trials[s]) * 100, 2)
    retained_trials.append([sub, int(tot_trials[s]), n_trials, float(perc)])

# Save the retained trials in a tsv file
file_name_tsv = os.path.join(save_dir, "retained_trials.tsv")
with open(file_name_tsv, "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(retained_trials)


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
colors = [
    (140/255, 90/255, 200/255),
    (125/255, 100/255, 210/255),
    (110/255, 115/255, 220/255),
    (95/255, 130/255, 225/255),
    (80/255, 145/255, 230/255),
    (65/255, 160/255, 235/255)
]


# =============================================================================
# Plot the 2D histograms of gaze position # !!!
# =============================================================================
fig, axs = plt.subplots(1, len(args.used_subj), sharex=True, sharey=True)
axs = np.reshape(axs, (-1))
for s, sub in enumerate(args.used_subj):
    # Plot the heatmap
    axs[s].imshow(gaze_heatmaps[s], cmap='inferno', aspect='equal')
    # Plot the video presentation box
    axs[s].plot([50, 150], [50, 50], 'w', [50, 150], [150, 150], 'w',
        [50, 50], [50, 150], 'w', [150, 150], [50, 150], 'w', linewidth=8)
    # Plot parameters
    if s == 0:
        axs[s].set_ylabel('Y°', fontsize=30)
    axs[s].set_xlabel('X°', fontsize=30)
    xticks = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 199]
    xlabels = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    plt.xticks(ticks=xticks, labels=xlabels)
    yticks = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 199]
    ylabels = [5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5]
    plt.yticks(ticks=yticks, labels=ylabels)
    title = 'Sub-' + format(sub)
    axs[s].set_title(title, fontsize=30)


# =============================================================================
# Plot the ERPs
# =============================================================================
# Create the figure
fig, axs = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(20, 30))
axs = np.reshape(axs, (-1))

# Loop across subjects
for s, sub in enumerate(args.subjects):

    # # Plot the stimulus onset and offset dashed lines
    axs[s].plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--', 
        linewidth=2, alpha=.25, label='_nolegend_')

    # Plot the ERPs of each subject
    axs[s].plot(times, np.transpose(erps[s]), color='k', linewidth=1,
        alpha=0.2)

    # Plot title
    axs[s].set_title(f'Participant {sub}', fontsize=fontsize)

    # x-axis parameters
    if s == len(args.subjects)-1:
        axs[s].set_xlabel('Time (s)', fontsize=fontsize)
        xticks = [0, .5, 1, 1.5, 2, 2.5, 3, 3.498]
        xlabels = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
        axs[s].set_xticks(ticks=xticks, labels=xlabels)
        axs[s].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    axs[s].set_ylabel("Voltage (µV)", fontsize=fontsize)
    ymin = np.nanmin(erps[s]) - abs((np.nanmin(erps[s])) * .1)
    ymax = np.nanmax(erps[s]) + abs((np.nanmax(erps[s])) * .1)
    axs[s].set_ylim(bottom=ymin, top=ymax)

# Save the figure
file_name = os.path.join(save_dir, 'eeg_erps.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the noise ceiling
# =============================================================================
# Plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(idx_ch))

# Create the figure
fig, axs = plt.subplots(6, 1, sharex=True, sharey=True, figsize=(20, 30))
axs = np.reshape(axs, (-1))

# Loop across subjects
for s, sub in enumerate(args.subjects):

    # Plot the stimulus onset and offset dashed lines
    axs[s].plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--', 
        linewidth=2, alpha=.25, label='_nolegend_')

    # Plot the noise ceiling of each subject and channel group
    for c in range(len(idx_ch)):
        nc = np.nanmean(noise_ceiling[s][idx_ch[c]], 0)
        axs[s].plot(times, nc, color=colors[c], linewidth=2, alpha=1,
            label=channel_type_names[c])

    # Plot title
    axs[s].set_title(f'Participant {sub}', fontsize=fontsize)

    # x-axis parameters
    if s == len(args.subjects)-1:
        axs[s].set_xlabel('Time (s)', fontsize=fontsize)
        xticks = [0, .5, 1, 1.5, 2, 2.5, 3, 3.498]
        xlabels = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
        axs[s].set_xticks(ticks=xticks, labels=xlabels)
        axs[s].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    axs[s].set_ylabel("Noise ceiling (%)", fontsize=fontsize)
    yticks = [0, 20, 40, 60, 80, 100]
    ylabels = [0, 20, 40, 60, 80, 100]
    axs[s].set_yticks(ticks=yticks, labels=ylabels)
    axs[s].set_ylim(bottom=0, top=90)

    # Legend
    if s == 0:
        axs[s].legend(loc=0, ncol=len(idx_ch), fontsize=fontsize,
            frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'noise_ceiling.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the pairwise decoding results
# =============================================================================
# Plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(channel_type_names))

# Create the figure
fig, axs = plt.subplots(6, 1, sharex=True, sharey=True, figsize=(20, 30))
axs = np.reshape(axs, (-1))

# Loop across subjects
for s, sub in enumerate(args.subjects):

    # Plot the stimulus onset/offset and decoding chance dashed lines
    axs[s].plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--',
        [-10, 10], [50, -50], 'k--', linewidth=2, alpha=.25,
        label='_nolegend_')

    # Plot the decoding results of each subject and channel group
    for c, chan in enumerate(args.channels):
        axs[s].plot(times, decoding[(sub, chan)], color=colors[c], linewidth=2,
            alpha=1, label=channel_type_names[c])

    # Plot title
    axs[s].set_title(f'Participant {sub}', fontsize=fontsize)

    # x-axis parameters
    if s == len(args.subjects)-1:
        axs[s].set_xlabel('Time (s)', fontsize=fontsize)
        xticks = [0, .5, 1, 1.5, 2, 2.5, 3, 3.498]
        xlabels = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
        axs[s].set_xticks(ticks=xticks, labels=xlabels)
        axs[s].set_xlim(left=min(times), right=max(times))

    # y-axis parameters
    axs[s].set_ylabel("Decoding accuracy (%)", fontsize=fontsize)
    yticks = [50, 60, 70, 80, 90, 100]
    ylabels = [0, 20, 40, 60, 80, 100]
    axs[s].set_yticks(ticks=yticks, labels=ylabels)
    axs[s].set_ylim(bottom=45, top=100)

    # Legend
    if s == 0:
        axs[s].legend(loc=0, ncol=len(args.channels), fontsize=fontsize,
            frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'pairwise_decoding.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()