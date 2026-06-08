"""Plot the EEG ERPs, noise ceiling, pairwise decoding, and save the number of
retained EEG trials.

Parameters
----------
subjects : list
    List of used subjects.
project_dir : str
    Directory of the project folder.

"""

import argparse
import os
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import csv


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
    'eyetracking', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the trial number
# =============================================================================
trial_number = []

for sub in args.subjects:
    data_dir = os.path.join(args.project_dir, 'derivatives', 'eyetracking',
        f'sub-{sub:02}', f'sub-{sub:02}_eyetracking_metadata.npy')
    metadata = np.load(data_dir, allow_pickle=True).item()
    trial_number.append(metadata['trial_number'])
    times = metadata['times']


# =============================================================================
# Load the gaze/pupil eyetracking metrics
# =============================================================================
gaze_heatmap = []
cdf_fixations = []
avg_pupil_size = []

for s, sub in enumerate(args.subjects):
    data_dir = os.path.join(args.project_dir, 'results', 'data_quality_check',
        'eyetracking', 'gaze_pupil_metrics',
        f'gaze_pupil_metrics_sub-{sub:02d}.npy')
    data = np.load(data_dir, allow_pickle=True).item()
    gaze_heatmap.append(data['gaze_heatmap'])
    cdf_fixations.append(data['cdf_fixations'])
    avg_pupil_size.append(data['avg_pupil_size'])


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
decoding_pupil = []

for s, sub in enumerate(args.subjects):
    data_dir = os.path.join(args.project_dir, 'results', 'data_quality_check',
        'eyetracking', 'pairwise_decoding_rdms')
    pupil_file = f'pupil_rdms_sub-{sub:02d}.npy'
    rdms_pupil = np.load(os.path.join(data_dir, pupil_file))
    if s == 0:
        idx_tril = np.tril_indices(rdms_pupil.shape[0], k=-1)
    decoding_pupil.append(np.mean(rdms_pupil[idx_tril], 0) * 100)


# =============================================================================
# Save the number of retained eyetracking trials
# =============================================================================
# Get the retained trials
n_sessions = 8
tot_trials = np.array([8448, 8448, 8448, 8345, 8432, 8448])
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


# =============================================================================
# Plot the 2D histograms of gaze position
# =============================================================================
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False

# Create the figure
fig, axs = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(15, 10))
axs = np.reshape(axs, (-1))

# Loop across subjects
for s, sub in enumerate(args.subjects):

    # Plot the 2D histograms
    axs[s].imshow(gaze_heatmap[s]['gaze_heatmap'], cmap='hot', aspect='equal')

    # Plot title
    axs[s].set_title(f'Participant {sub}', fontsize=fontsize)

    # x-axis parameters
    if s in [3, 4, 5]:
        axs[s].set_xlabel('X°', fontsize=fontsize)
        xticks = np.array([0, int(len(gaze_heatmap[s]['gaze_heatmap'])/2),
            len(gaze_heatmap[s]['gaze_heatmap'])-1])
        xlabels = np.array([-gaze_heatmap[s]['max_vis_deg'], 0,
            gaze_heatmap[s]['max_vis_deg']])
        axs[s].set_xticks(ticks=xticks, labels=xlabels)

    # y-axis parameters
    if s in [0, 3]:
        axs[s].set_ylabel('Y°', fontsize=fontsize)
        yticks = np.array([0, int(len(gaze_heatmap[s]['gaze_heatmap'])/2),
            len(gaze_heatmap[s]['gaze_heatmap'])-1])
        ylabels = np.array([gaze_heatmap[s]['max_vis_deg'], 0,
            -gaze_heatmap[s]['max_vis_deg']])
        axs[s].set_yticks(ticks=yticks, labels=ylabels)

# Save the figure
file_name = os.path.join(save_dir, 'gaze_heatmaps.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the percentage of fixations as a function of visual angle
# =============================================================================
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True

# Plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(cdf_fixations))

# Create the figure
fig = plt.figure(figsize=(7.5, 7.5))

# Plot the 1° threshold dashed line
plt.plot([1, 1], [0, 100], 'k--', linewidth=2, alpha=.25, label='_nolegend_')

# Plot the percentage of fixations as a function of visual angle
for s, sub in enumerate(args.subjects):
    visual_angles = cdf_fixations[s]['visual_angles']
    cdf_fix = cdf_fixations[s]['cdf_fixations'] * 100
    plt.plot(visual_angles, cdf_fix, color=colors[s], linewidth=2, alpha=1,
        label=f'Participant {sub}')

# x-axis parameters
plt.xlabel('Threshold', fontsize=fontsize)
xticks = [0, 0.5, 1, 1.5, 2]
xlabels = [0, 0.5, 1, 1.5, 2]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(visual_angles), right=max(visual_angles))

# y-axis parameters
plt.ylabel("Below threshold (%)", fontsize=fontsize)
yticks = [0, 20, 40, 60, 80, 100]
ylabels = [0, 20, 40, 60, 80, 100]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=0, top=100)

# Legend
plt.legend(loc=0, ncol=1, fontsize=fontsize, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'cdf_fixations.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the trial-average pupil size across the epoch time
# =============================================================================
# Plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(avg_pupil_size))

# Create the figure
fig = plt.figure(figsize=(7.5, 7.5))

# Plot the stimulus onset and offset dashed lines
# plt.plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--', 
#     linewidth=2, alpha=.25, label='_nolegend_')

# Plot the trial-average pupil size across the epoch time
for s, sub in enumerate(args.subjects):
    plt.plot(times, avg_pupil_size[s], color=colors[s], linewidth=2, alpha=1,
        label=f'Participant {sub}')

# x-axis parameters
plt.xlabel('Time (s)', fontsize=fontsize)
xticks = [0, 0.5, 1, 1.5, 2]
xlabels = [0, 0.5, 1, 1.5, 2]
# plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel("Pupil size (a.u.)", fontsize=fontsize)
# yticks = [0, 20, 40, 60, 80, 100]
# ylabels = [0, 20, 40, 60, 80, 100]
# plt.yticks(ticks=yticks, labels=ylabels)
# plt.ylim(bottom=0, top=100)

# Legend
plt.legend(loc=0, ncol=1, fontsize=fontsize, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'pupil_size.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the pairwise decoding results # !!! PLOT ALL SUBJECTS ON SAME PLOT !!!!!!!!!!!!!!!!
# =============================================================================
# Plot colors
colors = [(139/255, 0/255, 0/255), (0/255, 0/255, 0/255)]

# Create the figure
fig, axs = plt.subplots(6, 1, sharex=True, sharey=True, figsize=(20, 30))
axs = np.reshape(axs, (-1))

# Loop across subjects
for s, sub in enumerate(args.subjects):

    # Plot the stimulus onset/offset and decoding chance dashed lines
    axs[s].plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--',
        [-10, 10], [50, 50], 'k--', linewidth=2, alpha=.25,
        label='_nolegend_')

    # Plot the decoding results of each subject
    axs[s].plot(times, decoding_gaze[s], color=colors[0], linewidth=2,
        alpha=1, label='Gaze')
    axs[s].plot(times, decoding_pupil[s], color=colors[1], linewidth=2,
        alpha=1, label='Pupil')

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
    ylabels = [50, 60, 70, 80, 90, 100]
    axs[s].set_yticks(ticks=yticks, labels=ylabels)
    axs[s].set_ylim(bottom=45, top=80)

    # Legend
    if s == 0:
        axs[s].legend(loc=0, ncol=2, fontsize=fontsize, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'pairwise_decoding.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()