"""Plot the encoding model correlation and partial correlation results.

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
# parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
parser.add_argument('--emd_dir', default='/home/ale/aaa_stuff/science/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()


# =============================================================================
# Plot save directory
# =============================================================================
save_dir = os.path.join(args.emd_dir, 'results', 'encoding_models', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the EEG time points, channel names, and noise ceiling
# =============================================================================
noise_ceiling = []
for s, sub in enumerate(args.subjects):
    data_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg',
        f'sub-{args.subjects[0]:02}',
        f'sub-{args.subjects[0]:02}_eeg_metadata.npy')
    metadata = np.load(data_dir, allow_pickle=True).item()
    times = metadata['times']
    ch_names = metadata['ch_names']
    noise_ceiling.append(metadata['noise_ceiling'])
noise_ceiling = np.array(noise_ceiling)

# Average the noise ceiling across occipital and parietal EEG channels
idx_ch = []
for c, chan in enumerate(ch_names):
    if 'O' in chan or 'P' in chan:
        idx_ch.append(c)
noise_ceiling = np.mean(noise_ceiling[:,idx_ch], 1)

# Convert the noise ceiling to correlation values (now it reflects r² explained
# variance values)
noise_ceiling = np.sqrt(noise_ceiling/100)


# =============================================================================
# Load the partial correlation results
# =============================================================================
data_dir = os.path.join(args.emd_dir, 'results', 'encoding_models', 'stats',
    'stats.npy')
data = np.load(data_dir, allow_pickle=True).item()
partial_correlation = data['partial_correlation']
ci = data['ci']

res_types = ['variance_vision', 'variance_language', 'unique_variance_vision',
    'unique_variance_language']
res_labels = ['Vision', 'Language', 'Vision | Language', 'Language | Vision']


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
colors = [(139/255, 0/255, 0/255), (0/255, 0/255, 0/255)]


# =============================================================================
# Plot the correlation results
# =============================================================================
# Define the data to plot
res_types = ['variance_vision', 'variance_language']
labels = ['Vision', 'Language']

# Create the figure
fig = plt.figure(figsize=(20, 7.5))

# Plot the stimulus onset/offset and correlation chance dashed lines
plt.plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--',
    [-10, 10], [0, 0], 'k--', linewidth=2, alpha=.25, label='_nolegend_')

# Plot the noise ceiling
# plt.plot(times, np.mean(noise_ceiling, 0), 'k--', linewidth=2, alpha=.25,
#     label='Noise ceiling')

# Loop across vision and language models
for r, res_type in enumerate(res_types):

    # Plot the correlation results
    res = np.mean(partial_correlation[res_type], 0)
    plt.plot(times, res, color=colors[r], linewidth=2, alpha=1,
        label=labels[r])

    # Plot the confidence intervals
    plt.fill_between(times, ci[res_type][0], ci[res_type][1],
        color=colors[r], alpha=.1)

# x-axis parameters
plt.xlabel('Time (s)', fontsize=fontsize)
xticks = [0, .5, 1, 1.5, 2, 2.5, 3, 3.498]
xlabels = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ylabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-0.1, top=0.5)

# Legend
plt.legend(loc=2, ncol=len(res_types), fontsize=fontsize, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'correlation.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the partial correlation results
# =============================================================================
# Define the data to plot
res_types = ['unique_variance_vision', 'unique_variance_language']
labels = ['Vision | Language', 'Language | Vision']

# Create the figure
fig = plt.figure(figsize=(20, 7.5))

# Plot the stimulus onset/offset and correlation chance dashed lines
plt.plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--',
    [-10, 10], [0, 0], 'k--', linewidth=2, alpha=.25, label='_nolegend_')

# Plot the noise ceiling
# plt.plot(times, np.mean(noise_ceiling, 0), 'k--', linewidth=2, alpha=.25,
#     label='Noise ceiling')

# Loop across vision and language models
for r, res_type in enumerate(res_types):

    # Plot the correlation results
    res = np.mean(partial_correlation[res_type], 0)
    plt.plot(times, res, color=colors[r], linewidth=2, alpha=1,
        label=labels[r])

    # Plot the confidence intervals
    plt.fill_between(times, ci[res_type][0], ci[res_type][1],
        color=colors[r], alpha=.1)

# x-axis parameters
plt.xlabel('Time (s)', fontsize=fontsize)
xticks = [0, .5, 1, 1.5, 2, 2.5, 3, 3.498]
xlabels = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel("Partial Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ylabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-0.1, top=0.5)

# Legend
plt.legend(loc=2, ncol=len(res_types), fontsize=fontsize, frameon=False)

# Save the figure
file_name = os.path.join(save_dir, 'partial_correlation.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()