"""Plot the results from the cross-temporal RSA analysis between EEG AlexNet
RDMs.

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
from scipy.stats import spearmanr
from scipy.stats import linregress


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
save_dir = os.path.join(args.emd_dir, 'results', 'representational_dynamics',
    'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the cross-temporal correlation results
# =============================================================================
# Loop across AlexNet layers
ct_rsa = {}
layers = [
    'features_2',
    'features_5',
    'features_7',
    'features_9',
    'features_12',
    'classifier_2',
    'classifier_5',
    'classifier_6'
]

# Loop across EEG subjects
for s, sub in enumerate(args.subjects):

    # Load the CT-RSA results
    data_dir = os.path.join(args.emd_dir, 'results',
        'representational_dynamics', 'ct_rsa', f'ct_rsa_sub-{sub:02d}.npy')
    ct_rsa_sub = np.load(data_dir, allow_pickle=True).item()

    # Store the CT-RSA results of each subject
    for layer in layers:
        if s == 0:
            ct_rsa[layer] = []
        ct_rsa[layer].append(ct_rsa_sub[layer])

# Average the results across subjects
for layer in layers:
    ct_rsa[layer] = np.mean(ct_rsa[layer], 0)

# Reshape the results to (Stimulus time, EEG time)
for layer in layers:
    ct_rsa[layer] = np.transpose(ct_rsa[layer])


# =============================================================================
# Get the time points
# =============================================================================
# EEG time points
min_times = 0
max_times = 3
n_bins_eeg = ct_rsa[layer].shape[1]
times_eeg = np.linspace(min_times, max_times, n_bins_eeg)

# AlexNet time points
n_times_alex = ct_rsa[layers[0]].shape[0]
times_alex = np.linspace(0, 3, n_times_alex)


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
# Plot the cross-temporal RSA results (2D heatmap)
# =============================================================================
# Normalize the CT-RSA scores of each EEG time point in the range [0, 1],
# to emphasize the differences in alignment between AlexNet timepoints at each
# EEG timepoint
ct_rsa_norm = {}
for layer in layers:
    rsa = ct_rsa[layer]
    rsa_min = rsa.min(axis=0, keepdims=True)
    rsa_max = rsa.max(axis=0, keepdims=True)
    ct_rsa_norm[layer] = (rsa - rsa_min) / (rsa_max - rsa_min)

# Select the layer to plot
plot_layer = 'features_2'

# Create the figure
matplotlib.rcParams['axes.spines.left'] = False
matplotlib.rcParams['axes.spines.bottom'] = False
fig = plt.figure(figsize=(7.5, 7.5))

# Plot the CT-RSA results of each AlexNet layer
plt.imshow(ct_rsa_norm[plot_layer], aspect='auto', cmap='cividis',
    origin='lower')

# x-axis parameters
plt.xlabel('Time EEG (s)', fontsize=fontsize)
xticks = [0, 10, 21, 31]
xlabels = [0, 1, 2, 3]
plt.xticks(ticks=xticks, labels=xlabels)
# plt.xlim(left=0, right=len(times_eeg))

# y-axis parameters
plt.ylabel("Time stimulus (s)", fontsize=fontsize)
yticks = [0, 10, 21, 31]
ylabels = [0, 1, 2, 3]
plt.yticks(ticks=yticks, labels=ylabels)
# plt.ylim(bottom=0, top=len(times_alex))

# Colorbar
# plt.colorbar(label='Normalized CT-RSA score', fraction=0.046, pad=0.04)

# Save the figure
file_name = os.path.join(save_dir, '2d_heatmap.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the cross-temporal RSA results (best model time point)
# =============================================================================
# Get the best model time points for each EEG time point (based on the average
# of the top-5 model time points to get a more robust estimate)
ct_rsa_best_model_time = {}
for layer in layers:
    rsa = ct_rsa[layer]
    idx_best = np.mean(np.argsort(rsa, 0)[-5:], 0) / n_times_alex * max_times
    ct_rsa_best_model_time[layer] = idx_best
    del idx_best

# Compute the correlation between EEG and AlexNet time points
corr_eeg_model_times = {}
for layer in layers:
    corr_eeg_model_times[layer] = spearmanr(times_eeg,
        ct_rsa_best_model_time[layer])[0]

# Fit a regression line between EEG and AlexNet time points
regression = {}
for layer in layers:
    regression[layer] = linregress(times_eeg, ct_rsa_best_model_time[layer])

# Select the layer to plot
plot_layer = 'features_2'

# Create the figure
matplotlib.rcParams['axes.spines.left'] = True
matplotlib.rcParams['axes.spines.bottom'] = True
color = (139/255, 0/255, 0/255)
fig = plt.figure(figsize=(7.5, 7.5))

# Plot the CT-RSA results
plt.plot(times_eeg, ct_rsa_best_model_time[plot_layer], color=color,
    linewidth=2, alpha=1, label='_nolegend_')

# Plot the correlation between EEG and AlexNet time points
plt.text(0.25, 2.8, f'$ρ$ = {corr_eeg_model_times[plot_layer]:.2f}', color='k',
    fontsize=fontsize)

# Plot the regression line between EEG and AlexNet time points
slope = regression[plot_layer].slope
intercept = regression[plot_layer].intercept
x_fit = np.array([min_times, max_times])
y_fit = intercept + slope * x_fit
plt.plot(x_fit, y_fit, color='k', linewidth=2, linestyle='--',
    label=f'$y$ = {slope:.2f}$x$ + {intercept:.2f}', alpha=0.5)

# x-axis parameters
plt.xlabel('Time EEG (s)', fontsize=fontsize)
xticks = [0, 1, 2, 3]
xlabels = [0, 1, 2, 3]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min_times, right=max_times)

# y-axis parameters
plt.ylabel("Most similar stimulus time (s)", fontsize=fontsize)
yticks = [0, 1, 2, 3]
ylabels = [0, 1, 2, 3]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=min_times, top=max_times)

# Legend
plt.legend(frameon=False, loc=0)

# Save the figure
file_name = os.path.join(save_dir, 'best_model_timepoints.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()