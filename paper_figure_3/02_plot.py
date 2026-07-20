"""Plot the EEG ERPs, NCSNR, pairwise decoding, and save the number of retained
EEG trials.

Parameters
----------
subjects : list
    List of used subjects.
channels : list
    List of all channel types used for decoding.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).

"""

import argparse
import os
import h5py
from tqdm import tqdm
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import csv
import mne
from mne.channels.layout import _find_topomap_coords
from mne.viz.topomap import _make_head_outlines


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
save_dir = os.path.join(args.emd_dir, 'results', 'data_quality_check',
    'eeg', 'plots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Load the EEG data, and average it into the ERPs
# =============================================================================
sessions = 8

erps = []
for sub in tqdm(args.subjects):
    data_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg',
        f'sub-{sub:02}')
    data_ses = []
    for ses in range(1, sessions+1):
        data_dir_ses = os.path.join(data_dir,
            f'sub-{sub:02}_ses-{ses:02}_preprocessed_eeg.h5')
        data_ses.append(np.mean(h5py.File(data_dir_ses, 'r')['eeg'][:], 0))
    erps.append(np.mean(np.asarray(data_ses), 0))
    del data_ses
erps = np.asarray(erps)

# Convert the ERPs from volts to microvolts
erps = erps * 1e6


# =============================================================================
# Load the EEG channel names, times, NCSNR, and noise ceiling
# =============================================================================
ncsnr = []
noise_ceiling = []
trial_number = []

for sub in args.subjects:
    data_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg',
        f'sub-{sub:02}', f'sub-{sub:02}_eeg_metadata.npy')
    metadata = np.load(data_dir, allow_pickle=True).item()
    ncsnr.append(metadata['ncsnr'])
    noise_ceiling.append(metadata['noise_ceiling'])
    trial_number.append(metadata['trial_number'])
    times = metadata['times']
    ch_names = metadata['ch_names']
ncsnr = np.asarray(ncsnr)
noise_ceiling = np.asarray(noise_ceiling)


# =============================================================================
# Load the pairwise decoding results
# =============================================================================
# Load the pairwise decoding results of all subjects
decoding = []
for s, sub in enumerate(args.subjects):
    data_dir = os.path.join(args.emd_dir, 'results',
        'data_quality_check', 'eeg', 'pairwise_decoding_rdms',
        f'rdms_sub-{sub:02d}_channels-all.npy')
    rdms = np.load(data_dir)
    if s == 0:
        idx_tril = np.tril_indices(rdms.shape[0], k=-1)
    decoding.append(np.mean(rdms[idx_tril], 0) * 100)
decoding = np.array(decoding)


# =============================================================================
# Save the number of retained EEG trials
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


# =============================================================================
# Plot the ERPs
# =============================================================================
# Channel positions -> 2D
montage = mne.channels.make_standard_montage('standard_1005')
ch_pos = montage.get_positions()['ch_pos']

# O9/O10 are absent from standard_1005 (inferior-occipital ring); borrow neighbors
missing_fallback = {'O9': 'O1', 'O10': 'O2'}
coord_names = [c if c in ch_pos else missing_fallback[c] for c in ch_names]
info = mne.create_info(list(dict.fromkeys(coord_names)), 1000, 'eeg')
info.set_montage(montage)
pos2d = _find_topomap_coords(info, picks='eeg') # azimuthal-equidistant projection
name_to_xy = dict(zip(info.ch_names, pos2d))
xy = np.array([name_to_xy[c] for c in coord_names]) # Shape: (n_channels, 2)

# 2D position -> color (bilinear blend of 4 corner colors)
xy_n = (xy - xy.min(0)) / (np.ptp(xy, 0) + 1e-12) # np.ptp: ndarray.ptp removed in numpy 2.0
corners = {'ll': np.array([0.85, 0.10, 0.10]),   # posterior-left   red
           'lr': np.array([0.95, 0.75, 0.10]),   # posterior-right  yellow
           'ul': np.array([0.10, 0.25, 0.80]),   # anterior-left    blue
           'ur': np.array([0.10, 0.70, 0.35])}   # anterior-right   green
u, v = xy_n[:, 0][:, None], xy_n[:, 1][:, None]
colors = ((1 - u) * (1 - v) * corners['ll'] + u * (1 - v) * corners['lr']
          + (1 - u) * v * corners['ul'] + u * v * corners['ur']) # Shape: (n_channels, 3)

# Create the figure
fig = plt.figure(figsize=(20, 6))

# Plot the stimulus onset and offset dashed lines
plt.plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--', 
    linewidth=2, alpha=.25, label='_nolegend_')

# Plot the ERPs of the chosen subject
sub = 1
erp = erps[sub-1] # Shape: (n_channels, n_times)
for i in range(erp.shape[0]): 
    plt.plot(times, erp[i], color=colors[i], linewidth=1, alpha=0.75)

# x-axis parameters
plt.xlabel('Time (s)', fontsize=fontsize)
xticks = [0, .5, 1, 1.5, 2, 2.5, 3, 3.498]
xlabels = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel("Voltage (µV)", fontsize=fontsize)
ymin = np.nanmin(erp) - abs((np.nanmin(erp)) * .1)
ymax = np.nanmax(erp) + abs((np.nanmax(erp)) * .1)
plt.ylim(bottom=ymin, top=ymax)

# Save the figure
file_name = os.path.join(save_dir, 'eeg_erps.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the electrode position topography
# =============================================================================
# Channels -> 2D coords -> per-channel color
montage = mne.channels.make_standard_montage('standard_1005')
ch_pos = montage.get_positions()['ch_pos']
missing_fallback = {'O9': 'O1', 'O10': 'O2'} # absent from standard_1005; borrow neighbor for viz
coord_names = [c if c in ch_pos else missing_fallback[c] for c in ch_names]
info = mne.create_info(list(dict.fromkeys(coord_names)), 1000, 'eeg')
info.set_montage(montage)
pos2d = _find_topomap_coords(info, picks='eeg') # azimuthal-equidistant projection
name_to_xy = dict(zip(info.ch_names, pos2d))
xy = np.array([name_to_xy[c] for c in coord_names]) # Shape: (n_channels, 2)
xy_n = (xy - xy.min(0)) / (np.ptp(xy, 0) + 1e-12) # np.ptp: ndarray.ptp removed in numpy 2.0
corners = {'ll': np.array([0.85, 0.10, 0.10]),   # posterior-left  red
           'lr': np.array([0.95, 0.75, 0.10]),   # posterior-right yellow
           'ul': np.array([0.10, 0.25, 0.80]),   # anterior-left   blue
           'ur': np.array([0.10, 0.70, 0.35])}   # anterior-right  green
u, v = xy_n[:, 0][:, None], xy_n[:, 1][:, None]
colors = ((1 - u) * (1 - v) * corners['ll'] + u * (1 - v) * corners['lr']
          + (1 - u) * v * corners['ul'] + u * v * corners['ur']) # Shape: (n_channels, 3)

# Plot parameters
fontsize = 20
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rcParams['text.usetex'] = False
marker_size = 300
label_size = 4.5
show_labels = False
y_offset = 0.015

# Plot the electrode position topography
xy_plot = xy - np.array([0.0, y_offset]) # shift markers down relative to the fixed head outline
sphere = np.array([0., 0., 0., 0.095]) # MNE default head sphere (HEAD_SIZE_DEFAULT)
outlines = _make_head_outlines(sphere, xy, 'head', clip_origin=(0., 0.))
fig, ax = plt.subplots(figsize=(8, 8))
for key in ['head', 'nose', 'ear_left', 'ear_right']:
    ax.plot(*outlines[key], color='k', linewidth=1.5, zorder=1)
ax.scatter(xy_plot[:, 0], xy_plot[:, 1], c=colors, s=marker_size,
    edgecolors='k', linewidths=0.5, zorder=2)
if show_labels:
    for c, (x, y) in zip(ch_names, xy_plot): # true names, not fallback
        ax.annotate(c, (x, y), fontsize=label_size, ha='center', va='center', zorder=3)
ax.set_aspect('equal')
ax.axis('off')

# Save the figure
file_name = os.path.join(save_dir, 'topoplot_erp_colors.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the NCSNR
# =============================================================================
# EEG channel selection
idx_ch = []
for c, chan in enumerate(ch_names):
    if 'O' in chan or 'P' in chan:
        idx_ch.append(c)
idx_ch = np.array(idx_ch)

# Plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(args.subjects))

# Create the figure
fig = plt.figure(figsize=(20, 6))

# Plot the stimulus onset and offset dashed lines
plt.plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--', 
    linewidth=2, alpha=.25, label='_nolegend_')

# Plot the NCSNR of each subject
for s, sub in enumerate(args.subjects):
    nc = np.nanmean(ncsnr[s][idx_ch], 0)
    plt.plot(times, nc, color=colors[s], linewidth=2, alpha=1, label=sub)

# x-axis parameters
plt.xlabel('Time (s)', fontsize=fontsize)
xticks = [0, .5, 1, 1.5, 2, 2.5, 3, 3.498]
xlabels = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel("NCSNR", fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
ylabels = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=0, top=0.45)

# Legend
plt.legend(loc=0, ncol=6, fontsize=25, frameon=False, title='Participants:')

# Save the figure
file_name = os.path.join(save_dir, 'ncsnr.svg')
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
colors = sample_cmap(len(args.subjects))

# Create the figure
fig = plt.figure(figsize=(20, 6))

# Plot the stimulus onset/offset and decoding chance dashed lines
plt.plot([0, 0], [100, -100], 'k--', [3, 3], [100, -100], 'k--',
    [-10, 10], [50, 50], 'k--', linewidth=2, alpha=.25, label='_nolegend_')

# Plot the decoding results of each subject
for s, sub in enumerate(args.subjects):
    plt.plot(times, decoding[s], color=colors[s], linewidth=2, alpha=1,
        label=sub)

    # x-axis parameters
    plt.xlabel('Time (s)', fontsize=fontsize)
    xticks = [0, .5, 1, 1.5, 2, 2.5, 3, 3.498]
    xlabels = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
    plt.xticks(ticks=xticks, labels=xlabels)
    plt.xlim(left=min(times), right=max(times))

    # y-axis parameters
    plt.ylabel("Decoding accuracy (%)", fontsize=fontsize)
    yticks = [50, 60, 70, 80, 90, 100]
    ylabels = [50, 60, 70, 80, 90, 100]
    plt.yticks(ticks=yticks, labels=ylabels)
    plt.ylim(bottom=45, top=80)

# Legend
plt.legend(loc=0, ncol=6, fontsize=25, frameon=False, title='Participants:')

# Save the figure
file_name = os.path.join(save_dir, 'pairwise_decoding.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()