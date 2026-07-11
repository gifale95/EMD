"""Plot the EEG-fMRI encoding fusion results.

Parameters
----------
fmri_hemis : list
    List of fMRI hemispheres.
emd_dir : str
    Directory of the EEG Moments Dataset (EMD).

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from nilearn import plotting, datasets
import cortex
import matplotlib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--fmri_hemis', default=['left', 'right'], type=list)
parser.add_argument('--emd_dir', default='/scratch/giffordale95/projects/eeg_moments_dataset', type=str)
args, unknown = parser.parse_known_args()

print('>>> Plot <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))


# =============================================================================
# Load the results
# =============================================================================
# Load the results
results_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion',
    'stats', 'stats.npy')
stats = np.load(results_dir, allow_pickle=True).item()
lh_correlation = stats['lh_correlation']
rh_correlation = stats['rh_correlation']
roi_correlation = stats['roi_correlation']
ci_roi_correlation = stats['ci_roi_correlation']

# Load the EEG time points
metadata_dir = os.path.join(args.emd_dir, 'derivatives', 'eeg', 'sub-01',
    'sub-01_eeg_metadata.npy')
times = np.load(metadata_dir, allow_pickle=True).item()['times']


# =============================================================================
# Create the plots save directory
# =============================================================================
save_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion',
    'plots', 'encoding_accuracy_surfaceplots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot the encoding accuracy of the EEG-fMRI fusion encoding models on brain
# surfaces (selected time points)
# =============================================================================
plot_times = [0, 0.024, 0.05, 0.074, 0.1, 0.124, 0.15, 0.174, 0.2, 0.224, 0.25,
    0.274, 0.3, 0.324, 0.35, 0.374, 0.4, 0.424, 0.45, 0.474, 0.5, 0.524, 0.55,
    0.574, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
plot_views = ['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior']

fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage')  # 163842 verts/hemi

# Loop across plotting time points
for p_times in tqdm(plot_times):

    # Select the data time point to plot
    idx_time = np.where(times == p_times)[0][0]
    data_lh = lh_correlation[:,idx_time]
    data_rh = rh_correlation[:,idx_time]

    # Loop across plotting views
    for p_views in plot_views:

        # Plot the left hemisphere
        title = f'LH, Time (s): {p_times}, View: {p_views}'
        output_file = os.path.join(save_dir,
            f'correlation_lh_time-{p_times}_view-{p_views}.png')
        plotting.plot_surf_stat_map(
            fsaverage['infl_left'],
            stat_map=data_lh,
            bg_map=fsaverage['sulc_left'],
            hemi='left',
            view=p_views,
            vmin=0,
            vmax=0.5,
            cmap='afmhot',
            colorbar=False,
            title=title,
            output_file=output_file
        )

        # Plot the right hemisphere
        title = f'RH, Time (s): {p_times}, View: {p_views}'
        output_file = os.path.join(save_dir,
            f'correlation_rh_time-{p_times}_view-{p_views}.png')
        plotting.plot_surf_stat_map(
            fsaverage['infl_right'],
            stat_map=data_rh,
            bg_map=fsaverage['sulc_right'],
            hemi='right',
            view=p_views,
            vmin=0,
            vmax=0.5,
            cmap='afmhot',
            colorbar=False,
            title=title,
            output_file=output_file
        )


# =============================================================================
# Plot the encoding accuracy of the EEG-fMRI fusion encoding models on brain
# surfaces (all time points)
# =============================================================================
# # Plot parameters
# fontsize = 40
# matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
# matplotlib.rcParams['font.size'] = fontsize
# plt.rc('xtick', labelsize=19)
# plt.rc('ytick', labelsize=19)
# matplotlib.use("svg")
# plt.rcParams["text.usetex"] = False
# plt.rcParams['svg.fonttype'] = 'none'
# subject = 'fsaverage_nsd_sub-01'

# # Loop over EEG time points
# for t, time in enumerate(tqdm(times)):

#     # Average the results across subjects, and append them across left and
#     # right hemishperes
#     data = np.append(lh_correlation[:,t], rh_correlation[:,t])

#     # Create the flat brain surface
#     vertex_data = cortex.Vertex(
#         data,
#         subject,
#         cmap='afmhot',
#         vmin=0,
#         vmax=0.5,
#         with_colorbar=True)

#     # Plot the flat brain surface
#     fig = cortex.quickshow(
#         vertex_data,
#         #height=2000, # Increase resolution of map and ROI contours
#         with_curvature=True,
#         with_rois=True,
#         roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
#         linewidth=3,
#         linecolor=(1, 1, 1),
#         with_labels=False,
#         labelsize=25,
#         curvature_brightness=0.4,
#         with_colorbar=True
#         )

#     # Add title
#     title = f'Time (s): {time}'
#     plt.title(title, fontsize=fontsize)

#     # Save the plot
#     plot_file = os.path.join(save_dir, f'correlation_time-{t:04d}.png')
#     plt.savefig(plot_file, dpi=300, bbox_inches='tight', format='png')
#     plt.close()


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
# Plot the ROI-wise correlations between t-fMRI and fMRI responses
# (whole epoch)
# =============================================================================
# Define the ROIs to plot]
rois = ['V1', 'V2', 'V3', 'hV4', 'PPA', 'EBA', 'FFA', 'MT', 'LOC']

# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(rois))

# Create the figure
fig = plt.figure(figsize=(10, 7.5))

# Plot the stimulus onset and chance dashed line
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    [3, 3], [100, -100], 'k--', linewidth=2, alpha=.25, label='_nolegend_')

# Loop across ROIs
for r, roi in enumerate(rois):

    # Plot the correlation
    plt.plot(times, np.mean(roi_correlation[roi], 0),
        color=colors[r], linewidth=2, label=roi)

    # Plot the CIs
    plt.fill_between(times, ci_roi_correlation[roi][1],
        ci_roi_correlation[roi][0], color=colors[r], alpha=.1)

    # Plot the peak time point
    peak = times[np.argmax(np.mean(roi_correlation[roi], 0))]
    max_corr = max(np.mean(roi_correlation[roi], 0))
    plt.scatter(peak, max_corr, color=colors[r], s=200, marker='o',
        edgecolors='k', linewidths=1, zorder=3, label='_nolegend_')

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .5, 1, 1.5, 2, 2.5, 3, 3.498]
xlabels = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=max(times))

# y-axis parameters
plt.ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3, 0.4]
ylabels = [0, 0.1, 0.2, 0.3, 0.4]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.05, top=.41)

# Legend
plt.legend(fontsize=fontsize, loc=0, ncols=4, frameon=False)

# Save the figure
save_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion',
    'plots')
file_name = os.path.join(save_dir, 'roi_correlation_whole_epoch.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()


# =============================================================================
# Plot the ROI-wise correlations between t-fMRI and fMRI responses
# (first 500 epoch ms)
# =============================================================================
# Define the ROIs to plot
rois = ['V1', 'V2', 'V3', 'hV4', 'PPA', 'EBA', 'FFA', 'MT', 'LOC']

# Get the plot colors
def sample_cmap(N):
    cmap = plt.cm.get_cmap('inferno')
    values = np.linspace(0, 1, N+2)
    colors = cmap(values)[1:-1]
    return colors
colors = sample_cmap(len(rois))

# Create the figure
fig = plt.figure(figsize=(10, 7.5))

# Plot the stimulus onset and chance dashed line
plt.plot([-10, 10], [0, 0], 'k--', [0, 0], [100, -100], 'k--',
    [3, 3], [100, -100], 'k--', linewidth=2, alpha=.25, label='_nolegend_')

# Loop across ROIs
for r, roi in enumerate(rois):

    # Plot the correlation
    plt.plot(times, np.mean(roi_correlation[roi], 0),
        color=colors[r], linewidth=2, label=roi)

    # Plot the CIs
    plt.fill_between(times, ci_roi_correlation[roi][1],
        ci_roi_correlation[roi][0], color=colors[r], alpha=.1)

    # Plot the peak time point
    peak = times[np.argmax(np.mean(roi_correlation[roi], 0))]
    max_corr = max(np.mean(roi_correlation[roi], 0))
    plt.scatter(peak, max_corr, color=colors[r], s=200, marker='o',
        edgecolors='k', linewidths=1, zorder=3, label='_nolegend_')

# x-axis parameters
plt.xlabel('Time (ms)', fontsize=fontsize)
xticks = [0, .1, .2, .3, .4, .5]
xlabels = [0, .1, .2, .3, .4, .5]
plt.xticks(ticks=xticks, labels=xlabels)
plt.xlim(left=min(times), right=0.5)

# y-axis parameters
plt.ylabel("Pearson's $r$", fontsize=fontsize)
yticks = [0, 0.1, 0.2, 0.3, 0.4]
ylabels = [0, 0.1, 0.2, 0.3, 0.4]
plt.yticks(ticks=yticks, labels=ylabels)
plt.ylim(bottom=-.05, top=.41)

# Legend
plt.legend(fontsize=fontsize, loc=0, ncols=4, frameon=False)

# Save the figure
save_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion',
    'plots')
file_name = os.path.join(save_dir, 'roi_correlation_first_500ms.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()