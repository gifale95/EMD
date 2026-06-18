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
results_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion_all_sub',
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
save_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion_all_sub',
    'plots', 'encoding_accuracy_surfaceplots')
os.makedirs(save_dir, exist_ok=True)


# =============================================================================
# Plot the encoding accuracy of the EEG-fMRI fusion encoding models on brain
# surfaces (subject-average) # !!!
# =============================================================================
# Plot parameters
fontsize = 40
matplotlib.rcParams['font.sans-serif'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = fontsize
plt.rc('xtick', labelsize=19)
plt.rc('ytick', labelsize=19)
matplotlib.use("svg")
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none'
subject = 'fsaverage_nsd_sub-01'

# Loop over EEG time points
for t, time in enumerate(tqdm(times)):

    # Average the results across subjects, and append them across left and
    # right hemishperes
    data = np.append(np.nanmean(corr_tfmri_fmri[:,0,:,t], 0),
        np.nanmean(corr_tfmri_fmri[:,1,:,t], 0))

    # Create the flat brain surface
    vertex_data = cortex.Vertex(
        data,
        subject,
        cmap='afmhot',
        vmin=0,
        vmax=1,
        with_colorbar=True)

    # Plot the flat brain surface
    fig = cortex.quickshow(
        vertex_data,
        #height=2000, # Increase resolution of map and ROI contours
        with_curvature=True,
        with_rois=True,
        roi_list=['Early', 'Intermediate', 'Ventral', 'Lateral', 'Dorsal'],
        linewidth=3,
        linecolor=(1, 1, 1),
        with_labels=True,
        labelsize=25,
        curvature_brightness=0.4,
        with_colorbar=True
        )

    # Add title
    title = f'Time (ms): {np.round(time*1000)}'
    plt.title(title, fontsize=fontsize)

    # Save the plot
    plot_file = os.path.join(save_dir, f'correlation_time-{t:03d}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', format='png')
    plt.close()


# =============================================================================
# Plot the ROI-wise correlations between t-fMRI and fMRI responses
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

# Define the ROIs to plot]
rois = ['7AL', 'BA2', 'EBA', 'FFA', 'IPS0', 'IPS1-2-3', 'LOC', 'MT', 'OFA',
    'PFop', 'PFt', 'PPA', 'RSC', 'STS', 'TOS', 'V1', 'V2', 'V3', 'V3ab','hV4']
rois = ['V1', 'V2', 'V3', 'hV4', 'LOC', 'FFA', 'EBA', 'PPA']
rois = ['V1', 'V2', 'V3', 'hV4']

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
    # peak = times[np.argmax(np.mean(roi_correlation[roi], 0))]
    # max_corr = max(np.mean(roi_correlation[roi], 0))
    # plt.scatter(peak, max_corr, color=colors[r], s=200, marker='o',
    #     edgecolors='k', linewidths=1, zorder=3, label='_nolegend_')

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
plt.ylim(bottom=-.05, top=.35)

# Legend
plt.legend(fontsize=fontsize, loc=0, ncols=4, frameon=False)

# Save the figure
save_dir = os.path.join(args.emd_dir, 'results', 'eeg_fmri_encoding_fusion_all_sub',
    'plots')
file_name = os.path.join(save_dir, 'roi_correlation.svg')
fig.savefig(file_name, bbox_inches='tight', transparent=True, format='svg')
plt.close()