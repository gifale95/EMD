"""Plot encoding.

The feature maps come from a CLIP vision transformer, and are downsampled to 250
principal components using PCA.

https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_32.html

Parameters
----------
subject : int
    Number of the used THINGS EEG2 subject.
model : str
    Name of the used encoding model.
things_eeg_2_dir : str
    Directory of the THINGS EEG2 dataset.
    https://osf.io/3jk45/
berg_dir : str
    Directory of the Brain Encoding Response Generator (BERG).
    https://github.com/gifale95/BERG

"""

import argparse
import torch
import numpy as np
import os
import matplotlib
from matplotlib import pyplot as plt


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--subjects', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
parser.add_argument('--project_dir', default='/scratch/giffordale95/projects/encoding_things_eeg2_zitong', type=str)
args, unknown = parser.parse_known_args()



# =============================================================================
# Load the results
# =============================================================================
correlation_vitb32 = []
correlation_img2eeg = []
for sub in args.subjects:
    save_dir = os.path.join(args.project_dir, 'encoding_accuracy',
        f'encoding_accuracy_sub-{format(sub,"02")}.npy')
    acc = np.load(save_dir, allow_pickle=True).item()
    correlation_vitb32.append(acc['correlation_vitb32'])
    correlation_img2eeg.append(acc['correlation_img2eeg'])
    times = acc['times']
correlation_vitb32 = np.array(correlation_vitb32)
correlation_img2eeg = np.array(correlation_img2eeg)


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
# Plot
# =============================================================================
plt.figure(figsize=(10,5))
plt.plot(times, np.mean(correlation_vitb32, 0), label='ViT-B/32', color='tab:blue')
plt.plot(times, np.mean(correlation_img2eeg, 0), label='Image-to-EEG', color='tab:orange')
plt.xlabel('Time (ms)')
plt.ylabel('Correlation')
plt.legend()
plt.tight_layout()
save_dir = os.path.join(args.project_dir, 'figures')
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
plt.savefig(os.path.join(save_dir, 'encoding_accuracy.svg'), dpi=300)