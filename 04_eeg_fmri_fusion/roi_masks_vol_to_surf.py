import os
import numpy as np
from nilearn import datasets, surface, plotting
import pickle
from tqdm import tqdm
import nibabel as nib
import cortex
import cortex.polyutils
import matplotlib.pyplot as plt

# !!! Create masks with NSD-mapdata code and/or freesurfer, for better precision.

# Input parameters
dataset_dir = '/scratch/giffordale95/projects/eeg_moments/bold_moments_dataset'
subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
hemispheres = ['l', 'r']
rois = ['7AL', 'BA2', 'EBA', 'FFA', 'IPS0', 'IPS1-2-3', 'LOC', 'MT', 'OFA',
	'PFop', 'PFt', 'PPA', 'RSC', 'STS', 'TOS', 'V1d', 'V1v', 'V2d', 'V2v',
	'V3ab', 'V3d', 'V3v', 'hV4']

# Loop over subjects and hemispheres
for s in tqdm(subjects):
	for h in hemispheres:

		# ROI masks dictionary
		roi_masks = {}

		# Loop over ROIs
		for r in rois:
			file_dir = os.path.join(dataset_dir, 'derivatives', 'versionB',
				'MNI152', 'GLM', 'sub-'+format(s, '02'), 'ROIs',
				'ROI-'+h+r+'_indices.pkl')

			# Load the ROI masks in MNI space
			file = open(file_dir, 'rb')
			vol_all = pickle.load(file)
			vol_img = vol_all[2]

			# ROI voxel selection
			vol_data = vol_img.get_fdata()
			vol_data[:] = 0
			roi_voxels = vol_all[0]
			for v in range(len(roi_voxels)):
				vol_data[roi_voxels[v,0],roi_voxels[v,1],roi_voxels[v,2]] = 1
			vol_img_roi = nib.Nifti1Image(vol_data, vol_img.affine,
				vol_img.header)

			# Plot the ROI in volume space
#			plotting.plot_glass_brain(vol_img_roi)
#			plt.show()

			# Load the fsaverage template
			fsaverage = datasets.fetch_surf_fsaverage("fsaverage")

			# Transform the ROI masks into fsaverage space
			if h == 'l':
				surf = surface.vol_to_surf(
					vol_img_roi,
					surf_mesh=fsaverage["pial_left"],
					inner_mesh=fsaverage["white_left"],
					interpolation='linear'
					)
			elif h == 'r':
				surf = surface.vol_to_surf(
					vol_img_roi,
					surf_mesh=fsaverage["pial_right"],
					inner_mesh=fsaverage["white_right"],
					interpolation='linear'
				)

			# Set the mask to binary values (the transformation creates floats, regardless
			# of the interpolation method used)
			threshold = 0.2
			surf[surf>=threshold] = 1
			surf[surf<threshold] = 0

			# Store the ROI masks
			roi_masks[r] = surf
			del surf

		# Save the ROI masks of each subject and hemisphere
		if h == 'l':
			hemi = 'left'
		elif h == 'r':
			hemi = 'right'
		save_dir = os.path.join(dataset_dir, 'derivatives', 'versionB',
			'fsaverage', 'GLM', 'sub-'+format(s, '02'), 'prepared_betas',
			'sub-'+format(s, '02')+'_roi_masks_hemi-'+hemi+'.npy')
		np.save(save_dir, roi_masks)
		del roi_masks

# Plot
# Create the vertex data
# subject = 'fsaverage'
# data = np.append(lh_surf, rh_surf)
# vertex_data = cortex.Vertex(data, subject, cmap='hot', vmin=0, vmax=1,
# 	with_colorbar=True)
# # Plot the figure
# fig = cortex.quickshow(vertex_data,
# #	height=500, # Increase resolution of map and ROI contours
# 	with_curvature=True,
# 	curvature_brightness=0.5,
# 	with_rois=False,
# 	with_labels=False,
# 	linewidth=5,
# 	linecolor=(1, 1, 1),
# 	with_colorbar=True
# 	)
# plt.show()
