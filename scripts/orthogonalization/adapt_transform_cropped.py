import os
import slicer
import numpy as np

# --- Configuration ---
case_id = "001"
base_path = "/home/group1/Desktop/Challange1/results"
side = "left" # or "right"

# Paths
path_volume = os.path.join(base_path, f"all_masked_normalized_scans/normalized_scans_V/pz{case_id}_normalized.nii.gz")
path_aorta = os.path.join(base_path, f"total_segmentator_type_V/patient_segmentations_{case_id}_0CT_po1_V/aorta.nii.gz")
path_iliac = os.path.join(base_path, f"total_segmentator_type_V/patient_segmentations_{case_id}_0CT_po1_V/iliac_artery_{side}.nii.gz")

path_transform = os.path.join(base_path, "Result_CPR", case_id, f"transform_{side}.h5")
path_output_volume = os.path.join(base_path, "Result_CPR", case_id, f"pz{case_id}_full_ct_straightened_{side}.nii.gz")

# --- 1. Load Data ---
print("Loading original CT...")
volume_node = slicer.util.loadVolume(path_volume)

print("Loading segmentations to calculate Region of Interest (ROI)...")
aorta_node = slicer.util.loadSegmentation(path_aorta)
iliac_node = slicer.util.loadSegmentation(path_iliac)

# --- 2. Calculate Bounding Box bounds ---
bounds_aorta = np.zeros(6) # [xmin, xmax, ymin, ymax, zmin, zmax]
bounds_iliac = np.zeros(6)

aorta_node.GetBounds(bounds_aorta)
iliac_node.GetBounds(bounds_iliac)

# Find the maximum extents that cover both the aorta and the specific iliac artery
min_bounds = np.minimum(bounds_aorta[0::2], bounds_iliac[0::2])
max_bounds = np.maximum(bounds_aorta[1::2], bounds_iliac[1::2])

# Drastically reduce padding. CPR transform is only valid within ~27.5mm of the centerline.
# Large padding grabs tissue outside the grid domain, causing wavy extrapolation artifacts.
padding_mm = 5.0
min_bounds -= padding_mm
max_bounds += padding_mm

# --- 3. Create ROI and Crop ---
print("Creating ROI and cropping CT...")
roi_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsROINode", "Crop_ROI")
center = (min_bounds + max_bounds) / 2.0
size = max_bounds - min_bounds

roi_node.SetCenter(center)
roi_node.SetSize(size)

# Set up the Crop Volume Logic
crop_logic = slicer.modules.cropvolume.logic()
crop_params = slicer.vtkMRMLCropVolumeParametersNode()
slicer.mrmlScene.AddNode(crop_params)

cropped_volume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", f"Cropped_CT_{side}")

crop_params.SetInputVolumeNodeID(volume_node.GetID())
crop_params.SetROINodeID(roi_node.GetID())
crop_params.SetOutputVolumeNodeID(cropped_volume.GetID())
crop_params.SetVoxelBased(True) # Snaps to original CT resolution

# Run crop
crop_logic.Apply(crop_params)

# Clean up parameter node and raw uncropped volume (free memory)
slicer.mrmlScene.RemoveNode(crop_params)
slicer.mrmlScene.RemoveNode(volume_node)
slicer.mrmlScene.RemoveNode(aorta_node)
slicer.mrmlScene.RemoveNode(iliac_node)

# --- 4. Load & Apply Transform to Cropped Volume ---
print("Loading the non-linear transform (.h5)...")
transform_node = slicer.util.loadTransform(path_transform)

print("Applying transform to the cropped volume...")
cropped_volume.SetAndObserveTransformNodeID(transform_node.GetID())

print("Hardening transform (Resampling the cropped image into orthogonal space)...")
slicer.vtkSlicerTransformLogic().hardenTransform(cropped_volume)

# --- 5. Save ---
print(f"Saving straightened cropped CT to: {path_output_volume}")
slicer.util.saveNode(cropped_volume, path_output_volume)
print("Done!")