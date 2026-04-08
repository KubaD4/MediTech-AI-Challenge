import os
import slicer

# --- Configuration ---
case_id = "001"
base_path = "/home/group1/Desktop/Challange1/results"
side = "left" # or "right"

# Paths to the specific files
path_volume = os.path.join(base_path, f"all_masked_normalized_scans/normalized_scans_V/pz{case_id}_normalized.nii.gz")
path_transform = os.path.join(base_path, "Result_CPR", case_id, f"transform_{side}.h5")
path_output_volume = os.path.join(base_path, "Result_CPR", case_id, f"pz{case_id}_full_ct_straightened_{side}.nii.gz")

# --- 1. Load the Data ---
print("Loading original CT...")
volume_node = slicer.util.loadVolume(path_volume)

print("Loading the non-linear transform (.h5)...")
transform_node = slicer.util.loadTransform(path_transform)

# --- 2. Apply Transform ---
print("Applying transform to the volume...")
# Set the transform node as the parent of the volume node
volume_node.SetAndObserveTransformNodeID(transform_node.GetID())

# --- 3. Harden Transform (Resample the Image) ---
# Because this is a non-linear transform, hardening will actually create an explicit 
# resampling grid for the CT volume, altering the underlying voxel data to match the grid.
print("Hardening transform (this may take a moment for large CTs...)")
slicer.vtkSlicerTransformLogic().hardenTransform(volume_node)

# --- 4. Save Resampled Volume ---
print(f"Saving fully straightened CT to: {path_output_volume}")
slicer.util.saveNode(volume_node, path_output_volume)
print("Done!")