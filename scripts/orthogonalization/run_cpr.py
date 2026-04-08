import os
import slicer
import vtk
import numpy as np
import sys

# Add script folder to path to import utility_slicer
my_script_folder = "/home/group1/Desktop/Challange1/orthogonalization/aorta_both_iliacs"

if my_script_folder not in sys.path:
    sys.path.append(my_script_folder)

from utility_slicer import SegmentEditorManager, CPRProcessor

# --- CONFIG ---
case_id = "001" 
base_path = "/home/group1/Desktop/Challange1/results" 

# Paths
path_volume = os.path.join(base_path, f"all_masked_normalized_scans/normalized_scans_V/pz{case_id}_normalized.nii.gz")
path_seg_folder = os.path.join(base_path, f"total_segmentator_type_V/patient_segmentations_{case_id}_0CT_po1_V")

# Load Volume
print(f"Caricamento volume: {path_volume}")
volume_node = slicer.util.loadVolume(path_volume)

# Load Aorta
print("Caricamento Aorta...")
seg_aorta_path = os.path.join(path_seg_folder, "aorta.nii.gz")
segmentation_node = slicer.util.loadSegmentation(seg_aorta_path) 
segmentation_node.SetName("Segmentation_Main")

# Rename internal segment (usually "Segment_1") to "aorta" for consistency
seg_logic = segmentation_node.GetSegmentation()
if seg_logic.GetNumberOfSegments() > 0:
    first_id = seg_logic.GetNthSegmentID(0)
    seg_logic.GetSegment(first_id).SetName("aorta")

# Define configurations for both sides
sides_config = [
    {
        "name": "Right",
        "filename": "iliac_artery_right.nii.gz",
        "segment_name": "iliac_artery_right"
    },
    {
        "name": "Left",
        "filename": "iliac_artery_left.nii.gz",
        "segment_name": "iliac_artery_left"
    }
]

# --- 1. Load and Merge both Iliac Arteries into the main segmentation node ---
for side in sides_config:
    print(f"Caricamento Iliaca {side['name']}...")
    seg_path = os.path.join(path_seg_folder, side['filename'])
    
    if not os.path.exists(seg_path):
        print(f"ATTENZIONE: File {seg_path} non trovato. Salto questo lato.")
        continue

    temp_iliac_node = slicer.util.loadSegmentation(seg_path)
    seg_logic_temp = temp_iliac_node.GetSegmentation()

    if seg_logic_temp.GetNumberOfSegments() > 0:
        first_id_temp = seg_logic_temp.GetNthSegmentID(0)
        seg_logic_temp.GetSegment(first_id_temp).SetName(side['segment_name'])
        
        # Copy to main node
        seg_logic.CopySegmentFromSegmentation(
            seg_logic_temp,
            first_id_temp
        )

    slicer.mrmlScene.RemoveNode(temp_iliac_node)

# Init manager
manager = SegmentEditorManager(segmentation_node, volume_node)
cpr = CPRProcessor()

# --- 2. Process Both Sides ---
for side in sides_config:
    side_name = side['name']
    iliac_seg_name = side['segment_name']
    
    # Check if segment exists (loaded successfully)
    if not segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(iliac_seg_name):
        print(f"Skipping {side_name}: Segment {iliac_seg_name} not found.")
        continue

    print(f"\n--- Elaborazione Lato: {side_name} (Aorta + {iliac_seg_name}) ---")
    combined_seg_name = f"Full_Path_{side_name}"

    try:
        # Create combined vessel
        print(f"Merging segmenti in {combined_seg_name}...")
        manager.copy_segment(combined_seg_name, "aorta") 
        manager.add_segment(combined_seg_name, iliac_seg_name) 
        manager.smoothing(combined_seg_name, kernel_size=1)
        
        # Calculate endpoints to guide centerline extraction
        seg_id = manager.get_segment_id(combined_seg_name)
        bounds = manager.get_bounds_of_segment(seg_id)
        z_top = bounds['z_max'] - 2
        z_bottom = bounds['z_min'] + 2

        start_point = manager.get_segment_centroid_ras(combined_seg_name, slice_height=z_top)
        end_point = manager.get_segment_centroid_ras(combined_seg_name, slice_height=z_bottom)

        # VMTK Centerline extraction
        print("Estrazione Centerline...")
        centerline_name = f"Aorta_Centerline_{side_name}"
        centerline_node = manager.extract_centerline(
            segment_id=seg_id,
            case_id=case_id, 
            setting="norm",
            centerline_endpoints=[start_point, end_point],
            name=centerline_name
        )

        # Run CPR
        print("Esecuzione CPR...")
        straightened_vol, transform, panoramic = cpr.do_cpr(
            centerline_node=centerline_node,
            volume_node=volume_node,
            volume_name=f"Normalized_CPR_{side_name}",
            output_spacing=[0.5, 0.5, 0.5] 
        )

        # Save results in patient specific folder
        save_dir = os.path.join(base_path, "Result_CPR", case_id)
        os.makedirs(save_dir, exist_ok=True)
        
        # Unique filenames for Right and Left
        vol_filename = f"straightened_{side_name.lower()}.nii.gz"
        cl_filename = f"centerline_{side_name.lower()}.json"
        
        slicer.util.saveNode(straightened_vol, os.path.join(save_dir, vol_filename))
        slicer.util.saveNode(centerline_node, os.path.join(save_dir, cl_filename))
        
        print(f"SUCCESS {side_name}: Salvato in {os.path.join(save_dir, vol_filename)}")

    except Exception as e:
        print(f"ERRORE durante elaborazione {side_name}: {e}")
        # Continue to the next side even if this one fails
        continue

# Close Slicer if running headless
if slicer.app.commandOptions().noMainWindow:
    sys.exit(0)