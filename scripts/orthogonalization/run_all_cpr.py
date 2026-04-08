import os
import slicer
import vtk
import numpy as np
import sys
import glob

# --- CONFIG ---
# Add script folder to path to import utility_slicer
my_script_folder = "/home/group1/Desktop/Challange1/orthogonalization/aorta_both_iliacs"

if my_script_folder not in sys.path:
    sys.path.append(my_script_folder)

from utility_slicer import SegmentEditorManager, CPRProcessor

base_path = "/home/group1/Desktop/Challange1/results"
norm_scans_dir = os.path.join(base_path, "normalized_scans_V")
all_segs_dir = os.path.join(base_path, "total_segmentator_type_V")
results_base_dir = os.path.join(base_path, "Result_CPR")

# --- HELPER FUNCTIONS ---

def process_patient(case_id, volume_path, seg_folder_path):
    print(f"\n==================================================")
    print(f"PROCESSING PATIENT: {case_id}")
    print(f"==================================================")

    try:
        # Clear Scene before starting new patient to avoid memory issues
        slicer.mrmlScene.Clear(0)
        
        # Load Volume
        print(f"Caricamento volume: {volume_path}")
        volume_node = slicer.util.loadVolume(volume_path)

        # Load Aorta
        print("Caricamento Aorta...")
        seg_aorta_path = os.path.join(seg_folder_path, "aorta.nii.gz")
        
        if not os.path.exists(seg_aorta_path):
            print(f"SKIP {case_id}: Aorta non trovata in {seg_aorta_path}")
            return

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
            seg_path = os.path.join(seg_folder_path, side['filename'])
            
            if not os.path.exists(seg_path):
                print(f"Iliaca {side['name']} non trovata. Skip.")
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

        # --- 2. Process Each Side ---
        for side in sides_config:
            side_name = side['name']
            iliac_seg_name = side['segment_name']
            
            # Check if segment exists (loaded successfully)
            if not segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(iliac_seg_name):
                # print(f"Skipping {side_name}: Segment {iliac_seg_name} not loaded.")
                continue

            print(f"--- Elaborazione Lato: {side_name} ({case_id}) ---")
            combined_seg_name = f"Full_Path_{side_name}"

            try:
                # Create combined vessel
                manager.copy_segment(combined_seg_name, "aorta") 
                manager.add_segment(combined_seg_name, iliac_seg_name) 
                manager.smoothing(combined_seg_name, kernel_size=1)
                
                # Calculate endpoints for centerline
                seg_id = manager.get_segment_id(combined_seg_name)
                bounds = manager.get_bounds_of_segment(seg_id)
                # Slightly narrow search to ensure point is inside vessel
                z_top = bounds['z_max'] - 2
                z_bottom = bounds['z_min'] + 2

                start_point = manager.get_segment_centroid_ras(combined_seg_name, slice_height=z_top)
                end_point = manager.get_segment_centroid_ras(combined_seg_name, slice_height=z_bottom)

                # VMTK Centerline extraction
                centerline_name = f"Aorta_Centerline_{side_name}"
                centerline_node = manager.extract_centerline(
                    segment_id=seg_id,
                    case_id=case_id, 
                    setting="norm",
                    centerline_endpoints=[start_point, end_point],
                    name=centerline_name
                )

                # Run CPR
                straightened_vol, transform, panoramic = cpr.do_cpr(
                    centerline_node=centerline_node,
                    volume_node=volume_node,
                    volume_name=f"Normalized_CPR_{side_name}",
                    output_spacing=[0.5, 0.5, 0.5] 
                )

                # Save results
                save_dir = os.path.join(results_base_dir, case_id)
                os.makedirs(save_dir, exist_ok=True)
                
                vol_filename = f"straightened_{side_name.lower()}.nii.gz"
                cl_filename = f"centerline_{side_name.lower()}.json"
                
                slicer.util.saveNode(straightened_vol, os.path.join(save_dir, vol_filename))
                slicer.util.saveNode(centerline_node, os.path.join(save_dir, cl_filename))
                
                print(f"SUCCESS {case_id} {side_name}")

            except Exception as e:
                print(f"ERRORE {case_id} {side_name}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
    except Exception as e:
         print(f"CRITICAL FAIL on patient {case_id}: {e}")

# --- MAIN LOOP ---

def main():
    # Find all normalized files: pzXXX_normalized.nii.gz
    # Looking for pattern: "pz*_normalized.nii.gz"
    search_pattern = os.path.join(norm_scans_dir, "pz*_normalized.nii.gz")
    norm_files = glob.glob(search_pattern)
    
    print(f"Found {len(norm_files)} normalized scans to check.")

    for vol_path in norm_files:
        filename = os.path.basename(vol_path)
        # Extract case_id (e.g., "pz007") from "pz007_normalized.nii.gz"
        # Split by "_", take first part, strip 'pz' if you want pure number, or keep '007'
        # Based on your previous code: case_id = "007" leads to folder pz007_normalized.
        # Let's extract the number part.
        
        # Example filename: pz016_normalized.nii.gz
        parts = filename.split('_') # ['pz016', 'normalized.nii.gz']
        patient_code = parts[0] # 'pz016'
        
        # If your folders are named like "patient_segmentations_007_0CT_po1_V", we need just "007"
        case_id_number = patient_code.replace("pz", "") 
        
        # Construct expected segmentation folder path
        # Pattern: patient_segmentations_{NUM}_0CT_po1_V
        seg_folder_name = f"patient_segmentations_{case_id_number}_0CT_po1_V"
        seg_folder_path = os.path.join(all_segs_dir, seg_folder_name)
        
        if os.path.exists(seg_folder_path):
            process_patient(case_id_number, vol_path, seg_folder_path)
        else:
            print(f"Skipping {patient_code}: Segmentation folder not found ({seg_folder_name})")

    print("\nBatch processing completed.")
    
    # Close Slicer if running headless
    if slicer.app.commandOptions().noMainWindow:
        sys.exit(0)

if __name__ == "__main__":
    main()