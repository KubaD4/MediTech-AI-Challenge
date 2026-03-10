import vtk
import numpy as np
import logging
import functools
import os
import sys
import slicer

from CurvedPlanarReformat import CurvedPlanarReformatLogic
import SegmentStatistics 

from slicer.util import getNode, arrayFromVolume, arrayFromSegmentBinaryLabelmap, updateSegmentBinaryLabelmapFromArray


import csv



def log_method_call(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        logging.debug(f"→ Calling `{method.__name__}` with args={args}, kwargs={kwargs}")
        result = method(self, *args, **kwargs)
        logging.debug(f"← Finished `{method.__name__}`")
        return result
    return wrapper

class CoordinateTransformer:
    # x,y,z -> R A S (mm) 
    # R A S -> x y z
   
    def __init__(self, volume_node):
        self.volume_node = volume_node
        self.logger = logging.getLogger(self.__class__.__name__)

    @log_method_call
    def ijk_to_ras(self, point_kji):
        point_ijk = [point_kji[2], point_kji[1], point_kji[0]]
        #point_ijk = point_kji
        matrix = vtk.vtkMatrix4x4()
        self.volume_node.GetIJKToRASMatrix(matrix)
        point_ras = [0, 0, 0, 1]
        matrix.MultiplyPoint(np.append(point_ijk, 1.0), point_ras)

        self.logger.debug(f"Called IJKtoRAS for point {point_ijk}")
        self.logger.debug(f"IJK to RAS matrix:\n{matrix}")
        self.logger.debug(f"Result: {point_ijk} → {point_ras[:-1]}")
        return point_ras[:-1]
    
    @log_method_call
    def ras_to_ijk(self, point_ras):
        matrix = vtk.vtkMatrix4x4()
        self.volume_node.GetIJKToRASMatrix(matrix)
        matrix.Invert()
        point_ijk = [0, 0, 0, 1]
        matrix.MultiplyPoint(np.append(point_ras, 1.0), point_ijk)

        self.logger.debug(f"Called RAStoIJK for point {point_ras}")
        self.logger.debug(f"RAS to IJK matrix (inverted):\n{matrix}")
        self.logger.debug(f"Result: {point_ras} → {point_ijk[:-1]}")
        return [round(x) for x in point_ijk[:-1]]
    
    
class CPRProcessor:
    # curved planar reformat
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logic = CurvedPlanarReformatLogic()

    def do_cpr(
        self,
        centerline_node,
        volume_node,
        volume_name,
        field_of_view=[55.0, 55.0],
        output_spacing=[1.0, 1.0, 0.5],
    ):
        self.logger.debug(
            f"Calculating CPR for centerline {centerline_node.GetName()} and volume {volume_node.GetName()}"
        )

        straightening_transform = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLTransformNode", f"Straightening_transform_{volume_name}"
        )

        self.logic.computeStraighteningTransform(
            straightening_transform, centerline_node, field_of_view, output_spacing[2]
        )

        straightened_volume = slicer.modules.volumes.logic().CloneVolume(
            volume_node, f"{volume_node.GetName()}_{volume_name}_straightened"
        )
        self.logic.straightenVolume(
            straightened_volume, volume_node, output_spacing, straightening_transform
        )

        panoramic_volume = slicer.modules.volumes.logic().CloneVolume(
            straightened_volume, f"{straightened_volume.GetName()}_{volume_name}_panoramic"
        )
        self.logic.projectVolume(panoramic_volume, straightened_volume)

        slicer.util.setSliceViewerLayers(
            background=straightened_volume, fit=True, rotateToVolumePlane=True
        )

        self.logger.debug(f"Straightened volume: {straightened_volume.GetName()}")
        self.logger.debug(f"Transform: {straightening_transform.GetName()}")

        return straightened_volume, straightening_transform, panoramic_volume

    def transfer_segmentation_world_to_cpr(self, segment_names, from_seg_node, to_seg_node, volume_node):
        self.logger.debug(f"Transferring segments {segment_names} to CPR space")
        for name in segment_names:
            seg_array = arrayFromSegmentBinaryLabelmap(from_seg_node, name, volume_node)
            updateSegmentBinaryLabelmapFromArray(seg_array, to_seg_node, name, volume_node)

    def transfer_segmentation_cpr_to_world(self, segment_names, from_seg_node, to_seg_node):
        self.logger.debug(f"Transferring segments {segment_names} back to world space")
        for name in segment_names:
            seg_id = from_seg_node.GetSegmentation().GetSegmentIdBySegmentName(name)
            to_seg_node.GetSegmentation().CopySegmentFromSegmentation(
                from_seg_node.GetSegmentation(), seg_id
            )

    def invert_transform(self, transform_node):
        self.logger.debug(f"Inverting transform {transform_node.GetName()}")
        shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
        itemID = shNode.GetItemByDataNode(transform_node)
        clonedItemID = slicer.modules.subjecthierarchy.logic().CloneSubjectHierarchyItem(shNode, itemID)
        cloned_node = shNode.GetItemDataNode(clonedItemID)
        cloned_node.Inverse()
        self.logger.debug(f"Inverted transform created: {cloned_node.GetName()}")
        return cloned_node

    def from_cpr_to_world(
        self,
        cpr_seg_node,
        main_seg_node,
        volume_node,
        inverse_transform,
        save_path,
        segments_to_export=None,
    ):
        if segments_to_export:
            self.make_only_segments_visible(segments_to_export, cpr_seg_node)

        visible_segments = []
        display_node = cpr_seg_node.GetDisplayNode()
        for seg_id in cpr_seg_node.GetSegmentation().GetSegmentIDs():
            if display_node.GetSegmentVisibility(seg_id):
                visible_segments.append(seg_id)

        cpr_seg_node.SetAndObserveTransformNodeID(inverse_transform.GetID())

        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode", cpr_seg_node.GetName()
        )

        
        slicer.modules.segmentations.logic().ExportVisibleSegmentsToLabelmapNode(
            cpr_seg_node, labelmap_node
        )
        slicer.util.exportNode(
            labelmap_node,
            os.path.join(save_path, f"{labelmap_node.GetName()}.nii.gz"),
            {"useCompression": 0},
            world=True,
        )

     

        loaded_seg = slicer.util.loadSegmentation(
            os.path.join(save_path, f"{labelmap_node.GetName()}.nii.gz")
        )

        for i, seg_id in enumerate(loaded_seg.GetSegmentation().GetSegmentIDs()):
            loaded_seg.GetSegmentation().GetSegment(seg_id).SetName(visible_segments[i])

        self.transfer_segmentation_cpr_to_world(
            visible_segments,
            loaded_seg,
            main_seg_node,
        )

        cpr_seg_node.SetAndObserveTransformNodeID(None)
        slicer.mrmlScene.RemoveNode(loaded_seg)
        slicer.mrmlScene.RemoveNode(labelmap_node)

    def make_only_segments_visible(self, segment_names, segmentation_node):
        display_node = segmentation_node.GetDisplayNode()
        for seg_id in segmentation_node.GetSegmentation().GetSegmentIDs():
            is_visible = segmentation_node.GetSegmentation().GetSegment(seg_id).GetName() in segment_names
            display_node.SetSegmentVisibility(seg_id, is_visible)
            



class SegmentEditorManager:
    def __init__(self, segmentation_node, volume_node=None, mask_mode=None, overwrite=None, mask_segment_id=None):
        self.segmentation_node = segmentation_node
        self.volume_node = volume_node
        self.ct = CoordinateTransformer(volume_node) if volume_node else None
        self.segment_editor_widget = slicer.qMRMLSegmentEditorWidget()
        self.segment_editor_node = slicer.vtkMRMLSegmentEditorNode()
        slicer.mrmlScene.AddNode(self.segment_editor_node)
        
        self.segment_editor_widget.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        self.segment_editor_widget.setMRMLScene(slicer.mrmlScene)

        if volume_node:
            self.segment_editor_widget.setSourceVolumeNode(volume_node)

        self.segment_editor_node.SetOverwriteMode(
            overwrite if overwrite is not None else slicer.vtkMRMLSegmentEditorNode.OverwriteNone
        )

        if mask_mode is not None:
            self.segment_editor_node.SetMaskMode(mask_mode)

        if mask_segment_id is not None:
            self.segment_editor_node.SetMaskMode(5)
            self.segment_editor_node.SetMaskSegmentID(mask_segment_id)

        self.segmentation_node.RemoveClosedSurfaceRepresentation()

    @log_method_call
    def set_segment(self, segment_id):
        self.segment_editor_node.SetSelectedSegmentID(segment_id)
        
    @log_method_call
    def apply_effect(self, effect_name, params: dict):
        self.segment_editor_widget.setActiveEffectByName(effect_name)
        effect = self.segment_editor_widget.activeEffect()
        for key, val in params.items():
            effect.setParameter(key, val)
        effect.self().onApply()
        
    @log_method_call
    def island_effect(self, segment_id, operation="KEEP_LARGEST_ISLAND", min_size=10000):
        logging.debug(f"Island effect on: {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Islands", {
            "Operation": operation,
            "MinimumSize": str(min_size)
        })
    @log_method_call
    def close_holes(self, segment_id, kernel_size=10):
        logging.debug(f"Close holes on: {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Smoothing", {
            "SmoothingMethod": "MORPHOLOGICAL_CLOSING",
            "KernelSizeMm": kernel_size
        })
    @log_method_call
    def smoothing(self, segment_id, method="GAUSSIAN", kernel_size=1):
        logging.debug(f"Smoothing on: {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Smoothing", {
            "SmoothingMethod": method,
            "GaussianStandardDeviationMm": kernel_size
        })
    @log_method_call
    def hollow(self, segment_id, thickness_mm=1, mode="OUTSIDE_SURFACE"):
        logging.debug(f"Hollowing on: {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Hollow", {
            "ShellThicknessMm": thickness_mm,
            "ShellMode": mode,
            "ApplyToAllVisibleSegments": 0
        })
    @log_method_call
    def margin(self, segment_id, kernel_size=1):
        logging.debug(f"Margining on: {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Margin", {
            "MarginSizeMm": kernel_size,
            "ApplyToAllVisibleSegments": 0
        })
    @log_method_call
    def copy_segment(self, new_segment_name, from_segment_id):
        logging.debug(f"Copying {from_segment_id} -> {new_segment_name}")
        new_id = self.segmentation_node.GetSegmentation().AddEmptySegment(new_segment_name)
        self.segment_editor_node.SetSelectedSegmentID(new_id)
        
        from_id = from_segment_id
        if not from_segment_id.startswith("Segment"):
            from_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(from_segment_id)
            
        self.apply_effect("Logical operators", {
            "Operation": SegmentEditorEffects.LOGICAL_COPY,
            "ModifierSegmentID": from_id,
            "BypassMasking": 1
        })
        return new_id
    @log_method_call
    def subtract_segment(self, segment_id, subtract_id):
        logging.debug(f"Subtracting {subtract_id} from {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Logical operators", {
            "Operation": SegmentEditorEffects.LOGICAL_SUBTRACT,
            "ModifierSegmentID": self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(subtract_id),
            "BypassMasking": 1
        })
    @log_method_call
    def add_segment(self, segment_id, add_id):
        logging.debug(f"Adding {add_id} into {segment_id}")
        seg_id = segment_id
        if not seg_id.startswith("Segment"):
            seg_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_id)
        self.set_segment(seg_id)
        self.apply_effect("Logical operators", {
            "Operation": SegmentEditorEffects.LOGICAL_UNION,
            "ModifierSegmentID": self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(add_id),
            "BypassMasking": 1
        })
    @log_method_call
    def threshold(self, new_segment_name, min_t, max_t, mask_segment_id=None):
        logging.debug(f"Thresholding {new_segment_name} with range {min_t}-{max_t}")
        new_id = self.segmentation_node.GetSegmentation().AddEmptySegment(new_segment_name)
        self.segment_editor_node.SetSelectedSegmentID(new_id)
        if mask_segment_id:
            self.segment_editor_node.SetMaskMode(5)
            self.segment_editor_node.SetMaskSegmentID(mask_segment_id)
        self.apply_effect("Threshold", {
            "MinimumThreshold": str(min_t),
            "MaximumThreshold": str(max_t)
        })
        return new_id
    
    @log_method_call
    def cleanup(self):
        slicer.mrmlScene.RemoveNode(self.segment_editor_node)
        
    @log_method_call
    def get_centroids_of_segment(self):
        logging.debug(f"get_centroids_of_segment | computing centroids for {self.segmentation_node.GetName()} on volume {self.volume_node.GetName()}")
        segStatLogic = SegmentStatistics.SegmentStatisticsLogic()
        segStatLogic.getParameterNode().SetParameter("Segmentation", self.segmentation_node.GetID())
        segStatLogic.getParameterNode().SetParameter("ScalarVolume", self.volume_node.GetID())
        segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.centroid_ras.enabled", "True")

        for key in [
            "LabelmapSegmentStatisticsPlugin.volume_mm3",
            "LabelmapSegmentStatisticsPlugin.volume_cm3",
            "ScalarVolumeSegmentStatisticsPlugin.voxel_count",
            "ScalarVolumeSegmentStatisticsPlugin.volume_mm3",
            "ScalarVolumeSegmentStatisticsPlugin.volume_cm3",
            "ScalarVolumeSegmentStatisticsPlugin.min",
            "ScalarVolumeSegmentStatisticsPlugin.max",
            "ScalarVolumeSegmentStatisticsPlugin.mean",
            "ScalarVolumeSegmentStatisticsPlugin.stdev",
            "ScalarVolumeSegmentStatisticsPlugin.median",
        ]:
            segStatLogic.getParameterNode().SetParameter(key, "False")

        segStatLogic.computeStatistics()
        data = segStatLogic.getStatistics()
        logging.debug(f"get_centroids_of_segment | centroids computed: {data}")
        return data

    @log_method_call
    def get_centroids_of_segment_numpy(self, segment_name, slice_height=None,volume_node = None):
        logging.debug(f"get_centroids_of_segment_numpy | {segment_name}")
        vol = self.volume_node if volume_node is None else volume_node

        seg_array = arrayFromSegmentBinaryLabelmap(self.segmentation_node, self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name), vol)
        if slice_height is not None:
            h = min(slice_height, seg_array.shape[0] - 1)
            centroids = np.mean(np.argwhere(seg_array[int(h), :, :] == 1), axis=0)
            return int(centroids[0]), int(centroids[1])
        
        centroids = np.mean(np.argwhere(seg_array == 1), axis=0)
        return (
            int(centroids[2]) if not np.isnan(centroids[2]) else np.nan,
            int(centroids[1]) if not np.isnan(centroids[1]) else np.nan,
            int(centroids[0]) if not np.isnan(centroids[0]) else np.nan,
        )

    @log_method_call
    def get_segment_centroid_ras(self, segment_name: str, slice_height=None,coord_t=None):

        ct = self.ct if coord_t is None else coord_t
        if slice_height is not None:
            x, y = self.get_centroids_of_segment_numpy(segment_name, slice_height= slice_height,volume_node=ct.volume_node)
            ras = ct.ijk_to_ras([slice_height, x,y])
        else:
            ijk = self.get_centroids_of_segment_numpy(segment_name,slice_height=None,volume_node=ct.volume_node)
            ras = ct.ijk_to_ras(ijk)
       
        return np.array(ras)
    
    def get_segment_id(self, segment_name: str):
        return self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name)


    def get_volume_spacing(self):
        return self.volume_node.GetSpacing()
        
    
    @log_method_call
    def _set_only_segments_visible(self, segments: list[str]):
        logging.debug(f"_set_only_segments_visible | Target segments: {segments} on node: {self.segmentation_node.GetName()}")
        display_node = self.segmentation_node.GetDisplayNode()

        if display_node is None:
            logging.warning("_set_only_segments_visible | No display node found for segmentation")
            return

        display_node.SetAllSegmentsVisibility(False)
        for segment_id in segments:
            logging.debug(f"_set_only_segments_visible | Setting segment '{segment_id}' to visible")
            display_node.SetSegmentVisibility(segment_id, True)
            

    @log_method_call
    def extract_centerline(
        self,
        segment_id,
        case_id,
        setting,
        centerline_endpoints,
        name="full_centerline",
        sampling_distance=1,
    ):
        logging.debug(
            f"extract_centerline | creating centerline '{name}' for segment '{segment_id}' and case '{case_id}'"
        )
        logging.debug(f"endpoints: {centerline_endpoints}")
        logging.debug(f"segmentation node: {self.segmentation_node.GetName()}")
        if self.volume_node:
            logging.debug(f"volume node: {self.volume_node.GetName()}")
        logging.debug(f"sampling distance: {sampling_distance}")

        ecWidgetRepresentation = slicer.modules.extractcenterline.widgetRepresentation()
        ecUI = ecWidgetRepresentation.self().ui

        ecUI.inputSurfaceSelector.setCurrentNode(self.segmentation_node)
        ecUI.inputSegmentSelectorWidget.setCurrentSegmentID(segment_id)

        centerline_endpoints_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", f"{case_id}_{setting}_{name}_endpoints"
        )
        for point in centerline_endpoints:
            centerline_endpoints_node.AddControlPoint(point)

        model = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", f"{case_id}_{setting}_{name}_model"
        )
        curve = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", f"{case_id}_{setting}_{name}_centerline"
        )

        ecUI.preprocessInputSurfaceModelCheckBox.checked = True
        ecUI.endPointsMarkupsSelector.setCurrentNode(centerline_endpoints_node)
        ecUI.outputCenterlineModelSelector.setCurrentNode(model)
        ecUI.outputCenterlineCurveSelector.setCurrentNode(curve)
        ecUI.decimationAggressivenessWidget.setValue(4)
        ecUI.targetKPointCountWidget.value = 100.0
        ecUI.curveSamplingDistanceSpinBox.setValue(sampling_distance)

        print(f"ready to extract centerline {name}")
        ecUI.applyButton.click()

        logging.debug(f"curve created: {curve.GetName()}")
        logging.debug(f"model created: {model.GetName()}")

        # Return the created centerline curve node
        return slicer.util.getNode(f"{case_id}_{setting}_{name}_centerline (0)")
    
    
    @log_method_call
    def compute_segment_volumes(self, features: dict) -> dict:
        logging.debug("compute_segment_volumes | Calculating volumes for predefined segments")

        segments_of_interest = [
            "located_renal_artery_right",
            "located_renal_artery_left",
            "located_sma",
            "located_celiac",
            "located_common_iliac_right",
            "located_common_iliac_left",
            "located_neck",
            "located_internal_iliac_left",
            "located_internal_iliac_right",
            "located_aneurysm_sac_lumen",
            "located_aneurysm_sac_thrombus",
            "located_aneurysm_sac_calc",
            "located_external_iliac_right",
            "located_external_iliac_left",
            "located_distal_sealing_right",
            "located_distal_sealing_left",
        ]

        spacing = self.volume_node.GetSpacing()
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # in mm³

        for segment_name in segments_of_interest:
            try:
                segment_id = self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
                if segment_id is None:
                    logging.warning(f"Segment '{segment_name}' not found in segmentation")
                    continue

                seg_array = arrayFromSegmentBinaryLabelmap(self.segmentation_node, segment_id, self.volume_node)
                n_voxels = np.sum(seg_array)
                volume_mm3 = n_voxels * voxel_volume
                volume_cm3 = volume_mm3 / 1000.0

                features[f"{segment_name}_volume"] = volume_cm3
                logging.debug(f"{segment_name}: {volume_cm3:.2f} cm³")

            except Exception as e:
                logging.error(f"Error processing segment '{segment_name}': {str(e)}")

        # Compute total infrarenal aortic volume
        try:
            total_volume = (
                features["located_neck_volume"]
                + features["located_aneurysm_sac_thrombus_volume"]
                + features["located_aneurysm_sac_calc_volume"]
                + features["located_aneurysm_sac_lumen_volume"]
            )
            features["infrarenal_aorta_global_volume"] = total_volume
            logging.debug(f"infrarenal_aorta_global_volume: {total_volume:.2f} cm³")
        except KeyError as e:
            logging.warning(f"Missing component for infrarenal aorta volume: {str(e)}")

        return features
    
    @log_method_call
    def get_bounds_of_segment(self, segment_id: str, box_size: int = None):

    
        seg_array = arrayFromSegmentBinaryLabelmap(
            self.segmentation_node, segment_id, self.volume_node
        )

        Z, Y, X = seg_array.shape


        if box_size is not None:
            y_c = Y // 2
            x_c = X // 2

            y_min = max(0, y_c - box_size)
            y_max = min(Y, y_c + box_size)

            x_min = max(0, x_c - box_size)
            x_max = min(X, x_c + box_size)

            seg_array = seg_array[:, y_min:y_max, x_min:x_max]

            Z, Y, X = seg_array.shape

 
        def find_bounds(arr):
            z_min = z_max = y_min = y_max = x_min = x_max = None

            # Z bounds
            for i in range(arr.shape[0]):
                if np.any(arr[i, :, :]):
                    z_min = i
                    break
            for i in range(arr.shape[0] - 1, -1, -1):
                if np.any(arr[i, :, :]):
                    z_max = i
                    break

            # Segment is empty
            if z_min is None:
                return None, None, None, None, None, None

            # Y bounds
            for j in range(arr.shape[1]):
                if np.any(arr[z_min:z_max+1, j, :]):
                    y_min = j
                    break
            for j in range(arr.shape[1] - 1, -1, -1):
                if np.any(arr[z_min:z_max+1, j, :]):
                    y_max = j
                    break

            # X bounds
            for k in range(arr.shape[2]):
                if np.any(arr[z_min:z_max+1, :, k]):
                    x_min = k
                    break
            for k in range(arr.shape[2] - 1, -1, -1):
                if np.any(arr[z_min:z_max+1, :, k]):
                    x_max = k
                    break

            return z_min, z_max, y_min, y_max, x_min, x_max

        z_min, z_max, y_min, y_max, x_min, x_max = find_bounds(seg_array)


        logging.debug(
            f"Final segment bounds in cropped volume: "
            f"Z=[{z_min},{z_max}]  Y=[{y_min},{y_max}]  X=[{x_min},{x_max}]"
        )

        return {
            "z_min": z_min, "z_max": z_max,
            "y_min": y_min, "y_max": y_max,
            "x_min": x_min, "x_max": x_max,
            "seg_array": seg_array
        }


    @log_method_call
    def extract_walls_of_segment(
        self,
        segment_id_to_extract_walls_from: str,
        helper_segment_id: str,
        final_segment_name: str,
        thickness: float = 2.0,
    ):
        
        logging.debug(f"extract_walls_of_segment | Extracting walls from: {segment_id_to_extract_walls_from}")
        logging.debug(f"Segmentation node: {self.segmentation_node.GetName()}")
        logging.debug(f"Volume node: {self.volume_node.GetName()}")
        logging.debug(f"Helper segment to subtract: {helper_segment_id}")
        logging.debug(f"Final wall segment name: {final_segment_name}")

        self.island_effect(
            segment_id=segment_id_to_extract_walls_from,
            operation="KEEP_LARGEST_ISLAND",
            min_size=100,
        )

        self.smoothing(
            segment_id=segment_id_to_extract_walls_from,
            method="GAUSSIAN",
            kernel_size=1,
        )

        self.copy_segment(final_segment_name, segment_id_to_extract_walls_from)

        self.hollow(
            segment_id=final_segment_name,
            thickness_mm=thickness,
            mode="OUTSIDE_SURFACE",
        )

        self.subtract_segment(final_segment_name, helper_segment_id)

        logging.debug(f"extract_walls_of_segment | Final wall segment '{final_segment_name}' created successfully.")
        
    @log_method_call
    def extract_diameters_fedez(self, seg_name, ref_fetta, points_of_interest, offset=0, ascending=True,straight_vol = None,straight_transform = None,save_csv_path = None, view="axial", retain_biggest=False,crop_spatial=None):
        
        from feret.main import Calculater
        
        import numpy as np
        from skimage.measure import label, regionprops

        def keep_largest_component(mask):
            labeled = label(mask)             
            if labeled.max() == 0:
                return mask                    
            
            # get the region with the largest area
            largest_region = max(regionprops(labeled), key=lambda r: r.area)
            largest_label = largest_region.label
            
            # keep only that region
            return (labeled == largest_label).astype(mask.dtype)

        
        
        def feret_diameters_and_coords(img, edge=False):
            calc = Calculater(img, edge)
            calc.calculate_minferet()
            calc.calculate_maxferet()
            
            min_dist, min_endpoints = calc.calculate_distances(
                calc.minf_angle - np.pi / 2
            )

            return {
                "min_diameter": calc.minf,
                "min_endpoints": min_endpoints, 
                "min_coords": calc.minf_coords,
                "min_angle": calc.minf_angle,  
                "max_diameter": calc.maxf,
                "max_coords": calc.maxf_coords,
                "max_angle": calc.maxf_angle       
            }
        
        vol = self.volume_node if straight_vol is None else straight_vol
        if straight_transform is not None:
            self.segmentation_node.SetAndObserveTransformNodeID(
                    straight_transform.GetID()
                )
            
        seg_array = arrayFromSegmentBinaryLabelmap(
            self.segmentation_node, seg_name, vol
        )
        
        if crop_spatial is not None:

            x_start = (64 - crop_spatial) // 2
            y_start = (64 - crop_spatial) // 2
            mask = np.zeros((64, 64), dtype=bool)
            mask[x_start:x_start + crop_spatial, y_start:y_start + crop_spatial] = True
            seg_array = seg_array*mask
            
            
        spacings = vol.GetSpacing()  # (x,y,z)
        depth = seg_array.shape[0]

        toRtnMaxFerets = []
        toRtnMinFerets = []
        rows_to_save = []
        
        for p in points_of_interest:
            slice_height = int(p / spacings[2])
            if not ascending:
                slice_height = ref_fetta - slice_height
            else:
                slice_height = ref_fetta + slice_height

            offsetUp = offset
            offsetDown = offset
            if p == 0:
                if ascending:
                    offsetUp = 0
                else:
                    offsetDown = 0

            start_idx = max(0, slice_height - offsetUp)
            end_idx = min(depth - 1, slice_height + offsetDown)
            indices_of_interest = range(start_idx, end_idx + 1)

            min_ferets = []
            max_ferets = []

            for idx in indices_of_interest:
                seg_slice = seg_array[idx, :, :]
                if retain_biggest:
                    seg_slice = keep_largest_component(seg_slice)
                if np.all(seg_slice == 0):
                    continue
                feret_info = feret_diameters_and_coords(seg_array[idx, :, :])

                # Convert diameter from pixels to mm
                min_dia = feret_info['min_diameter'] * spacings[0]
                max_dia = feret_info['max_diameter'] * spacings[0]

                min_ferets.append(min_dia)
                max_ferets.append(max_dia)

                min_coords = feret_info['min_coords']
                max_coords = feret_info['max_coords']
                
                min_pts = feret_info["min_endpoints"]

                row_min = [
                    p, idx, "min", min_dia,
                    min_pts[0][0], min_pts[0][1],
                    min_pts[1][0], min_pts[1][1],
                    "", "",                      # no p3
                    feret_info["min_angle"], view
                ]
                row_max = [
                    p, idx, "max", max_dia,
                    *max_coords[0], *max_coords[1], "", "", feret_info['max_angle'],view
                ]
                rows_to_save.append(row_min)
                rows_to_save.append(row_max)

            toRtnMinFerets.append(np.mean(min_ferets))
            toRtnMaxFerets.append(np.mean(max_ferets))

        if straight_transform is not None:
            self.segmentation_node.SetAndObserveTransformNodeID(None)

        if save_csv_path:
            dirpath = os.path.dirname(save_csv_path)
            if dirpath:  # Only create directories if one is specified
                os.makedirs(dirpath, exist_ok=True)
            with open(save_csv_path, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    "point_distance", "slice_index", "feret_type", "diameter_mm",
                    "p1_y", "p1_x", "p2_y", "p2_x", "p3_y", "p3_x", "angle_rad","view"
                ])
                writer.writerows(rows_to_save)
            logging.info(f"Saved Feret diameter data to: {save_csv_path}")

        return toRtnMinFerets, toRtnMaxFerets
    
    
    @staticmethod
    def transfer_segmentation_from_seg_nodes_world_to_cpr(segmentations,sm_from,sm_to,transform):
        sm_from.segmentation_node.SetAndObserveTransformNodeID(transform.GetID())
        SegmentEditorManager.transfer_segmentations(segmentations,sm_from,sm_to)
        sm_from.segmentation_node.SetAndObserveTransformNodeID(None)
    
    @staticmethod
    def transfer_segmentations(segmentations,sm_from,sm_to):

        logging.debug(f"transfering segmentations {segmentations}")
        logging.debug(f"from {sm_from.segmentation_node.GetName()}")
        logging.debug(f"to {sm_to.segmentation_node.GetName()}")

        for segmentation in segmentations:
            seg_array = arrayFromSegmentBinaryLabelmap(
                sm_from.segmentation_node, segmentation, sm_from.volume_node
            )

            updateSegmentBinaryLabelmapFromArray(
                seg_array,
                sm_to.segmentation_node,
                segmentation,
                sm_from.volume_node,
            )

    
    
    @staticmethod
    def load_segmentation_from_node(name):
        try:
            return slicer.util.getNode(f"{name}.nii.gz")
        except Exception as e:
            logging.error(f"Failed to load segmentation {name}: {e}")
            return None

  
   
    @log_method_call
    def mask_volume(self,
                    segmentations,
                    blacklist = {"aorta","iliac_artery_left","iliac_artery_right","iliac_vena_left","iliac_vena_right"},
                    mask_value=-1000):
        logging.debug(
            f"Masking volume {self.volume_node.GetName()} based on segmentations in {self.segmentation_node.GetName()}"
        )

        vol_array = arrayFromVolume(
            self.volume_node
        )

        for seg_name in segmentations:
            if seg_name in blacklist:
                continue

            segment = self.segmentation_node.GetSegmentation().GetSegment(seg_name)
            if segment is None:
                logging.warning(f"Segment '{seg_name}' not found in combined segmentation.")
                continue

            mask = arrayFromSegmentBinaryLabelmap(
                self.segmentation_node, seg_name, self.volume_node
            )

            vol_array[mask != 0] = mask_value

        updateVolumeFromArray(self.volume_node, vol_array)  
        logging.debug("Volume masking completed.")

        return self.volume_node, vol_array
    
    

    @log_method_call
    def add_empty_segment(self, segment_name: str):
   
        if self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name):
            logging.warning(f"Segment '{segment_name}' already exists. Skipping addition.")
            return

        self.segmentation_node.GetSegmentation().AddEmptySegment(segment_name)
        logging.debug(f"Added empty segment '{segment_name}' to segmentation node.")
        
    def retain_segment_inside_box(self, segment_id: str, box_bounds: dict):

        seg_array = arrayFromSegmentBinaryLabelmap(self.segmentation_node, segment_id, self.volume_node)
        z_len, y_len, x_len = seg_array.shape
        z_min = box_bounds["z_min"] if "z_min" in box_bounds else 0
        z_max = box_bounds["z_max"] if "z_max" in box_bounds else z_len - 1
        y_min = box_bounds["y_min"] if "y_min" in box_bounds else 0
        y_max = box_bounds["y_max"] if "y_max" in box_bounds else y_len - 1
        x_min = box_bounds["x_min"] if "x_min" in box_bounds else 0
        x_max = box_bounds["x_max"] if "x_max" in box_bounds else x_len - 1

        box_mask = np.zeros_like(seg_array, dtype=bool)
        box_mask[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = True

        new_seg_array = np.where(box_mask, seg_array, 0)

        slicer.util.updateSegmentBinaryLabelmapFromArray(new_seg_array, self.segmentation_node, segment_id, self.volume_node)
        logging.debug(f"Retained part of segment '{segment_id}' inside specified box bounds.")
        
        
    def add_segment_from_numpy(self,segment_id,segmentation):
        self.add_empty_segment(segment_id)
        
        slicer.util.updateSegmentBinaryLabelmapFromArray(segmentation, self.segmentation_node, segment_id, self.volume_node)
        
    def set_segment_name(self,segment_id,new_name):
        self.segmentation_node.GetSegmentation().GetSegment(segment_id).SetName(new_name)
        logging.debug(f"changed name of segment {segment_id} to {new_name}")
        
        
    def intersect_segment(self,seg_id,modifier_seg_id):
        
        seg_array = arrayFromSegmentBinaryLabelmap(self.segmentation_node, seg_id, self.volume_node)
        seg_modifier_array = arrayFromSegmentBinaryLabelmap(
        self.segmentation_node, modifier_seg_id, self.volume_node
    )

        seg_array = seg_array & seg_modifier_array
        updateSegmentBinaryLabelmapFromArray(
        seg_array, self.segmentation_node, seg_id, self.volume_node
    )
        
    def getSegmentationAtRASNeighborhood(self, ras, segmentation_names, neighborhood_radius=2):

        ijk = self.ct.ras_to_ijk(ras)
        i, j, k = map(int, map(round, ijk))

        max_count = 0
        best_segmentation = None
        voxel_counts = 0
        for name in segmentation_names:
            
            seg = arrayFromSegmentBinaryLabelmap(self.segmentation_node, name, self.volume_node)
            
            i_min = max(0, i - neighborhood_radius)
            i_max = min(seg.shape[2], i + neighborhood_radius + 1)
            j_min = max(0, j - neighborhood_radius)
            j_max = min(seg.shape[1], j + neighborhood_radius + 1)
            k_min = max(0, k - neighborhood_radius)
            k_max = min(seg.shape[0], k + neighborhood_radius + 1)

            neighborhood = seg[k_min:k_max, j_min:j_max, i_min:i_max]
            count = np.count_nonzero(neighborhood == 1)

            if count > max_count:
                max_count = count
                best_segmentation = name
                voxel_counts = np.count_nonzero(seg)

        return best_segmentation, max_count,voxel_counts

    def export_segment_as_vol(self,segment_name,path):

        labelmap_node = slicer.mrmlScene.AddNewNodeByClass(
                    "vtkMRMLScalarVolumeNode", self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name)
                )

        m = vtk.vtkMatrix4x4()
        self.volume_node.GetIJKToRASDirectionMatrix(m)
        labelmap_node.SetIJKToRASDirectionMatrix(m)
        labelmap_node.SetSpacing(self.volume_node.GetSpacing())
        labelmap_node.SetOrigin(self.volume_node.GetOrigin())


        segmentation_arr_to_export  = arrayFromSegmentBinaryLabelmap(self.segmentation_node, self.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(segment_name), self.volume_node)
        segmentation_arr_to_export = segmentation_arr_to_export[:,:,:]
        updateVolumeFromArray(labelmap_node,segmentation_arr_to_export)


        slicer.util.exportNode(
                    labelmap_node,
                    os.path.join(path, f"{labelmap_node.GetName()}.nii.gz"),
                    {"useCompression": 0},
                    world=True,
                )

        slicer.mrmlScene.RemoveNode(labelmap_node)

        return True 


def hide_scene():
    for node in list(slicer.mrmlScene.GetNodesByClass('vtkMRMLDisplayableNode')):
        if node.GetName() in ["Combined","Red Transform", "Red Volume Slice","Green Transform","Green Volume Slice","Yellow Transform","Yellow Volume Slice"]:
            continue
        print(node.GetName())
        
        d_s = node.GetDisplayNode()
        if d_s:
            d_s.SetVisibility(False)
            




def load_data(folder_path, case_id, setting, segmentations):

    ct_filename = f"CT_{case_id}_{setting}_001_0002.nii.gz"
    ct_path = os.path.join(folder_path,case_id, ct_filename)

    seg_folder = os.path.join(folder_path,case_id, f"Predictions_{setting}")

    logging.debug(f"Loading CT scan: {ct_filename}")
    if not os.path.exists(ct_path):
        logging.error(f"CT file not found: {ct_path}")
        raise FileNotFoundError(f"CT file not found: {ct_path}")

    slicer.util.loadVolume(ct_path)

    for seg_name in segmentations:
        seg_filename = f"{seg_name}.nii.gz"
        seg_path = os.path.join(seg_folder, seg_filename)

        logging.debug(f"Loading segmentation '{seg_name}' from {seg_path}")
        if not os.path.exists(seg_path):
            logging.error(f"Segmentation file not found: {seg_path}")
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

        slicer.util.loadSegmentation(seg_path)


def get_max_hu_from_volume(volume_node):

    if not volume_node:
        logging.error("Volume node is None.")
        raise ValueError("Volume node is None.")

    vol_array = arrayFromVolume(volume_node)
    max_hu = np.max(vol_array)

    logging.debug(f"Maximum HU value in the volume: {max_hu}")

    return max_hu


def extract_diameters_of_centerline(centerline_node):
    logging.debug(f"Extracting diameters from {centerline_node.GetName()}")
    points = slicer.util.arrayFromMarkupsCurvePoints(centerline_node, world=True)
    indices = centerline_node.GetCurveWorld().GetPointData().GetArray("PedigreeIDs")
    n_points = len(points)

    mis_radii, ce_radii = np.zeros(n_points), np.zeros(n_points)
    radius_vals = centerline_node.GetMeasurement("Radius").GetControlPointValues()

    for i in range(n_points - 1):
        poly = compute_cross_section_polydata(i, centerline_node, slicer.util.getNode("Combined"))
        if not poly:
            continue
        mass = vtk.vtkMassProperties()
        mass.SetInputData(poly)
        mass.Update()
        ce_radii[i] = np.sqrt(mass.GetSurfaceArea() / np.pi)

        float_idx = indices.GetValue(i)
        idx_a, idx_b = int(float_idx), int(float_idx) + 1
        r_a, r_b = radius_vals.GetValue(idx_a), radius_vals.GetValue(idx_b)
        interpolated = r_a * (float_idx - idx_a) + r_b * (idx_b - float_idx)
        mis_radii[i] = interpolated

    mis_radii[-1] = radius_vals.GetValue(radius_vals.GetNumberOfValues() - 1)
    return mis_radii * 2, ce_radii * 2

def get_curve_point_to_world_transform(point_index, node):
    mtx = vtk.vtkMatrix4x4()
    node.GetCurvePointToWorldTransformAtPointIndex(point_index, mtx)
    return mtx

def compute_cross_section_polydata(point_index, centerline_node, segmentation_node):
    mtx = get_curve_point_to_world_transform(point_index, centerline_node)
    center, normal = [mtx.GetElement(i, 3) for i in range(3)], [mtx.GetElement(i, 2) for i in range(3)]

    plane = vtk.vtkPlane()
    plane.SetOrigin(center)
    plane.SetNormal(normal)

    segmentation_node.CreateClosedSurfaceRepresentation()
    surface = vtk.vtkPolyData()
    segmentation_node.GetClosedSurfaceRepresentation("aorta_iliacs", surface)

    if segmentation_node.GetParentTransformNode():
        transform = vtk.vtkGeneralTransform()
        slicer.vtkMRMLTransformNode.GetTransformBetweenNodes(
            segmentation_node.GetParentTransformNode(), None, transform
        )
        tf_filter = vtk.vtkTransformPolyDataFilter()
        tf_filter.SetTransform(transform)
        tf_filter.SetInputData(surface)
        tf_filter.Update()
        surface = tf_filter.GetOutput()

    cutter = vtk.vtkCutter()
    cutter.SetInputData(surface)
    cutter.SetCutFunction(plane)
    cutter.Update()

    if not cutter.GetOutput().GetPoints() or cutter.GetOutput().GetNumberOfPoints() < 3:
        slicer.util.showStatusMessage("Could not cut segment. Is it visible in 3D view?", 3000)
        return None

    conn = vtk.vtkConnectivityFilter()
    conn.SetInputData(cutter.GetOutput())
    conn.SetClosestPoint(center)
    conn.SetExtractionModeToClosestPointRegion()
    conn.Update()

    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(conn.GetOutput())
    triangulator.Update()

    return triangulator.GetOutput()



def process_margin(seg_mgr: SegmentEditorManager, base_segment="aorta_iliacs", margin_segment="aorta_iliacs_margin_m_3", wall_segments=None):

    seg_mgr.copy_segment(margin_segment, base_segment)

    seg_mgr.smoothing(margin_segment, method="MEDIAN", kernel_size=1)

    seg_mgr.margin(margin_segment, kernel_size=-3)
    
    if wall_segments is not None:
        for ref_segment_name, params in wall_segments.items():
            seg_mgr.extract_walls_of_segment(
                seg_mgr.segmentation_node.GetSegmentation().GetSegmentIdBySegmentName(ref_segment_name),margin_segment,
                params['name'],
                thickness=params['thickness'])

