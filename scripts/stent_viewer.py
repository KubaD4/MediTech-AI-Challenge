import os
import sys
import signal
import numpy as np
import nibabel as nib
from skimage.measure import marching_cubes
from skimage.morphology import remove_small_objects
import pyvista as pv

signal.signal(signal.SIGINT, signal.SIG_DFL)

def create_pyvista_mesh(verts, faces, intensities=None):
    if verts is None or len(verts) == 0:
        return None
    faces_pv = np.hstack([[3] + list(f) for f in faces])
    mesh = pv.PolyData(verts, faces_pv)
    
    if intensities is not None:
        mesh.point_data["Intensity"] = intensities
        
    return mesh

def visualize_stent_heatmap(nifti_path, metal_th=1000):
    print(f"Caricamento del volume: {nifti_path}")
    img = nib.load(nifti_path)
    volume = img.get_fdata()
    spacing = img.header.get_zooms()[:3]
    
    FAST_STEP = 1

    # ---------------------------------------------------------
    # 1. ESTRAZIONE E STATISTICHE
    # ---------------------------------------------------------
    print("Ricerca voxel dello Stent...")
    mask_metal = volume > metal_th
    mask_metal = remove_small_objects(mask_metal, min_size=800)
    
    stent_voxels = volume[mask_metal]
    
    if len(stent_voxels) == 0:
        print("Nessun voxel supera la soglia specificata!")
        return

    v_min = stent_voxels.min()
    v_max = stent_voxels.max()
    v_mean = stent_voxels.mean()
    
    # Calcoliamo un range molto stretto (es. tra il 5% e il 95% della distribuzione)
    # Questo serve ad "allargare" al massimo il contrasto visivo
    p_low = np.percentile(stent_voxels, 5)
    p_high = np.percentile(stent_voxels, 95)

    print("\n" + "="*50)
    print("📊 STATISTICHE INTENSITÀ STENT (Heatmap Contrast)")
    print("="*50)
    print(f"Valore Minimo Assoluto : {v_min:.4f}")
    print(f"Valore Medio           : {v_mean:.4f}")
    print(f"5° Percentile (Verde)  : {p_low:.4f}")
    print(f"95° Percentile (Rosso) : {p_high:.4f}")
    print(f"Valore Max Assoluto    : {v_max:.4f}")
    print("="*50 + "\n")

    # ---------------------------------------------------------
    # 2. CREAZIONE MESH
    # ---------------------------------------------------------
    print("Estrazione della superficie 3D...")
    verts_m, faces_m, _, _ = marching_cubes(mask_metal, level=0.5, spacing=spacing, step_size=FAST_STEP)
    
    voxel_coords = np.round(verts_m / spacing).astype(int)
    voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, volume.shape[0]-1)
    voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, volume.shape[1]-1)
    voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, volume.shape[2]-1)
    
    # Estraiamo le intensità grezze
    raw_intensities = volume[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]]
    
    # CLIPPING: Questa è la magia matematica. 
    # Tutti i valori sopra p_high diventano p_high (ROSSO PIENO).
    # Tutti i valori sotto p_low diventano p_low (VERDE PIENO).
    clipped_intensities = np.clip(raw_intensities, p_low, p_high)
    
    mesh_metal = create_pyvista_mesh(verts_m, faces_m, clipped_intensities)

    # ---------------------------------------------------------
    # 3. RENDER 3D
    # ---------------------------------------------------------
    print("Avvio visualizzatore 3D (Heatmap)... (Premi CTRL+C per uscire)")
    plotter = pv.Plotter(title="Stent Heatmap (Stretched Contrast)")
    plotter.set_background("white")

    plotter.add_mesh(
        mesh_metal, 
        scalars="Intensity", 
        cmap='RdYlGn_r', 
        clim=[p_low, p_high], # Forza strettamente la barra a questi due valori
        show_scalar_bar=True,
        scalar_bar_args={'title': 'Intensita Normalizzata', 'color': 'black'},
        specular=0.3, 
        ambient=0.2, 
        smooth_shading=True
    )

    plotter.view_xz() 
    plotter.camera.up = (0.0, 0.0, -1.0) 
    
    plotter.add_axes()
    plotter.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # Puntiamo alla TAC mascherata (che dovrebbe avere gli HU originali intatti)
        base_dir = os.path.expanduser("~/Desktop/Challange1")
        test_file = os.path.join(base_dir, "results", "masked_scans", "pz001_masked.nii.gz")
    
    if os.path.exists(test_file):
        # CAMBIAMENTO FONDAMENTALE: 
        # Usiamo 1500 HU come soglia base per estrarre la protesi.
        # I marker in oro, essendo molto più densi, schizzeranno verso i 3000+ HU 
        # e diventeranno i punti rossi nella heatmap!
        visualize_stent_heatmap(test_file, metal_th=1000) 
    else:
        print("File non trovato")