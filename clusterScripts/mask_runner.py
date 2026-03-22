#!/usr/bin/env python3
"""
masking_runner_V.py
-------------------
Processa TUTTI i pazienti presenti in SCAN_DIR.
Cerca specificamente le scan di tipo V (*_V.nii.gz) e le relative
segmentazioni in results/total_segmentator_type_V/.
"""

import os
import sys
import glob
import logging
import numpy as np
import nibabel as nib
import concurrent.futures

# ---------------------------------------------------------------------------
# Configurazione Percorsi (Cluster Version)
# ---------------------------------------------------------------------------
# os.getcwd() will be your main "scripts" folder when you run the job
WORK_DIR = os.getcwd() 

SCAN_DIR = os.path.join(WORK_DIR, "../patients")          # Where the raw CTs live
SEG_DIR  = os.path.join(WORK_DIR, "segmentatorResultsVfiles")  # Your V-scan segmentations
OUT_DIR  = os.path.join(WORK_DIR, "masked_scans_V")       # Where to save the output

MASK_VALUE = -1000  
WINDOW_MARGIN_VOXELS = 5

# Whitelist strutture da tenere
STRUCTURES_TO_KEEP = {
    "aorta", "iliac_artery_left", "iliac_artery_right", "iliac_vena_left",
    "iliac_vena_right", "inferior_vena_cava", "portal_vein_and_splenic_vein",
    "kidney_left", "kidney_right", "kidney_cyst_left", "kidney_cyst_right",
    "adrenal_gland_left", "adrenal_gland_right", "liver", "spleen",
    "pancreas", "gallbladder", "stomach", "duodenum", "small_bowel",
    "colon", "prostate", "urinary_bladder"
}

WINDOW_INFERIOR_STRUCTURE = "sacrum"
WINDOW_SUPERIOR_STRUCTURE = "liver"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("masking_V")

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def load_nifti(path: str):
    img = nib.load(path)
    return np.asarray(img.dataobj), img

def find_slice_axis(affine: np.ndarray) -> int:
    rot = affine[:3, :3]
    si_components = np.abs(rot[2, :])
    return int(np.argmax(si_components))

def get_z_extent(seg_array: np.ndarray):
    nz = np.nonzero(seg_array)
    if len(nz[0]) == 0: return None
    return {ax: (int(nz[ax].min()), int(nz[ax].max())) for ax in range(seg_array.ndim)}

def mask_outside_window(vol, axis, lo, hi, val):
    n = vol.shape[axis]
    keep = np.zeros(n, dtype=bool)
    keep[lo:hi + 1] = True
    shape = [1] * vol.ndim
    shape[axis] = n
    outside_mask = ~np.broadcast_to(keep.reshape(shape), vol.shape)
    vol[outside_mask] = val
    return int(outside_mask.sum())

# ---------------------------------------------------------------------------
# Core Processing
# ---------------------------------------------------------------------------

def process_patient(patient_folder_name: str):
    """Esegue la pulizia per un singolo paziente (es. 'pz001')"""

    # ---> RESUME CHECK <---
    os.makedirs(OUT_DIR, exist_ok=True)
    out_filename = f"{patient_folder_name}_V_masked.nii.gz"
    out_path = os.path.join(OUT_DIR, out_filename)
    
    if os.path.exists(out_path):
        log.info(f"[{patient_folder_name}] Masked file already exists. Skipping!")
        return
    # ----------------------

    path_pz = os.path.join(SCAN_DIR, patient_folder_name)
    
    # 1. Identifica il file CT di tipo V
    v_scans = glob.glob(os.path.join(path_pz, "*_V.nii.gz"))
    if not v_scans:
        log.warning(f"[{patient_folder_name}] Nessun file *_V.nii.gz trovato. Salto.")
        return
    ct_path = v_scans[0]
    
    # 2. Trova la cartella segmentazione corrispondente
    patient_id_num = "".join(filter(str.isdigit, patient_folder_name))
    
    all_seg_folders = [d for d in os.listdir(SEG_DIR) if os.path.isdir(os.path.join(SEG_DIR, d))]
    target_seg_folder = None
    for folder in all_seg_folders:
        if f"_{patient_id_num}_" in folder or f"_{int(patient_id_num)}_" in folder:
            target_seg_folder = os.path.join(SEG_DIR, folder)
            break
            
    if not target_seg_folder:
        log.warning(f"[{patient_folder_name}] Cartella segmentazione NON TROVATA in {SEG_DIR}. Salto.")
        return

    log.info(f"--- Processing {patient_folder_name} ---")
    log.info(f"Scan V: {os.path.basename(ct_path)}")
    log.info(f"Seg Folder: {os.path.basename(target_seg_folder)}")

    # Caricamento Volume
    vol_array, ct_img = load_nifti(ct_path)
    vol = vol_array.copy()

    # --- STAGE A: Anatomical Windowing ---
    si_axis = find_slice_axis(ct_img.affine)
    lo, hi = 0, vol.shape[si_axis] - 1
    
    # Calcolo limiti fegato/sacro
    for struct, is_inf in [(WINDOW_INFERIOR_STRUCTURE, True), (WINDOW_SUPERIOR_STRUCTURE, False)]:
        s_path = os.path.join(target_seg_folder, f"{struct}.nii.gz")
        if os.path.isfile(s_path):
            s_arr, _ = load_nifti(s_path)
            extents = get_z_extent(s_arr)
            if extents and si_axis in extents:
                ax_min, ax_max = extents[si_axis]
                if is_inf: lo = max(0, ax_min - WINDOW_MARGIN_VOXELS)
                else: hi = min(vol.shape[si_axis] - 1, ax_max + WINDOW_MARGIN_VOXELS)
    
    if lo > hi: lo, hi = hi, lo
    mask_outside_window(vol, si_axis, lo, hi, MASK_VALUE)

    # --- STAGE B: Structure Masking ---
    total_struct_masked = 0
    for seg_file in glob.glob(os.path.join(target_seg_folder, "*.nii.gz")):
        struct_name = os.path.basename(seg_file).replace(".nii.gz", "")
        if struct_name in STRUCTURES_TO_KEEP:
            continue
        
        seg_arr, _ = load_nifti(seg_file)
        if seg_arr.shape == vol.shape:
            idxs = seg_arr != 0
            vol[idxs] = MASK_VALUE
            total_struct_masked += int(idxs.sum())

    # --- SALVATAGGIO ---
    masked_img = nib.Nifti1Image(vol, ct_img.affine, ct_img.header)
    nib.save(masked_img, out_path)
    log.info(f"Completato! Salvato in: {out_path}")


def main():
    if not os.path.exists(SCAN_DIR):
        log.error(f"SCAN_DIR non esiste: {SCAN_DIR}")
        return

    # Prendi TUTTI i pazienti presenti in scan_test
    patients = sorted([d for d in os.listdir(SCAN_DIR) if os.path.isdir(os.path.join(SCAN_DIR, d))])
    
    if not patients:
        log.error("Nessun paziente trovato in scan_test.")
        return

    log.info(f"Trovati {len(patients)} pazienti da processare.")
    log.info("Avvio del Multiprocessing su 4 CPU...")

    # Usa ProcessPoolExecutor per lanciare 4 pazienti contemporaneamente!
    # Nota: impostiamo max_workers=4 perché hai richiesto --cpus-per-task=4 in SLURM
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        # Questo invierà la lista dei pazienti ai 4 core. 
        # Man mano che un core finisce, prende automaticamente il paziente successivo!
        futures = {executor.submit(process_patient, p): p for p in patients}
        
        for future in concurrent.futures.as_completed(futures):
            p = futures[future]
            try:
                future.result() # Controlla se ci sono stati errori
            except Exception as e:
                log.error(f"Errore critico durante il processamento di {p}: {e}")

    log.info("Tutti i pazienti completati!")

if __name__ == "__main__":
    main()
