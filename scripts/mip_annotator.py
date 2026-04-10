import os
import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger("Arterial_MIP_Tuner_Annotated")

def ras_to_ijk(affine, point_ras):
    """
    Converte un punto dallo spazio fisico RAS (mm) allo spazio indice IJK (voxel).
    Usa la matrice affine del volume NIfTI.
    """
    inv_affine = np.linalg.inv(affine)
    point_ras_hom = np.append(point_ras, 1.0)
    point_ijk = inv_affine.dot(point_ras_hom)
    return [int(round(coord)) for coord in point_ijk[:3]]

def generate_arterial_mip_variants(input_path, seg_dir, output_dir, z_renale_mm=None, z_colletto_mm=None):
    if not os.path.exists(input_path):
        log.error(f"File di input non trovato: {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path).split('.')[0]

    log.info("Caricamento volume...")
    img = nib.load(input_path)
    data = img.get_fdata()
    affine = img.affine
    vox_size = img.header.get_zooms() # (dx, dy, dz)
    Z_total = data.shape[2]

    # 1. Trovare il baricentro Y (Antero-Posteriore) dell'aorta
    aorta_path = os.path.join(seg_dir, "aorta.nii.gz")
    
    if "straightened" in input_path.lower():
        log.info("Volume raddrizzato rilevato! Uso il centro esatto dell'immagine ortogonalizzata.")
        y_center = data.shape[1] // 2
    elif os.path.exists(aorta_path):
        aorta_mask = nib.load(aorta_path).get_fdata().astype(bool)
        y_coords = np.where(aorta_mask)[1]
        y_center = int(np.median(y_coords))
        log.info(f"Centro Y dell'aorta trovato all'indice: {y_center}")
    else:
        log.warning("Maschera aorta non trovata! Uso il centro esatto del volume.")
        y_center = data.shape[1] // 2

    # =========================================================================
    # TRASFORMAZIONE DA MILLIMETRI A FETTE (IJK)
    # =========================================================================
    draw_annotations = (z_renale_mm is not None and z_colletto_mm is not None)
    if draw_annotations:
        point_renale_ras = [0.0, 0.0, z_renale_mm]
        point_colletto_ras = [0.0, 0.0, z_colletto_mm]

        idx_renale = ras_to_ijk(affine, point_renale_ras)[2]
        idx_colletto = ras_to_ijk(affine, point_colletto_ras)[2]

        log.info(f"Ostio Renale: {z_renale_mm} mm -> Slice (Z) = {idx_renale}")
        log.info(f"Fine Colletto: {z_colletto_mm} mm -> Slice (Z) = {idx_colletto}")

        # Inversione Y immagine vs Z Numpy (a causa del flipud)
        y_renale = Z_total - 1 - idx_renale
        y_colletto = Z_total - 1 - idx_colletto

        # Assicuriamoci che non escano dai bordi
        y_renale = max(0, min(Z_total - 1, y_renale))
        y_colletto = max(0, min(Z_total - 1, y_colletto))

    # ==========================================
    # PARAMETRI DA TESTARE (Grid Search)
    # ==========================================
    depths_mm = [None, 80, 50, 30] 
    windows = [
        (-100, 800),  # Larga
        (0, 500),     # Media 
        (50, 350)     # Stretta (Fa brillare il contrasto)
    ]

    for depth in depths_mm:
        # --- A. Taglio della profondità (Slab) ---
        if depth is None:
            slab_data = data
            depth_name = "FULL"
        else:
            half_vox = int((depth / 2) / vox_size[1])
            y_start = max(0, y_center - half_vox)
            y_end = min(data.shape[1], y_center + half_vox)
            slab_data = data[:, y_start:y_end, :]
            depth_name = f"{depth}mm"

        log.info(f"Calcolo MIP per profondità: {depth_name}...")
        mip_array = np.max(slab_data, axis=1)

        # --- B. Applicazione del Contrasto (Windowing) ---
        for w_min, w_max in windows:
            log.info(f"  -> Applico Windowing [{w_min}, {w_max}]")
            
            clipped = np.clip(mip_array, w_min, w_max)
            norm = ((clipped - w_min) / (w_max - w_min) * 255).astype(np.uint8)
            
            # Creazione e orientamento immagine
            img_array = np.flipud(norm.T)
            
            # Converti in RGB per disegnare le linee colorate
            img_2d = Image.fromarray(img_array).convert("RGB")
            
            # --- DISEGNO LINEE ---
            if draw_annotations:
                draw = ImageDraw.Draw(img_2d)
                img_width = img_2d.width
                
                # Linea ROSSA (Renale)
                draw.line([(0, y_renale), (img_width, y_renale)], fill="red", width=2)
                draw.text((10, max(0, y_renale - 15)), f"Ostio Renale ({z_renale_mm}mm -> Slice {idx_renale})", fill="red")

                # Linea VERDE (Colletto)
                draw.line([(0, y_colletto), (img_width, y_colletto)], fill="green", width=2)
                draw.text((10, max(0, y_colletto - 15)), f"Fine Colletto ({z_colletto_mm}mm -> Slice {idx_colletto})", fill="green")
            
            # Salvataggio
            filename = f"{base_name}_MIP_Depth-{depth_name}_Win-{w_min}to{w_max}.png"
            img_2d.save(os.path.join(output_dir, filename))
            
            if depth == 50 and w_max == 500:
                np.save(os.path.join(output_dir, f"{base_name}_MIP_Depth-{depth_name}.npy"), mip_array)

if __name__ == "__main__":
    # =========================================================================
    # PARAMETRI DA MODIFICARE PER OGNI TEST
    # =========================================================================
    
    PZ_ID = "212"  # Modifica questo per cambiare paziente (es: "001", "002", "027")

    # Inserisci i valori in MILLIMETRI presi dal foglio Excel (Mantieni i segni negativi se presenti!)
    # I valori qui sotto sono presi dalla riga 1 del tuo screenshot per il Paziente 1
    MM_RENALE = 388.3   
    MM_COLLETTO = 349.3

    # =========================================================================
    # PATH GENERATI AUTOMATICAMENTE
    # =========================================================================
    
    # Path della scansione mascherata "normale" (non CPR)
    INPUT_FILE = f"/home/group1/Desktop/Challange1/results/all_masked_normalized_scans/masked_scans_A/pz{PZ_ID}_A_masked.nii.gz"
    
    # Path delle segmentazioni (serve per trovare il centro dell'aorta)
    SEG_DIR = f"/home/group1/Desktop/Challange1/results/total_segmentator_type_A/patient_segmentations_{PZ_ID}_0CT_po1_A" 
    
    # Path di output specifico per questo paziente
    OUTPUT_FOLDER = f"/home/group1/Desktop/Challange1/results/mip_clinical_validation/pz_{PZ_ID}"

    # Esecuzione
    generate_arterial_mip_variants(
        INPUT_FILE, 
        SEG_DIR, 
        OUTPUT_FOLDER, 
        z_renale_mm=MM_RENALE, 
        z_colletto_mm=MM_COLLETTO
    )