import os
import nibabel as nib
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger("Arterial_MIP_Tuner")

def generate_arterial_mip_variants(input_path, seg_dir, output_dir):
    if not os.path.exists(input_path):
        log.error(f"File di input non trovato: {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path).split('.')[0]

    log.info("Caricamento volume arterioso...")
    img = nib.load(input_path)
    data = img.get_fdata()
    vox_size = img.header.get_zooms() # (dx, dy, dz)

    # 1. Trovare il baricentro Y (Antero-Posteriore) dell'aorta
    # Ci serve per centrare la nostra "fetta" di profondità
    aorta_path = os.path.join(seg_dir, "aorta.nii.gz")
    if os.path.exists(aorta_path):
        aorta_mask = nib.load(aorta_path).get_fdata().astype(bool)
        y_coords = np.where(aorta_mask)[1]
        y_center = int(np.median(y_coords))
        log.info(f"Centro Y dell'aorta trovato all'indice: {y_center}")
    else:
        log.warning("Maschera aorta non trovata! Uso il centro esatto del volume.")
        y_center = data.shape[1] // 2

    # ==========================================
    # PARAMETRI DA TESTARE (Grid Search)
    # ==========================================
    
    # 1. Spessore della fetta in millimetri (Y-axis depth)
    # None = Tutto il volume (come hai fatto tu)
    # 40 = Solo 2 cm davanti e 2 cm dietro l'aorta (taglia via colonna e sterno)
    depths_mm = [None, 80, 50, 30] 
    
    # 2. Windowing (Min HU, Max HU)
    # I vasi in fase arteriosa sono di solito tra 200 e 450 HU
    windows = [
        (-100, 800),  # Larga (vede bene i tessuti ma poco contrasto sui vasi)
        (0, 500),     # Media (Bilanciata)
        (50, 350)     # Stretta (Fa "brillare" il mezzo di contrasto)
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
            
            # Prendiamo tutto in X e Z, ma limitiamo la Y
            slab_data = data[:, y_start:y_end, :]
            depth_name = f"{depth}mm"

        log.info(f"Calcolo MIP per profondità: {depth_name}...")
        mip_array = np.max(slab_data, axis=1)

        # --- B. Applicazione del Contrasto (Windowing) ---
        for w_min, w_max in windows:
            log.info(f"  -> Applico Windowing [{w_min}, {w_max}]")
            
            clipped = np.clip(mip_array, w_min, w_max)
            norm = ((clipped - w_min) / (w_max - w_min) * 255).astype(np.uint8)
            
            img_2d = Image.fromarray(np.flipud(norm.T)) 
            
            filename = f"{base_name}_MIP_Depth-{depth_name}_Win-{w_min}to{w_max}.png"
            img_2d.save(os.path.join(output_dir, filename))
            
            # Salviamo il file NPY solo per le configurazioni più promettenti per risparmiare spazio
            if depth == 50 and w_max == 500:
                np.save(os.path.join(output_dir, f"{base_name}_MIP_Depth-{depth_name}.npy"), mip_array)

if __name__ == "__main__":
    # Inserisci i percorsi corretti per il server
   
    # Test su ortogonalizzata
    #INPUT_FILE = "/home/group1/Desktop/Challange1/results/all_masked_normalized_scans/masked_scans_A/pz057_A_masked.nii.gz"
    INPUT_FILE = "/home/group1/Desktop/Challange1/results/Result_CPR/001/straightened.nii.gz"
    # Abbiamo bisogno della directory delle segmentazioni per trovare il centro dell'aorta!
    SEG_DIR = "/home/group1/Desktop/Challange1/results/total_segmentator_type_A/patient_segmentations_001_0CT_po1_A" # <- Assicurati che sia la cartella corretta per la fase A!
    OUTPUT_FOLDER = "/home/group1/Desktop/Challange1/results/mip_arterial_CPR_tuning"

    generate_arterial_mip_variants(INPUT_FILE, SEG_DIR, OUTPUT_FOLDER)