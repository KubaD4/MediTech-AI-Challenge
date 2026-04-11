import os
import nibabel as nib
import numpy as np
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger("Arterial_MIP_Tuner")

def generate_arterial_mip_variants(input_path, output_dir):
    if not os.path.exists(input_path):
        log.error(f"File di input non trovato: {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(input_path).split('.')[0]

    log.info("Caricamento volume arterioso mascherato...")
    img = nib.load(input_path)
    data = img.get_fdata()

    log.info("Calcolo MIP sull'intera profondità (FULL)...")
    mip_array = np.max(data, axis=1)

    # ==========================================
    # NUOVI PARAMETRI DI TUNING (Fine-grained)
    # ==========================================
    windows = [
        # 1. Evoluzione di 0-500 (Più scuri/contrastati)
        (30, 450),    
        (50, 600),
        
        # 2. Evoluzione di -100-2000 (Più visibilità vasi piccoli)
        (-50, 1000),
        (0, 800),
        
        # 3. Range specifici per fase arteriosa (Focus renali)
        (80, 400),
        (100, 500)
    ]

    for w_min, w_max in windows:
        log.info(f"  -> Generazione variante Windowing [{w_min}, {w_max}]")
        
        clipped = np.clip(mip_array, w_min, w_max)
        # Normalizzazione
        norm = ((clipped - w_min) / (w_max - w_min) * 255).astype(np.uint8)
        
        # Conversione in immagine (flip per correggere orientamento Slicer->Image)
        img_2d = Image.fromarray(np.flipud(norm.T)) 
        

        filename = f"{base_name}_MIP_FULL_Win-{w_min}to{w_max}.png"
        img_2d.save(os.path.join(output_dir, filename))
        
        # Salviamo NPY per il vincitore probabile (bilanciato)
        if w_min == 50 and w_max == 600:
            np.save(os.path.join(output_dir, f"{base_name}_MIP_FULL_FinalCandidate.npy"), mip_array)

if __name__ == "__main__": 
    PZ_ID = "011"
    
    # Assicurati che il percorso punti al file generato dal NUOVO aorta_masking_runner.py (quello con le iliache)
    # INPUT_FILE = f"/home/group1/Desktop/Challange1/results/masked_scans_A_aorta_only/pz{PZ_ID}_A_aorta_masked.nii.gz"
    INPUT_FILE = f"/home/group1/Desktop/Challange1/results/Result_CPR/{PZ_ID}/straightened_right.nii.gz"
    OUTPUT_FOLDER = f"/home/group1/Desktop/Challange1/results/mip_aortaOnly_arterial_tuning/patient_{PZ_ID}"

    generate_arterial_mip_variants(INPUT_FILE, OUTPUT_FOLDER)
    log.info(f"Tuning completato. Controlla le immagini in: {OUTPUT_FOLDER}")