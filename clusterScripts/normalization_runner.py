#!/usr/bin/env python3
"""
normalization_runner.py
=======================
Post-masking normalization for EVAR CT scans.
MULTIPROCESSING CLUSTER VERSION
"""

import os
import sys
import glob
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import concurrent.futures

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required. Install with: pip install nibabel")

# ---------------------------------------------------------------------------
# Configuration (Cluster Version)
# ---------------------------------------------------------------------------
WORK_DIR = os.getcwd()

MASKED_DIR = os.path.join(WORK_DIR, "masked_scans_V")
SEG_DIR    = os.path.join(WORK_DIR, "segmentatorResultsVfiles")
OUT_DIR    = os.path.join(WORK_DIR, "normalized_scans_V")
STATS_DIR  = os.path.join(WORK_DIR, "normalization_stats")
PLOT_DIR   = os.path.join(WORK_DIR, "normalization_plots")

MASK_VALUE = -1000  

# ---------------------------------------------------------------------------
# Anchor configuration
# ---------------------------------------------------------------------------
SPLEEN_LABEL  = "spleen.nii.gz"
SPLEEN_TARGET = 60.0
LIVER_LABEL   = "liver.nii.gz"
LIVER_TARGET  = 62.0

APPLY_FIXED_SCALE = False
FIXED_SCALE       = 80.0

MONITOR_STRUCTURES = {
    "Aorta":  "aorta.nii.gz",
    "IVC":    "inferior_vena_cava.nii.gz",
    "Spleen": "spleen.nii.gz",
    "Liver":  "liver.nii.gz",
    "Kidney": "kidney_left.nii.gz",
}

STRUCT_COLORS = {
    "Aorta":  "#4a9eed",
    "IVC":    "#06b6d4",
    "Spleen": "#ef4444",
    "Liver":  "#f59e0b",
    "Kidney": "#22c55e",
}

CLIP_MIN = -200.0
CLIP_MAX =  400.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("normalization_runner")

# ---------------------------------------------------------------------------
# Global accumulators for plots (Handled in main())
# ---------------------------------------------------------------------------
_all_raw_samples  = {s: [] for s in MONITOR_STRUCTURES}
_all_norm_samples = {s: [] for s in MONITOR_STRUCTURES}
_anchor_log       = []  

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_nifti(path):
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32), img

def extract_structure_values(vol, seg_folder, label_file, max_samples=20000):
    p = os.path.join(seg_folder, label_file)
    if not os.path.isfile(p):
        return None
    mask_arr, _ = load_nifti(p)
    valid = (mask_arr > 0) & (vol != MASK_VALUE)
    indices = np.where(valid)
    n = len(indices[0])
    if n == 0:
        return None
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        return vol[indices[0][idx], indices[1][idx], indices[2][idx]]
    return vol[indices]

def save_stats(patient_id, stats):
    os.makedirs(STATS_DIR, exist_ok=True)
    path = os.path.join(STATS_DIR, f"{patient_id}_norm_stats.txt")
    with open(path, "w") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")

def compute_anchor(vol, seg_folder):
    spleen_vals = extract_structure_values(vol, seg_folder, SPLEEN_LABEL)
    if spleen_vals is not None and len(spleen_vals) > 50:
        return float(np.mean(spleen_vals)), SPLEEN_TARGET, "spleen"

    liver_vals = extract_structure_values(vol, seg_folder, LIVER_LABEL)
    if liver_vals is not None and len(liver_vals) > 50:
        return float(np.mean(liver_vals)), LIVER_TARGET, "liver"
    return None, None, None

def single_anchor_normalize(vol, anchor_mean, anchor_target):
    was_masked = vol == MASK_VALUE
    vol_norm = vol - anchor_mean + anchor_target
    if APPLY_FIXED_SCALE:
        vol_norm = vol_norm / FIXED_SCALE
    vol_norm[was_masked] = MASK_VALUE
    return vol_norm

def scale_to_unit(vol_norm):
    was_masked = vol_norm == MASK_VALUE
    clipped = np.clip(vol_norm, CLIP_MIN, CLIP_MAX)
    scaled  = (clipped - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)
    scaled[was_masked] = 0.0
    return scaled

# ---------------------------------------------------------------------------
# Per-patient pipeline
# ---------------------------------------------------------------------------
def process_patient(patient_id):
    """Ritorna un dizionario con i dati per i plot, o None se saltato."""
    
    # ---> RESUME CHECK <---
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{patient_id}_normalized.nii.gz")
    if os.path.exists(out_path):
        log.info(f"[{patient_id}] Normalized file already exists. Skipping!")
        return None
    # ----------------------

    log.info(f"Normalizing patient: {patient_id}")

    masked_files = sorted(glob.glob(os.path.join(MASKED_DIR, f"{patient_id}_V_masked.nii.gz")))
    if not masked_files:
        log.error(f"No masked file found for {patient_id}. Skipping.")
        return None
    ct_path = masked_files[0]

    num_id = "".join(filter(str.isdigit, patient_id))
    seg_folder = os.path.join(SEG_DIR, f"patient_segmentations_{num_id}_0CT_po1_V")
    
    if not os.path.isdir(seg_folder):
        log.error(f"Segmentation folder not found: {seg_folder}. Skipping.")
        return None

    vol_array, ct_img = load_nifti(ct_path)
    vol = vol_array.copy()
    stats = {"patient_id": patient_id}
    
    # Dizionari locali per inviare i dati dei plot al processo principale
    local_raw_samples = {}
    local_norm_samples = {}
    local_anchor = None

    for struct_name, label_file in MONITOR_STRUCTURES.items():
        samples = extract_structure_values(vol, seg_folder, label_file)
        if samples is not None:
            local_raw_samples[struct_name] = samples

    anchor_mean, anchor_target, anchor_name = compute_anchor(vol, seg_folder)

    if anchor_mean is None:
        log.error(f"[{patient_id}] No anchor found. Saving unmodified (clipped) volume.")
        vol_ml = scale_to_unit(vol)
    else:
        shift = anchor_target - anchor_mean
        stats["anchor_structure"] = anchor_name
        stats["anchor_mean_raw"]  = f"{anchor_mean:.2f}"
        stats["anchor_target"]    = f"{anchor_target:.2f}"
        stats["shift_applied"]    = f"{shift:.2f}"
        
        local_anchor = {"patient_id": patient_id, "anchor_name": anchor_name, "anchor_mean": anchor_mean}
        vol_norm = single_anchor_normalize(vol, anchor_mean, anchor_target)

        for struct_name, label_file in MONITOR_STRUCTURES.items():
            samples = extract_structure_values(vol_norm, seg_folder, label_file)
            if samples is not None:
                local_norm_samples[struct_name] = samples

        liver_norm = extract_structure_values(vol_norm, seg_folder, LIVER_LABEL)
        if liver_norm is not None:
            stats["liver_mean_post_norm"] = f"{np.mean(liver_norm):.2f}"

        vol_ml = scale_to_unit(vol_norm)

    tissue_mask = vol_ml != 0.0
    if tissue_mask.any():
        stats["post_norm_min"]  = f"{vol_ml[tissue_mask].min():.4f}"
        stats["post_norm_max"]  = f"{vol_ml[tissue_mask].max():.4f}"
        stats["post_norm_mean"] = f"{vol_ml[tissue_mask].mean():.4f}"
        stats["post_norm_std"]  = f"{vol_ml[tissue_mask].std():.4f}"

    nib.save(nib.Nifti1Image(vol_ml, ct_img.affine, ct_img.header), out_path)
    save_stats(patient_id, stats)
    
    # Ritorna i dati in modo che il Multiprocessing possa raccoglierli
    return {"raw": local_raw_samples, "norm": local_norm_samples, "anchor": local_anchor}

# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------
def _kde_plot(ax, samples_list, color, x_range):
    for i, samples in enumerate(samples_list):
        if samples is None or len(samples) < 20: continue
        s = samples[np.isfinite(samples)]
        s = s[(s >= x_range[0]) & (s <= x_range[1])]
        if len(s) < 20: continue
        try:
            kde = gaussian_kde(s, bw_method=0.2)
            xs  = np.linspace(x_range[0], x_range[1], 400)
            ax.plot(xs, kde(xs), color=color, alpha=0.4, linewidth=1.3,
                    label="patient" if i == 0 else "_nolegend_")
        except Exception: pass

def plot_distributions(n_patients):
    os.makedirs(PLOT_DIR, exist_ok=True)
    XRANGE = (-200, 500)

    for struct, color in STRUCT_COLORS.items():
        raw_list  = _all_raw_samples.get(struct, [])
        norm_list = _all_norm_samples.get(struct, [])
        if not raw_list: continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f"HU distribution  --  {struct}   (n={n_patients} patients)", fontsize=13, fontweight="bold")

        ax = axes[0]
        _kde_plot(ax, raw_list, color, XRANGE)
        ax.set_title("Before normalization (raw HU)", fontsize=11)
        ax.set_xlabel("HU")
        ax.set_xlim(XRANGE)
        ax.axvline(SPLEEN_TARGET, color="#ef4444", ls="--", lw=1.5)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _kde_plot(ax, norm_list, color, XRANGE)
        ax.set_title("After normalization (spleen-anchored HU)", fontsize=11)
        ax.set_xlabel("HU (normalized)")
        ax.set_xlim(XRANGE)
        ax.axvline(SPLEEN_TARGET, color="#ef4444", ls="--", lw=1.5)
        ax.axvline(LIVER_TARGET, color="#f59e0b", ls="--", lw=1.5)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"distribution_{struct.lower()}.png"), dpi=150)
        plt.close(fig)

    structs    = list(MONITOR_STRUCTURES.keys())
    spread_raw = [float(np.std([np.mean(s) for s in _all_raw_samples.get(st, []) if s is not None])) if len(_all_raw_samples.get(st, [])) > 1 else 0.0 for st in structs]
    spread_norm = [float(np.std([np.mean(s) for s in _all_norm_samples.get(st, []) if s is not None])) if len(_all_norm_samples.get(st, [])) > 1 else 0.0 for st in structs]

    x = np.arange(len(structs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, spread_raw,  w, label="Before", color="#ef4444", alpha=0.85)
    ax.bar(x + w/2, spread_norm, w, label="After",  color="#22c55e", alpha=0.85)
    ax.set_title("Inter-patient spread of mean HU per structure", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(structs)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "spread_summary.png"), dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    if not os.path.isdir(MASKED_DIR):
        log.error(f"Masked scans directory does not exist: {MASKED_DIR}")
        sys.exit(1)

    masked_files = sorted(glob.glob(os.path.join(MASKED_DIR, "*_masked.nii.gz")))
    patients = [os.path.basename(f).replace("_V_masked.nii.gz", "") for f in masked_files]

    if not patients:
        log.error("No masked files found.")
        sys.exit(1)

    log.info(f"Found {len(patients)} patient(s). Avvio Multiprocessing (4 CPU)...")

    # The Multiprocessing Block
    processed_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_patient, p): p for p in patients}
        
        for future in concurrent.futures.as_completed(futures):
            p = futures[future]
            try:
                result = future.result()
                if result is not None:
                    # Gather the plotting data from the CPU core that just finished
                    processed_count += 1
                    for struct, samples in result["raw"].items():
                        _all_raw_samples[struct].append(samples)
                    for struct, samples in result["norm"].items():
                        _all_norm_samples[struct].append(samples)
                    if result["anchor"]:
                        _anchor_log.append(result["anchor"])
            except Exception as e:
                log.error(f"Errore critico in {p}: {e}")

    log.info("Tutti i pazienti completati! Generazione dei plot...")
    
    # Only generate plots if we actually processed someone new
    if processed_count > 0:
        plot_distributions(n_patients=processed_count)
    else:
        log.info("Nessun nuovo paziente processato. Plot saltati.")
        
    log.info("Fatto.")

if __name__ == "__main__":
    main()
