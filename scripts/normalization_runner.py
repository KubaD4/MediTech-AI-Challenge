#!/usr/bin/env python3
"""
normalization_runner.py
=======================
Post-masking normalization for EVAR CT scans.

Normalization strategy
----------------------
Fat and bone are NOT usable in these masked scans:
  - subcutaneous_fat / torso_fat: not segmented in this TotalSegmentator run
  - vertebrae: segmented but voxels zeroed out during masking -> all MASK_VALUE

We use a SINGLE-ANCHOR shift with the SPLEEN as reference:
  - Spleen is physically stable in arterial-phase CT (~50-70 HU, population mean ~60 HU)
  - Not significantly affected by contrast timing (parenchymal enhancement is mild)
  - Present in STRUCTURES_TO_KEEP -> voxels are intact in the masked volume
  - Liver is used as fallback and cross-validation

Formula:
    HU_norm = HU - spleen_mean + SPLEEN_TARGET

This corrects per-scanner OFFSET (main source of inter-scanner variability).
We use a FIXED divisor (FIXED_SCALE) instead of per-patient std to avoid
destroying relative HU differences between structures (lumen vs thrombus).

Pipeline per patient:
  1. Shift   : HU_shifted = HU - spleen_mean + SPLEEN_TARGET
  2. (optional) Scale: HU_norm = HU_shifted / FIXED_SCALE
  3. Clip to [CLIP_MIN, CLIP_MAX] + scale to [0, 1] for ML

Distribution plots (after all patients):
  - Per-structure KDE: raw vs normalized, one curve per patient
  - Spread summary bar chart: inter-patient std of mean HU before vs after
  - Anchor variability: raw spleen means across patients (shows what we corrected)

Dependencies: nibabel, numpy, matplotlib, scipy
Usage:        python normalization_runner.py
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

try:
    import nibabel as nib
except ImportError:
    raise ImportError("nibabel is required. Install with: pip install nibabel")

# ---------------------------------------------------------------------------
# Configuration  (paths unchanged from original)
# ---------------------------------------------------------------------------

BASE_DIR   = os.path.expanduser("~/Desktop/Challange1")
MASKED_DIR = os.path.join(BASE_DIR, "results/masked_scans_V")
SEG_DIR    = os.path.join(BASE_DIR, "results/total_segmentator_type_V")
OUT_DIR    = os.path.join(BASE_DIR, "results/normalized_scans_V")
STATS_DIR  = os.path.join(BASE_DIR, "results/normalization_stats")
PLOT_DIR   = os.path.join(BASE_DIR, "results/normalization_plots")

MASK_VALUE = -1000  # background tag, same as masking_runner.py

# ---------------------------------------------------------------------------
# Anchor configuration
# ---------------------------------------------------------------------------

# Primary anchor: spleen
# Population mean in arterial-phase CT: ~60 HU (literature range 50-70 HU)
SPLEEN_LABEL  = "spleen.nii.gz"
SPLEEN_TARGET = 60.0

# Fallback / cross-check anchor: liver
# Population mean in arterial-phase CT: ~60-65 HU
LIVER_LABEL   = "liver.nii.gz"
LIVER_TARGET  = 62.0

# Optional gain correction using a fixed scale (disabled by default).
# Enable if inter-patient spread is still high after offset correction.
# FIXED_SCALE = population std of soft tissue in arterial CT (~80 HU).
# WARNING: enabling this changes the unit of the normalized volume.
APPLY_FIXED_SCALE = False
FIXED_SCALE       = 80.0

# ---------------------------------------------------------------------------
# Structures to monitor in plots
# All must be present in STRUCTURES_TO_KEEP from masking_runner.py
# ---------------------------------------------------------------------------

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

# Clip + scale to [0,1] for ML input.
# After spleen-anchor normalization: spleen lands at ~60 HU.
# Clip from -200 (covers fat/fluid) to +400 (covers contrast-enhanced vessels).
# Gold markers (>3000 HU) are intentionally excluded here -- use separate
# windowing on the raw volume for stent segmentation.
CLIP_MIN = -200.0
CLIP_MAX =  400.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("normalization_runner")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_nifti(path):
    img = nib.load(path)
    return np.asarray(img.dataobj, dtype=np.float32), img


def extract_structure_values(vol, seg_folder, label_file, max_samples=20000):
    """
    Return a random sub-sample of HU values inside a structure mask.
    Excludes voxels at MASK_VALUE (zeroed by masking_runner).
    Returns None if label file is missing or mask is empty.
    """
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
    log.info(f"Stats saved -> {path}")

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def compute_anchor(vol, seg_folder):
    """
    Returns (anchor_mean, anchor_target, anchor_name).
    Tries spleen first, falls back to liver.
    """
    spleen_vals = extract_structure_values(vol, seg_folder, SPLEEN_LABEL)
    if spleen_vals is not None and len(spleen_vals) > 50:
        return float(np.mean(spleen_vals)), SPLEEN_TARGET, "spleen"

    log.warning("Spleen not usable. Trying liver as fallback.")
    liver_vals = extract_structure_values(vol, seg_folder, LIVER_LABEL)
    if liver_vals is not None and len(liver_vals) > 50:
        return float(np.mean(liver_vals)), LIVER_TARGET, "liver"

    log.error("Neither spleen nor liver found. Normalization skipped.")
    return None, None, None


def single_anchor_normalize(vol, anchor_mean, anchor_target):
    """
    Shift the whole volume: HU_norm = HU - anchor_mean + anchor_target
    Optionally divide by FIXED_SCALE.
    Masked voxels are preserved.
    """
    was_masked = vol == MASK_VALUE
    vol_norm = vol - anchor_mean + anchor_target
    if APPLY_FIXED_SCALE:
        vol_norm = vol_norm / FIXED_SCALE
    vol_norm[was_masked] = MASK_VALUE
    return vol_norm


def scale_to_unit(vol_norm):
    """Clip to [CLIP_MIN, CLIP_MAX] and scale to [0, 1]. Background -> 0."""
    was_masked = vol_norm == MASK_VALUE
    clipped = np.clip(vol_norm, CLIP_MIN, CLIP_MAX)
    scaled  = (clipped - CLIP_MIN) / (CLIP_MAX - CLIP_MIN)
    scaled[was_masked] = 0.0
    return scaled

# ---------------------------------------------------------------------------
# Global accumulators for plots
# ---------------------------------------------------------------------------

_all_raw_samples  = {s: [] for s in MONITOR_STRUCTURES}
_all_norm_samples = {s: [] for s in MONITOR_STRUCTURES}
_anchor_log       = []   # {patient_id, anchor_name, anchor_mean} per patient

# ---------------------------------------------------------------------------
# Per-patient pipeline
# ---------------------------------------------------------------------------

def process_patient(patient_id):
    log.info("=" * 60)
    log.info(f"Normalizing patient: {patient_id}")
    log.info("=" * 60)

    # 1. CERCA IL FILE MASCHERATO (Aggiunta la _V_)
    masked_files = sorted(glob.glob(
        os.path.join(MASKED_DIR, f"{patient_id}_V_masked.nii.gz")
    ))
    if not masked_files:
        log.error(f"No masked file found for {patient_id} in {MASKED_DIR}. Skipping.")
        return
    ct_path = masked_files[0]

    # 2. CERCA LA CARTELLA SEGMENTAZIONE (Rimuove 'pz' dall'id)
    # pz001 -> 001
    num_id = "".join(filter(str.isdigit, patient_id))
    seg_folder = os.path.join(SEG_DIR, f"patient_segmentations_{num_id}_0CT_po1_V")
    
    if not os.path.isdir(seg_folder):
        log.error(f"Segmentation folder not found: {seg_folder}. Skipping.")
        return

    # Se arriva qui, i percorsi sono corretti!
    log.info(f"Loading masked CT: {ct_path}")
    vol_array, ct_img = load_nifti(ct_path)
    vol = vol_array.copy()

    stats = {"patient_id": patient_id}

    # Collect RAW samples (before any transform)
    for struct_name, label_file in MONITOR_STRUCTURES.items():
        samples = extract_structure_values(vol, seg_folder, label_file)
        if samples is not None:
            _all_raw_samples[struct_name].append(samples)

    # Compute anchor
    anchor_mean, anchor_target, anchor_name = compute_anchor(vol, seg_folder)

    if anchor_mean is None:
        log.error(f"[{patient_id}] No anchor found. Saving unmodified (clipped) volume.")
        vol_ml = scale_to_unit(vol)
    else:
        shift = anchor_target - anchor_mean
        log.info(f"Anchor: {anchor_name}  raw_mean={anchor_mean:.1f} HU  "
                 f"target={anchor_target:.1f} HU  shift={shift:+.1f} HU")

        stats["anchor_structure"] = anchor_name
        stats["anchor_mean_raw"]  = f"{anchor_mean:.2f}"
        stats["anchor_target"]    = f"{anchor_target:.2f}"
        stats["shift_applied"]    = f"{shift:.2f}"
        _anchor_log.append({"patient_id": patient_id,
                             "anchor_name": anchor_name,
                             "anchor_mean": anchor_mean})

        vol_norm = single_anchor_normalize(vol, anchor_mean, anchor_target)

        # Collect NORMALIZED samples for plots
        for struct_name, label_file in MONITOR_STRUCTURES.items():
            samples = extract_structure_values(vol_norm, seg_folder, label_file)
            if samples is not None:
                _all_norm_samples[struct_name].append(samples)

        # Cross-check: liver mean after normalization (should be close to LIVER_TARGET)
        liver_norm = extract_structure_values(vol_norm, seg_folder, LIVER_LABEL)
        if liver_norm is not None:
            stats["liver_mean_post_norm"] = f"{np.mean(liver_norm):.2f}"
            log.info(f"Cross-check liver post-norm mean: {np.mean(liver_norm):.1f} HU "
                     f"(expected ~{LIVER_TARGET:.0f})")

        vol_ml = scale_to_unit(vol_norm)

    tissue_mask = vol_ml != 0.0
    if tissue_mask.any():
        stats["post_norm_min"]  = f"{vol_ml[tissue_mask].min():.4f}"
        stats["post_norm_max"]  = f"{vol_ml[tissue_mask].max():.4f}"
        stats["post_norm_mean"] = f"{vol_ml[tissue_mask].mean():.4f}"
        stats["post_norm_std"]  = f"{vol_ml[tissue_mask].std():.4f}"

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{patient_id}_normalized.nii.gz")
    nib.save(nib.Nifti1Image(vol_ml, ct_img.affine, ct_img.header), out_path)
    log.info(f"Saved normalized volume -> {out_path}")
    save_stats(patient_id, stats)

# ---------------------------------------------------------------------------
# Distribution plots
# ---------------------------------------------------------------------------

def _kde_plot(ax, samples_list, color, x_range):
    for i, samples in enumerate(samples_list):
        if samples is None or len(samples) < 20:
            continue
        s = samples[np.isfinite(samples)]
        s = s[(s >= x_range[0]) & (s <= x_range[1])]
        if len(s) < 20:
            continue
        try:
            kde = gaussian_kde(s, bw_method=0.2)
            xs  = np.linspace(x_range[0], x_range[1], 400)
            ax.plot(xs, kde(xs), color=color, alpha=0.4, linewidth=1.3,
                    label="patient" if i == 0 else "_nolegend_")
        except Exception:
            pass


def plot_distributions(n_patients):
    os.makedirs(PLOT_DIR, exist_ok=True)
    XRANGE = (-200, 500)

    # Per-structure KDE plots
    for struct, color in STRUCT_COLORS.items():
        raw_list  = _all_raw_samples.get(struct, [])
        norm_list = _all_norm_samples.get(struct, [])
        if not raw_list:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        fig.suptitle(f"HU distribution  --  {struct}   (n={n_patients} patients)",
                     fontsize=13, fontweight="bold")

        ax = axes[0]
        _kde_plot(ax, raw_list, color, XRANGE)
        ax.set_title("Before normalization (raw HU)", fontsize=11)
        ax.set_xlabel("HU")
        ax.set_ylabel("Density")
        ax.set_xlim(XRANGE)
        ax.axvline(SPLEEN_TARGET, color="#ef4444", ls="--", lw=1.5,
                   label=f"spleen target = {SPLEEN_TARGET:.0f} HU")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        _kde_plot(ax, norm_list, color, XRANGE)
        ax.set_title("After normalization (spleen-anchored HU)", fontsize=11)
        ax.set_xlabel("HU (normalized)")
        ax.set_xlim(XRANGE)
        ax.axvline(SPLEEN_TARGET, color="#ef4444", ls="--", lw=1.5,
                   label=f"spleen target = {SPLEEN_TARGET:.0f} HU")
        ax.axvline(LIVER_TARGET, color="#f59e0b", ls="--", lw=1.5,
                   label=f"liver target = {LIVER_TARGET:.0f} HU")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(PLOT_DIR, f"distribution_{struct.lower()}.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        log.info(f"Plot saved -> {out}")

    # Spread summary
    structs    = list(MONITOR_STRUCTURES.keys())
    spread_raw = []
    spread_norm = []
    for struct in structs:
        mr = [float(np.mean(s)) for s in _all_raw_samples.get(struct, [])  if s is not None]
        mn = [float(np.mean(s)) for s in _all_norm_samples.get(struct, []) if s is not None]
        spread_raw.append(float(np.std(mr)) if len(mr) > 1 else 0.0)
        spread_norm.append(float(np.std(mn)) if len(mn) > 1 else 0.0)

    x = np.arange(len(structs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w/2, spread_raw,  w, label="Before", color="#ef4444", alpha=0.85)
    ax.bar(x + w/2, spread_norm, w, label="After",  color="#22c55e", alpha=0.85)
    ax.set_title("Inter-patient spread of mean HU per structure\n"
                 "(lower = better normalization)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Std of per-patient means (HU)")
    ax.set_xticks(x)
    ax.set_xticklabels(structs)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    for i, (raw, norm) in enumerate(zip(spread_raw, spread_norm)):
        if raw > 0:
            pct = (raw - norm) / raw * 100
            ax.text(i, max(raw, norm) * 1.04, f"{pct:+.0f}%",
                    ha="center", fontsize=9,
                    color="#22c55e" if pct > 0 else "#ef4444")
    plt.tight_layout()
    out = os.path.join(PLOT_DIR, "spread_summary.png")
    plt.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Spread summary saved -> {out}")

    # Anchor variability: raw spleen mean per patient
    spleen_entries = [e for e in _anchor_log if e["anchor_name"] == "spleen"]
    if len(spleen_entries) > 1:
        ids   = [e["patient_id"]  for e in spleen_entries]
        means = [e["anchor_mean"] for e in spleen_entries]
        fig, ax = plt.subplots(figsize=(max(8, len(ids) * 0.6), 4))
        ax.bar(ids, means, color="#ef4444", alpha=0.8)
        ax.axhline(SPLEEN_TARGET, color="#1e1e1e", ls="--", lw=1.5,
                   label=f"target = {SPLEEN_TARGET:.0f} HU")
        ax.set_title("Raw spleen mean HU per patient\n"
                     "(variability = what normalization corrects)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Raw spleen mean (HU)")
        ax.set_xlabel("Patient")
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        out = os.path.join(PLOT_DIR, "anchor_variability.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        log.info(f"Anchor variability saved -> {out}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    log.info("normalization_runner.py started")
    log.info(f"MASKED_DIR : {MASKED_DIR}")
    log.info(f"SEG_DIR    : {SEG_DIR}")
    log.info(f"OUT_DIR    : {OUT_DIR}")
    log.info(f"PLOT_DIR   : {PLOT_DIR}")

    if not os.path.isdir(MASKED_DIR):
        log.error(f"Masked scans directory does not exist: {MASKED_DIR}")
        sys.exit(1)

    masked_files = sorted(glob.glob(os.path.join(MASKED_DIR, "*_masked.nii.gz")))
    patients = [
        os.path.basename(f).replace("_V_masked.nii.gz", "")
        for f in masked_files
    ]

    if not patients:
        log.error("No masked files found.")
        sys.exit(1)

    log.info(f"Found {len(patients)} patient(s): {patients}")

    for patient_id in patients:
        try:
            process_patient(patient_id)
        except Exception:
            log.exception(f"Error processing {patient_id}")

    log.info("All patients processed. Generating distribution plots ...")
    plot_distributions(n_patients=len(patients))
    log.info("Done.")


if __name__ == "__main__":
    main()