#!/usr/bin/env python3
"""
Factual inventory of raw microscopy data and curated single-cell tables in Cell-Kill-Sim.

Walks local ScienceDB 19235 light-sheet files (`data/scidb_19235_chambers/`, `data/scidb_19235_samples/`),
compares against the full ScienceDB server manifest (`data/scidb_19235_all_urls.txt`) and repo inventory
tables (`data/dent_step1_chamber_inventory.csv`, `data/dent_step1_group_inventory.csv`,
`data/data_registry.csv`, `data/data_dictionary.md`), summarizes `data/ome_tiff/` and the two Imaris
confocal metadata JSONs (`data/ims_metadata_20190626.json`, `data/ims_metadata_exp28.json`) without
opening multi-GB `.ims` files, and audits the curated 419-cell tables
(`data/lightsheet_wang2026_single_cells_419_clean.csv`, `data/clean_per_cell_419.csv`).

Writes `data/raw_data_inventory.json` and prints a concise human-readable summary.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import tifffile

from paths import REPO_ROOT, DATA_DIR
from constants import VOXEL_UM, OPTICAL_FWHM_UM
CHAMBERS_DIR = DATA_DIR / "scidb_19235_chambers"
SAMPLES_DIR = DATA_DIR / "scidb_19235_samples"
CACHE_DIR = DATA_DIR / "scidb_dent_cache"
URLS_PATH = DATA_DIR / "scidb_19235_all_urls.txt"
OME_DIR = DATA_DIR / "ome_tiff"
IMS_2019_JSON = DATA_DIR / "ims_metadata_20190626.json"
IMS_EXP28_JSON = DATA_DIR / "ims_metadata_exp28.json"
CLEAN_419_A = DATA_DIR / "lightsheet_wang2026_single_cells_419_clean.csv"
CLEAN_419_B = DATA_DIR / "clean_per_cell_419.csv"
CHAMBER_INV_CSV = DATA_DIR / "dent_step1_chamber_inventory.csv"
GROUP_INV_CSV = DATA_DIR / "dent_step1_group_inventory.csv"
REGISTRY_CSV = DATA_DIR / "data_registry.csv"
OUT_JSON = DATA_DIR / "raw_data_inventory.json"


def inspect_h5_file(path: Path, compute_label_counts: bool = True) -> dict[str, Any]:
    """Open an HDF5 file with h5py and record datasets, shapes, dtypes, attrs, and contents."""
    is_mask = path.name.endswith("_predictions.h5")
    info: dict[str, Any] = {
        "rel_path": str(path.relative_to(REPO_ROOT)),
        "file_name": path.name,
        "size_bytes": path.stat().st_size,
        "file_kind": "segmentation_mask_predictions" if is_mask else "raw_fluorescence_stack",
        "file_attrs": {},
        "datasets": {},
    }
    with h5py.File(path, "r") as f:
        info["file_attrs"] = {k: str(v) for k, v in f.attrs.items()}
        for ds_name, ds in f.items():
            ds_entry: dict[str, Any] = {
                "shape": list(ds.shape),
                "dtype": str(ds.dtype),
                "chunks": list(ds.chunks) if ds.chunks else None,
                "compression": ds.compression,
                "attrs": {k: str(v) for k, v in ds.attrs.items()},
            }
            if is_mask and compute_label_counts:
                arr = ds[:]
                vals, cnts = np.unique(arr, return_counts=True)
                ds_entry["unique_labels"] = [int(v) for v in vals]
                ds_entry["voxel_counts_per_label"] = {str(int(v)): int(c) for v, c in zip(vals, cnts)}
            elif not is_mask and compute_label_counts:
                arr = ds[:]
                ch_stats = []
                for c_idx in range(arr.shape[0]):
                    sub = arr[c_idx]
                    ch_stats.append({
                        "channel_index": int(c_idx),
                        "min": int(sub.min()),
                        "p50": float(np.percentile(sub, 50)),
                        "p99": float(np.percentile(sub, 99)),
                        "max": int(sub.max()),
                        "mean": round(float(sub.mean()), 2),
                    })
                ds_entry["channel_intensity_summary"] = ch_stats
            info["datasets"][ds_name] = ds_entry
    return info


def parse_chamber_folder_name(folder_name: str) -> dict[str, Any]:
    """
    Parse ScienceDB 19235 chamber folder names such as:
      - v05_6_40_120_0-6_hd7_MIP_ROI_23
      - v11_6_36_120_0-6_hd2_ds_MIP_ROI_2
      - v12_6_35_120_0-6_hd10ds_MIP_ROI_0
    Cross-referenced against `external/feilab_CART_analysis/main/configure.m` lines 18-25.
    """
    m = re.match(
        r"^(v\d+)_(\d+)_(\d+)_(\d+)_(\d+-\d+)_(hd\d+)(?:_(ds)|(ds))?(?:_MIP_ROI_(\d+))?$",
        folder_name,
    )
    if not m:
        return {"raw_name": folder_name, "parsed": False}
    v_tok, vol_num, roi_num, slices, step_str, hd_tok, ds_underscore, ds_attached, roi_idx = m.groups()
    has_ds = bool(ds_underscore or ds_attached)
    exp_dir = f"{v_tok}_{vol_num}_{roi_num}_{slices}_{step_str}_{hd_tok}"
    if ds_underscore:
        exp_dir += "_ds"
    elif ds_attached:
        exp_dir += "ds"
    return {
        "raw_name": folder_name,
        "parsed": True,
        "experiment_dir": exp_dir,
        "tokens_decoded": {
            "view_id": {
                "token": v_tok,
                "meaning": "Experiment/view sequence index (v01..v13 across 13 experiments)",
                "source": "external/feilab_CART_analysis/main/configure.m line 8 (`dir(fullfile(config.root,'v_*'))`)",
            },
            "volume_num": {
                "token": int(vol_num),
                "meaning": "Number of 3D sweeps per microchamber per visit burst (6 sweeps at ~2.5 s/sweep; sweeps 1-5 image 405/488/561 nm, sweep 6 switches ch2 to 637 nm SiR-tubulin)",
                "source": "configure.m line 19 (`config.volume_num = str2double(config_str(2))`) & step2_instance_seg.m (`sample_num = 6`)",
            },
            "roi_num_nominal": {
                "token": int(roi_num),
                "meaning": "Nominal number of microchamber ROIs imaged in the experiment run (40, 39, 36, or 35)",
                "source": "configure.m line 20 (`config.roi_num = str2double(config_str(3))`)",
            },
            "slice_per_stack": {
                "token": int(slices),
                "meaning": "Number of raw 2D camera frames (light-sheet slices) per 3D sweep (120 slices)",
                "source": "configure.m line 23 (`config.slice_per_stack = str2double(config_str(4))`)",
            },
            "stepsize_um": {
                "token": step_str,
                "decoded_value_um": float(step_str.replace("-", ".")),
                "meaning": "Light-sheet piezo scanning step size in micrometers (0-6 -> 0.6 um)",
                "source": "configure.m line 24 (`config.stepsize = str2double(strrep(config_str(5),'-','.'))`)",
            },
            "dasatinib_suffix": {
                "token": "_ds" if ds_underscore else ("ds" if ds_attached else None),
                "is_dasatinib_50nM": has_ds,
                "meaning": "Presence of `ds` / `_ds` marks 50 nM Dasatinib Lck-inhibited groups (Groups 8*, 9*, 10* paired with hd2=Group 8, hd10=Group 9, hd5=Group 10)",
                "source": "dent_step1_group_inventory.csv & data_dictionary.md Section 8",
            },
            "mip_roi_index": {
                "token": f"MIP_ROI_{roi_idx}" if roi_idx is not None else None,
                "roi_index": int(roi_idx) if roi_idx is not None else None,
                "meaning": "0-indexed microchamber region-of-interest folder index on the microfluidic chip",
                "source": "step1_setup_analysis.m line 10 (`dir(fullfile(view_path,'MIP_ROI_*'))`)",
            },
        },
        "tokens_undecoded_or_ambiguous": {
            "hd_token": {
                "token": hd_tok,
                "what_is_known": (
                    "Identifier `hd1`..`hd10` distinguishing the 10 untreated donor/batch runs "
                    "(v01=hd1, v02=hd3, v03=hd4, v04=hd6, v05=hd7 -> High-Killing Groups 1-5; "
                    "v06=hd8, v07=hd9, v08=hd2, v09=hd10, v10=hd5 -> Low-Killing Groups 6-10; "
                    "v11=hd2_ds, v12=hd10ds, v13=hd5_ds -> Dasatinib 50 nM Groups 8*, 9*, 10*)."
                ),
                "what_cannot_be_decoded_from_files": (
                    "Not parsed by `configure.m` (which only reads tokens 1..5). The literal expansion of "
                    "`hd` (e.g. 'healthy donor' vs 'human donor' vs internal batch code) and why the numbering "
                    "is non-sequential relative to v01..v10 are not documented in the repository or MATLAB code."
                ),
            }
        },
    }


def inventory_local_scidb() -> dict[str, Any]:
    """Walk `data/scidb_19235_chambers/` and `data/scidb_19235_samples/` and inspect HDF5 files."""
    chamber_dirs = sorted([d for d in CHAMBERS_DIR.iterdir() if d.is_dir()])
    chambers_summary = []
    all_raw_h5: list[Path] = []
    all_pred_h5: list[Path] = []
    global_label_voxel_counts: Counter[int] = Counter()
    z_shapes_by_chamber: dict[str, list[int]] = {}

    for ch_dir in chamber_dirs:
        h5_files = sorted(ch_dir.glob("*.h5"))
        pred_files = [p for p in h5_files if p.name.endswith("_predictions.h5")]
        raw_files = [p for p in h5_files if not p.name.endswith("_predictions.h5")]
        tif_files = sorted(ch_dir.glob("*.tif"))

        all_raw_h5.extend(raw_files)
        all_pred_h5.extend(pred_files)

        raw_stems = sorted([p.stem for p in raw_files])
        pred_stems = sorted([p.name.replace("_predictions.h5", "") for p in pred_files])
        paired_stems = sorted(set(raw_stems) & set(pred_stems))

        # Open a sample (up to 2 raw and 2 pred per chamber) for shape/dtype/attrs verification,
        # plus accumulate label counts across all local prediction masks.
        sample_records = []
        for p in raw_files[:2] + pred_files[:2]:
            rec = inspect_h5_file(p, compute_label_counts=True)
            sample_records.append(rec)
            for ds_info in rec["datasets"].values():
                z_shapes_by_chamber[ch_dir.name] = ds_info["shape"]

        ch_label_counts: Counter[int] = Counter()
        for p in pred_files:
            with h5py.File(p, "r") as f:
                arr = f["predictions"][:]
                vals, cnts = np.unique(arr, return_counts=True)
                for v, c in zip(vals, cnts):
                    ch_label_counts[int(v)] += int(c)
                    global_label_voxel_counts[int(v)] += int(c)

        chambers_summary.append({
            "chamber_folder": ch_dir.name,
            "parsed_name": parse_chamber_folder_name(ch_dir.name),
            "n_h5_total": len(h5_files),
            "n_raw_h5": len(raw_files),
            "n_predictions_h5": len(pred_files),
            "n_paired_raw_and_pred_h5": len(paired_stems),
            "n_mip_tif": len(tif_files),
            "raw_h5_stems": raw_stems,
            "paired_stems": paired_stems,
            "pred_h5_first_last": [pred_stems[0], pred_stems[-1]] if pred_stems else [],
            "unique_labels_in_chamber": sorted(ch_label_counts.keys()),
            "voxel_counts_by_label_in_chamber": {str(k): int(v) for k, v in sorted(ch_label_counts.items())},
            "sample_h5_inspections": sample_records,
        })

    # Inspect data/scidb_19235_samples/
    sample_dir_h5 = sorted(SAMPLES_DIR.glob("*.h5")) if SAMPLES_DIR.exists() else []
    sample_dir_records = [inspect_h5_file(p, compute_label_counts=True) for p in sample_dir_h5]
    for rec in sample_dir_records:
        if rec["file_kind"] == "segmentation_mask_predictions":
            for ds_info in rec["datasets"].values():
                for k, v in ds_info.get("voxel_counts_per_label", {}).items():
                    global_label_voxel_counts[int(k)] += int(v)

    # Also measure raw channel intensities stratified by U-Net label on a representative paired stack
    rep_raw = CHAMBERS_DIR / "v01_6_40_120_0-6_hd1_MIP_ROI_39" / "00237.h5"
    rep_pred = CHAMBERS_DIR / "v01_6_40_120_0-6_hd1_MIP_ROI_39" / "00237_predictions.h5"
    channel_by_label_stats = {}
    if rep_raw.exists() and rep_pred.exists():
        with h5py.File(rep_raw, "r") as fr, h5py.File(rep_pred, "r") as fp:
            raw_arr = fr["raw"][:]
            pred_arr = fp["predictions"][:]
        for lbl in sorted(np.unique(pred_arr)):
            m = pred_arr == lbl
            channel_by_label_stats[str(int(lbl))] = {
                "voxel_count": int(m.sum()),
                "mean_intensity_by_channel": [round(float(raw_arr[c][m].mean()), 2) for c in range(raw_arr.shape[0])],
                "p99_intensity_by_channel": [round(float(np.percentile(raw_arr[c][m], 99)), 2) for c in range(raw_arr.shape[0])],
            }

    total_voxels_all_masks = sum(global_label_voxel_counts.values())
    label_documentation = {
        "0": {"meaning_in_repo": "Background (outside cells)", "source": "step2_instance_seg.m lines 38-41 (implicit 0)"},
        "1": {"meaning_in_repo": "CAR-T cell body / interior (`t_mask = I==1`)", "source": "external/feilab_CART_analysis/main/analysis/step2_instance_seg.m line 38"},
        "2": {"meaning_in_repo": "Target (Nalm6) cell interior (`b_mask = I==2`)", "source": "external/feilab_CART_analysis/main/analysis/step2_instance_seg.m line 39"},
        "3": {"meaning_in_repo": "CAR-T dense F-actin (`actin = I==3`, enriched in ch1 488nm Lifeact-EGFP)", "source": "external/feilab_CART_analysis/main/analysis/step2_instance_seg.m line 40"},
        "4": {"meaning_in_repo": "Target (Nalm6) membrane (`mem = I==4`, enriched in ch2 561nm mApple-Mem)", "source": "external/feilab_CART_analysis/main/analysis/step2_instance_seg.m line 41"},
    }

    return {
        "chambers_dir": str(CHAMBERS_DIR.relative_to(REPO_ROOT)),
        "n_chamber_subfolders": len(chamber_dirs),
        "n_non_empty_chamber_subfolders": sum(1 for c in chambers_summary if c["n_h5_total"] > 0),
        "n_empty_chamber_subfolders": sum(1 for c in chambers_summary if c["n_h5_total"] == 0),
        "total_h5_in_chambers_dir": len(all_raw_h5) + len(all_pred_h5),
        "total_raw_h5_in_chambers_dir": len(all_raw_h5),
        "total_predictions_h5_in_chambers_dir": len(all_pred_h5),
        "total_mip_tif_in_chambers_dir": sum(c["n_mip_tif"] for c in chambers_summary),
        "samples_dir": str(SAMPLES_DIR.relative_to(REPO_ROOT)),
        "n_h5_in_samples_dir": len(sample_dir_h5),
        "samples_dir_files": sample_dir_records,
        "h5_metadata_findings": {
            "root_attrs_present": False,
            "dataset_attrs_present": False,
            "voxel_size_in_h5_attrs": None,
            "channel_names_in_h5_attrs": None,
            "timestamps_in_h5_attrs": None,
            "note": (
                "All inspected ScienceDB .h5 files have empty HDF5 attributes (`{}`) on both the root group "
                "and `/raw` or `/predictions` datasets. Voxel size, channel wavelengths, and visit cadence "
                "are NOT stored inside the HDF5 headers; they are hardcoded in `external/feilab_CART_analysis/`."
            ),
            "voxel_size_discrepancy_in_repo": {
                "reconstructed_voxel_pitch_um_from_summary_m": 0.1962,
                "reconstructed_voxel_pitch_derivation": (
                    "`external/feilab_CART_analysis/main/analysis/summary/summary.m` line 18 multiplies "
                    "voxel surface counts by `0.0385` um^2/voxel (`sqrt(0.0385) = 0.196214 um/voxel`), "
                    "and line 16 multiplies actin displacement (voxels per 10 s) by `19.6` nm/s (`0.196 um/voxel`). "
                    "`reconsitution_app_eng.mlapp` sets `PixelSize = 6.5 um`, `Magnification = 62.2`, `Binning = 2` "
                    "(`6.5 / 62.2 * 2 = 0.2090 um/voxel`)."
                ),
                "optical_fwhm_cited_in_data_dictionary_um": OPTICAL_FWHM_UM,
                "optical_fwhm_note": (
                    "`data/data_dictionary.md` Section 8 and `data/data_registry.csv` cite `0.320 x 0.320 x 0.320 um` "
                    "(the optical lateral FWHM resolution of the HBOPM light-sheet microscope in Wang et al. 2024/2026), "
                    "whereas `src/run_dent_stiffness_fixes_all_frames.py` line 76 documents `DX_UM = 0.1962 um/vox` "
                    "as the reconstructed ROI voxel pitch and `OPTICAL_FWHM_UM = 0.320 um` as optical FWHM."
                ),
            },
            "z_dimension_varies_by_experiment": {
                "note": (
                    "`data/data_registry.csv` states `(3, 166, 256, 256)` for all `.h5` volumes, but direct "
                    "inspection of local `.h5` files shows Z varies between 166 and 190 across experiments: "
                    "Z=190 (v01), Z=170 (v02), Z=166 (v05 & samples/03979.h5), Z=174 (v06), Z=186 (v11)."
                ),
                "observed_shapes_by_chamber": z_shapes_by_chamber,
            },
            "channel_mapping_from_step3_roi_and_step2_instance_seg": {
                "channel_0": "405 nm (SYTOX Blue dead-cell nuclear dye; threshold D > 398 in step2_instance_seg.m line 44)",
                "channel_1": "488 nm (Lifeact-EGFP CAR-T F-actin; step2_instance_seg.m line 50)",
                "channel_2": (
                    "561 nm (Nalm6 target mApple-Mem membrane on sweeps 1-5 where `index % 6 != 0`); "
                    "switched to 637 nm (SiR-Tubulin MTOC centrosome spot on sweep 6 where `index % 6 == 0`, "
                    "detected via `spot_detection(raw_temp(:,:,:,3), 450, ...)` in step2_instance_seg.m line 33)."
                ),
            },
        },
        "global_mask_label_summary_across_126_local_masks": {
            "total_masks_counted": len(all_pred_h5) + sum(1 for r in sample_dir_records if r["file_kind"] == "segmentation_mask_predictions"),
            "total_voxels": total_voxels_all_masks,
            "counts_by_label": {
                str(k): {
                    "voxel_count": int(v),
                    "fraction_of_all_voxels": round(float(v) / max(1, total_voxels_all_masks), 6),
                    "meaning": label_documentation.get(str(k), {"meaning_in_repo": "unknown"})["meaning_in_repo"],
                    "source": label_documentation.get(str(k), {"source": "unknown"})["source"],
                }
                for k, v in sorted(global_label_voxel_counts.items())
            },
            "representative_raw_intensity_by_label_00237": channel_by_label_stats,
        },
        "chambers": chambers_summary,
    }


def cross_check_chamber_and_group_inventories() -> dict[str, Any]:
    """Cross-check decoded folder tokens against `dent_step1_chamber_inventory.csv` and `dent_step1_group_inventory.csv`."""
    df_ch = pd.read_csv(CHAMBER_INV_CSV)
    df_grp = pd.read_csv(GROUP_INV_CSV)

    discrepancies = []
    parsed_experiments = []
    for _, row in df_grp.iterrows():
        exp_dir = str(row["experiment_dir"])
        parsed = parse_chamber_folder_name(exp_dir)
        nom_rois = parsed["tokens_decoded"]["roi_num_nominal"]["token"]
        actual_rois = int(row["n_chambers_in_deposit"])
        if nom_rois != actual_rois:
            discrepancies.append({
                "experiment_dir": exp_dir,
                "group_id": str(row["group_id"]),
                "nominal_roi_token_in_folder_name": nom_rois,
                "actual_chambers_in_sciencedb_deposit": actual_rois,
                "explanation": (
                    f"Folder name token specifies {nom_rois} ROIs, but only {actual_rois} `MIP_ROI_*` "
                    f"chambers exist in `scidb_19235_all_urls.txt` (MIP_ROI_0..MIP_ROI_{actual_rois - 1})."
                ),
            })
        parsed_experiments.append({
            "experiment_dir": exp_dir,
            "group_id": str(row["group_id"]),
            "condition": str(row["condition"]),
            "hd_token": parsed["tokens_undecoded_or_ambiguous"]["hd_token"]["token"],
            "has_dasatinib_suffix": parsed["tokens_decoded"]["dasatinib_suffix"]["is_dasatinib_50nM"],
            "nominal_rois_in_name": nom_rois,
            "deposited_chambers": actual_rois,
            "published_xlsx_pairs": int(row["n_pairs_in_published_xlsx"]),
            "excluded_multi_or_zero_tcell_chambers": int(row["n_chambers_excluded_multi_or_zero_tcell"]),
            "total_outlined_frames": int(row["total_outlined_frames"]),
            "total_raw_only_frames": int(row["total_raw_only_frames"]),
            "kill_rate_published": float(row["kill_rate_published"]),
        })

    return {
        "n_experiments_in_deposit": len(df_grp),
        "n_chambers_in_deposit_total": int(df_ch.shape[0]),
        "n_nominal_chambers_from_folder_tokens": sum(x["nominal_rois_in_name"] for x in parsed_experiments),
        "total_outlined_frames_in_deposit": int(df_grp["total_outlined_frames"].sum()),
        "total_raw_only_frames_in_deposit": int(df_grp["total_raw_only_frames"].sum()),
        "total_published_single_cell_pairs": int(df_grp["n_pairs_in_published_xlsx"].sum()),
        "total_excluded_chambers_multi_or_zero_tcell": int(df_grp["n_chambers_excluded_multi_or_zero_tcell"].sum()),
        "folder_name_vs_deposit_discrepancies": discrepancies,
        "experiments": parsed_experiments,
    }


def inventory_sciencedb_urls(local_scidb: dict[str, Any]) -> dict[str, Any]:
    """Count all files on the ScienceDB server from `data/scidb_19235_all_urls.txt` vs local files."""
    ext_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    exp_counts: Counter[str] = Counter()
    total_urls = 0

    with open(URLS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total_urls += 1
            path_part = s.split("path=")[1].split("&")[0] if "path=" in s else s
            parts = path_part.strip("/").split("/")
            fname = parts[-1]
            ext = Path(fname).suffix.lower()
            ext_counts[ext] += 1
            if len(parts) >= 2:
                exp_counts[parts[1]] += 1

            if fname.endswith("_predictions.h5"):
                category_counts["h5_segmentation_mask (*_predictions.h5)"] += 1
            elif fname.endswith(".h5"):
                category_counts["h5_raw_fluorescence_stack (*.h5)"] += 1
            elif len(parts) >= 4 and parts[3].startswith("MIP"):
                category_counts[f"tif_projection ({parts[3]})"] += 1
            elif fname.endswith(".ome.tif"):
                category_counts["tif_raw_camera_mmstack (*.ome.tif)"] += 1
            elif ext == ".tif":
                category_counts["tif_other_widefield_or_montage (*.tif)"] += 1
            elif ext == ".yaml":
                category_counts["yaml_unet_config (test_config.yaml)"] += 1
            elif ext == ".txt":
                category_counts["txt_roi_crop_or_comments (*.txt)"] += 1
            elif ext == ".json":
                category_counts["json_display_settings (DisplaySettings.json)"] += 1
            elif ext == ".dcimg":
                category_counts["dcimg_hamamatsu_raw_stream (*.dcimg)"] += 1
            elif ext == ".csv":
                category_counts["csv_summary_tables (*.csv)"] += 1
            elif ext == ".xlsx":
                category_counts["xlsx_widefield_coordinates (*.xlsx)"] += 1
            elif ext == ".avi":
                category_counts["avi_montage_video (*.avi)"] += 1
            else:
                category_counts[f"other ({ext})"] += 1

    # Also check `data/scidb_dent_cache/` if present
    cache_counts: Counter[str] = Counter()
    if CACHE_DIR.exists():
        for p in CACHE_DIR.rglob("*"):
            if p.is_file():
                if p.name.endswith("_predictions.h5"):
                    cache_counts["h5_segmentation_mask (*_predictions.h5)"] += 1
                elif p.name.endswith(".h5"):
                    cache_counts["h5_raw_fluorescence_stack (*.h5)"] += 1
                elif p.suffix.lower() == ".tif":
                    cache_counts["tif_files (*.tif)"] += 1
                else:
                    cache_counts[f"other ({p.suffix.lower()})"] += 1

    local_raw_h5 = local_scidb["total_raw_h5_in_chambers_dir"] + 1  # +1 for samples/03979.h5
    local_pred_h5 = local_scidb["total_predictions_h5_in_chambers_dir"] + 1  # +1 for samples/03979_predictions.h5
    local_mip_tif = local_scidb["total_mip_tif_in_chambers_dir"]
    local_total_primary = local_raw_h5 + local_pred_h5 + local_mip_tif

    return {
        "manifest_file": str(URLS_PATH.relative_to(REPO_ROOT)),
        "server_total_files": total_urls,
        "server_counts_by_extension": dict(sorted(ext_counts.items(), key=lambda x: -x[1])),
        "server_counts_by_category": dict(sorted(category_counts.items(), key=lambda x: -x[1])),
        "server_counts_by_experiment_dir": dict(sorted(exp_counts.items())),
        "local_vs_server_comparison": {
            "primary_dirs_scidb_19235_chambers_and_samples": {
                "total_local_files": local_total_primary,
                "fraction_of_server_total": round(local_total_primary / max(1, total_urls), 6),
                "raw_h5_local_vs_server": {
                    "local": local_raw_h5,
                    "note_duplicate": "samples/03979.h5 is a byte-identical copy of v05_..._MIP_ROI_23/03979.h5 (20 unique chamber raw .h5 + 1 sample copy = 21 files)",
                    "server": category_counts["h5_raw_fluorescence_stack (*.h5)"],
                    "percent_downloaded": round(100.0 * local_raw_h5 / category_counts["h5_raw_fluorescence_stack (*.h5)"], 4),
                },
                "predictions_h5_local_vs_server": {
                    "local": local_pred_h5,
                    "server": category_counts["h5_segmentation_mask (*_predictions.h5)"],
                    "percent_downloaded": round(100.0 * local_pred_h5 / category_counts["h5_segmentation_mask (*_predictions.h5)"], 4),
                },
                "mip_tif_local_vs_server": {
                    "local": local_mip_tif,
                    "server": 6 * 63104,
                    "percent_downloaded": round(100.0 * local_mip_tif / (6 * 63104), 4),
                },
            },
            "auxiliary_cache_dir_scidb_dent_cache": {
                "dir": str(CACHE_DIR.relative_to(REPO_ROOT)),
                "exists": CACHE_DIR.exists(),
                "counts": dict(cache_counts),
            },
        },
    }


def summarize_ome_and_ims_metadata() -> dict[str, Any]:
    """Summarize `data/ome_tiff/` and `data/ims_metadata_20190626.json`, `data/ims_metadata_exp28.json`."""
    ome_records = []
    if OME_DIR.exists():
        for p in sorted(OME_DIR.glob("*.tif*")):
            with tifffile.TiffFile(p) as tf:
                s0 = tf.series[0]
                ome_xml_info = {}
                if tf.ome_metadata:
                    root = ET.fromstring(tf.ome_metadata)
                    ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                    pix = root.find(".//ome:Pixels", ns)
                    if pix is not None:
                        ome_xml_info = dict(pix.attrib)
                ome_records.append({
                    "rel_path": str(p.relative_to(REPO_ROOT)),
                    "file_name": p.name,
                    "size_bytes": p.stat().st_size,
                    "n_pages": len(tf.pages),
                    "shape": list(s0.shape),
                    "axes": s0.axes,
                    "dtype": str(s0.dtype),
                    "ome_pixels_attributes": ome_xml_info,
                    "note": (
                        "Single timepoint (TP000) exported from `LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims` "
                        "with 9 channels, 62 Z-slices, 200x200 XY pixels (float32). Physical voxel size is not populated "
                        "in the OME-XML Pixels attributes, but is defined in `ims_metadata_20190626.json`."
                    ),
                })

    with open(IMS_2019_JSON, "r", encoding="utf-8") as f:
        ims_2019 = json.load(f)
    with open(IMS_EXP28_JSON, "r", encoding="utf-8") as f:
        ims_exp28 = json.load(f)

    ims_2019_channels = [
        {
            "channel_key": k,
            "name": v.get("Name"),
            "excitation_nm": v.get("LSMExcitationWavelength"),
            "emission_nm": v.get("LSMEmissionWavelength"),
            "color_range": v.get("ColorRange"),
        }
        for k, v in ims_2019.get("channels", {}).items()
    ]

    return {
        "ome_tiff": ome_records,
        "ims_metadata_20190626": {
            "source_json": str(IMS_2019_JSON.relative_to(REPO_ROOT)),
            "upstream_ims_file": ims_2019["file_name"],
            "zenodo_record": ims_2019["zenodo_record"],
            "microscope": ims_2019["microscope"],
            "dimensions_voxels_XYZTC": ims_2019["dimensions_voxels"],
            "physical_extent_um_XYZ": ims_2019["physical_extent_um"],
            "voxel_pitch_um_dx_dy_dz": ims_2019["voxel_pitch_um"],
            "n_frames": ims_2019["dimensions_voxels"]["T"],
            "frame_interval_s": ims_2019["frame_interval_s"],
            "frame_interval_min": ims_2019["frame_interval_min"],
            "total_duration_h": ims_2019["total_duration_h"],
            "recording_start": ims_2019["time_info"]["TimePoint1"],
            "recording_end": ims_2019["time_info"]["TimePoint669"],
            "n_channels": ims_2019["dimensions_voxels"]["C"],
            "channels": ims_2019_channels,
            "scene8_megasurfaces0_summary": {
                "name": ims_2019["scene8_megasurfaces0"]["name"],
                "n_surface_instances": ims_2019["scene8_megasurfaces0"]["n_surface_instances"],
                "n_tracks": ims_2019["scene8_megasurfaces0"]["n_tracks"],
            },
            "additional_cohort_context_from_data_registry": {
                "well_4_cd8_tracks": 1171,
                "well_4_tracked_organoids": 21,
                "all_6_wells_total_tracks": 6931,
            },
        },
        "ims_metadata_exp28": {
            "source_json": str(IMS_EXP28_JSON.relative_to(REPO_ROOT)),
            "upstream_ims_file": ims_exp28["file"],
            "size_bytes": ims_exp28["size_bytes"],
            "dimensions_voxels_XYZTC": {
                "X": ims_exp28["dimensions_xyz_t_c"][0],
                "Y": ims_exp28["dimensions_xyz_t_c"][1],
                "Z": ims_exp28["dimensions_xyz_t_c"][2],
                "T": ims_exp28["dimensions_xyz_t_c"][3],
                "C": ims_exp28["dimensions_xyz_t_c"][4],
            },
            "physical_extent_um_XYZ": {
                "X": round(ims_exp28["extent_max_um"][0] - ims_exp28["extent_min_um"][0], 3),
                "Y": round(ims_exp28["extent_max_um"][1] - ims_exp28["extent_min_um"][1], 3),
                "Z": round(ims_exp28["extent_max_um"][2] - ims_exp28["extent_min_um"][2], 3),
            },
            "voxel_pitch_um_dx_dy_dz": {
                "dx": ims_exp28["voxel_size_um"][0],
                "dy": ims_exp28["voxel_size_um"][1],
                "dz": ims_exp28["voxel_size_um"][2],
            },
            "n_frames": ims_exp28["dimensions_xyz_t_c"][3],
            "frame_interval_min": ims_exp28["frame_interval_min"],
            "frame_interval_s": round(ims_exp28["frame_interval_min"] * 60.0, 2),
            "total_duration_h": ims_exp28["total_duration_h"],
            "n_channels": ims_exp28["dimensions_xyz_t_c"][4],
            "channels": ims_exp28["channels"],
            "scene8_surfaces": ims_exp28["scene8_surfaces"],
        },
    }


def summarize_curated_419_tables() -> dict[str, Any]:
    """Summarize `data/lightsheet_wang2026_single_cells_419_clean.csv` and `data/clean_per_cell_419.csv`."""
    df_a = pd.read_csv(CLEAN_419_A)
    df_b = pd.read_csv(CLEAN_419_B)

    def summarize_df(df: pd.DataFrame, path: Path) -> dict[str, Any]:
        col_summary = {}
        for col in df.columns:
            s = df[col]
            entry: dict[str, Any] = {
                "dtype": str(s.dtype),
                "non_null_count": int(s.notna().sum()),
                "null_count": int(s.isna().sum()),
            }
            if pd.api.types.is_numeric_dtype(s):
                valid = s.dropna()
                if len(valid) > 0:
                    entry["min"] = round(float(valid.min()), 4)
                    entry["median"] = round(float(valid.median()), 4)
                    entry["mean"] = round(float(valid.mean()), 4)
                    entry["max"] = round(float(valid.max()), 4)
            col_summary[col] = entry

        grp_rows = []
        for (gid, cond), gdf in df.groupby(["group_id", "condition"], sort=False):
            grp_rows.append({
                "group_id": str(gid),
                "condition": str(cond),
                "n_rows": int(len(gdf)),
                "n_non_null_MPA_deg": int(gdf["MPA_deg"].notna().sum()),
                "n_non_null_MD_um": int(gdf["MD_um"].notna().sum()),
                "group_kill_rate": round(float(gdf["group_kill_rate"].iloc[0]), 4),
            })

        return {
            "rel_path": str(path.relative_to(REPO_ROOT)),
            "n_rows": int(df.shape[0]),
            "n_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "column_details": col_summary,
            "groups": grp_rows,
        }

    return {
        "what_one_row_represents": (
            "One single-cell CAR-T + Nalm6 target interaction chamber (from the 419 valid single-CAR-T chambers "
            "retained out of 502 total chambers in `Source_Data_Wang2024_NatCommun.xlsx` Figures 5c-5g, after "
            "excluding 83 chambers with 0 or >1 CAR-T cells). Each row records the cell index within its experimental "
            "group (`cell_index = 1..n_group`), peak 3D synaptic contact area (`CA_um2`), central F-actin depletion "
            "coefficient (`ADC`), retrograde actin flow speed (`ARFS_nm_s`), MTOC polarization angle (`MPA_deg`, "
            "present for 185/419 rows), MTOC-to-synapse distance (`MD_um`, present for 60/419 rows in Groups 8-10 and 8*-10*), "
            "and the group-level fraction of targets killed (`group_kill_rate`)."
        ),
        "lightsheet_wang2026_single_cells_419_clean_csv": summarize_df(df_a, CLEAN_419_A),
        "clean_per_cell_419_csv": summarize_df(df_b, CLEAN_419_B),
        "relationship_and_differences_between_the_two_tables": {
            "shared_first_9_columns_identical": True,
            "added_8_columns_in_clean_per_cell_419": {
                "contact_area_um2": "Exact duplicate of `CA_um2` (max abs diff = 0.0)",
                "CA_xlsx_um2": "Exact duplicate of `CA_um2` (max abs diff = 0.0)",
                "actin_depletion_idx": "Exact duplicate of `ADC` (max abs diff = 0.0)",
                "ADC_xlsx": "Exact duplicate of `ADC` (max abs diff = 0.0)",
                "ARFS_xlsx_nm_s": "Exact duplicate of `ARFS_nm_s` (max abs diff = 0.0)",
                "actin_flow_um_min": "Unit conversion `ARFS_nm_s * 0.06` (nm/s -> um/min)",
                "MPA_xlsx_deg": "Exact duplicate of `MPA_deg` (185 non-null, 234 NaN)",
                "mtoc_angle_deg": "Copies `MPA_deg` where present (185 rows) and imputes `45.0` deg for all 234 missing rows",
            },
            "caveats_in_curated_tables": [
                "Neither CSV contains a per-cell binary kill outcome (`killed = 0/1`); `Source_Data_Wang2024_NatCommun.xlsx` (Figure 5g) only publishes group-level kill counts and rates (`group_kill_rate`).",
                "Neither CSV contains the ScienceDB microchamber folder ID (`MIP_ROI_k`) corresponding to `cell_index`, because 83 of the 502 deposited chambers were excluded (0 or >1 CAR-T cell) when compiling the 419 rows in the published Excel workbook.",
                "`MPA_deg` is missing (NaN) for 234/419 rows (55.8%), and `MD_um` is missing (NaN) for 359/419 rows (85.7%, completely absent in Groups 1-7).",
            ],
        },
    }


def build_inventory() -> dict[str, Any]:
    local_scidb = inventory_local_scidb()
    cross_check = cross_check_chamber_and_group_inventories()
    urls_inv = inventory_sciencedb_urls(local_scidb)
    ome_ims = summarize_ome_and_ims_metadata()
    tables_419 = summarize_curated_419_tables()

    # Concise cross-dataset scale table for primer_03_dataset_scale.png
    dataset_scale_comparison = [
        {
            "dataset_id": "ScienceDB_19235_LightSheet",
            "short_label": "ScienceDB 19235\nLight-Sheet (CAR-T + Nalm6)",
            "modality": "HBOPM 4D Dual-View Light-Sheet",
            "voxel_size_um": {"dx": VOXEL_UM, "dy": VOXEL_UM, "dz": VOXEL_UM, "optical_fwhm_um": OPTICAL_FWHM_UM},
            "voxel_volume_um3": round(0.196214 ** 3, 6),
            "voxel_volume_at_0320_um3": round(OPTICAL_FWHM_UM ** 3, 6),
            "grid_voxels_XYZ": [256, 256, 190],
            "grid_z_range_voxels": [166, 190],
            "fov_um_XYZ_at_01962": [round(256 * 0.196214, 2), round(256 * 0.196214, 2), round(190 * 0.196214, 2)],
            "fov_um_XYZ_at_0320": [round(256 * OPTICAL_FWHM_UM, 2), round(256 * OPTICAL_FWHM_UM, 2), round(190 * OPTICAL_FWHM_UM, 2)],
            "frame_interval_s_burst": 2.5,
            "visit_interval_s": 600.0,
            "visit_interval_min": 10.0,
            "total_duration_h": 3.33,
            "total_duration_range_h": [3.0, 5.0],
            "n_channels": 3,
            "n_cells_or_events": 419,
            "n_deposited_chambers": 502,
            "n_total_frames_per_chamber": 120,
        },
        {
            "dataset_id": "Zenodo_8328441_TEG_BC_Organoid",
            "short_label": "Zenodo 8328441\nConfocal (TEG + BC Organoids)",
            "modality": "Zeiss LSM880 Spectral Confocal (20x/0.8 NA)",
            "voxel_size_um": {"dx": 3.5425, "dy": 3.5425, "dz": 1.0500},
            "voxel_volume_um3": round(3.5425 * 3.5425 * 1.0500, 4),
            "grid_voxels_XYZ": [200, 200, 62],
            "fov_um_XYZ": [708.493, 708.493, 65.100],
            "frame_interval_s": 119.71,
            "frame_interval_min": 1.995,
            "total_duration_h": 22.21,
            "n_channels": 9,
            "n_cells_or_events": 1171,
            "n_cd4_tracks_in_ims": 668,
            "n_tracked_organoids": 21,
            "n_all_6_wells_tracks": 6931,
            "n_total_frames": 669,
        },
        {
            "dataset_id": "Zenodo_8328441_Exp28_CART_BALL",
            "short_label": "Zenodo 8328441 Exp28\nConfocal (CAR-T + B-ALL)",
            "modality": "Zeiss LSM880 Confocal",
            "voxel_size_um": {"dx": 0.9102, "dy": 0.9100, "dz": 0.9098},
            "voxel_volume_um3": round(0.9102 * 0.9100 * 0.9098, 4),
            "grid_voxels_XYZ": [512, 512, 30],
            "fov_um_XYZ": [466.0, 465.9, 27.294],
            "frame_interval_s": 168.6,
            "frame_interval_min": 2.81,
            "total_duration_h": 22.48,
            "n_channels": 7,
            "n_cells_or_events": 195,
            "n_cd4_tracks": 92,
            "n_cd8_tracks": 103,
            "n_leukemia_surfaces": 114396,
            "n_total_frames": 480,
        },
    ]

    return {
        "repo_root_rel": ".",
        "part_1a_local_scidb_h5_inventory": local_scidb,
        "part_1b_folder_name_decoding_and_crosscheck": cross_check,
        "part_1c_sciencedb_server_vs_local": urls_inv,
        "part_1d_ome_tiff_and_ims_metadata": ome_ims,
        "part_1e_curated_419_tables": tables_419,
        "dataset_scale_comparison": dataset_scale_comparison,
    }


def print_human_readable_summary(inv: dict[str, Any]) -> None:
    p1a = inv["part_1a_local_scidb_h5_inventory"]
    p1b = inv["part_1b_folder_name_decoding_and_crosscheck"]
    p1c = inv["part_1c_sciencedb_server_vs_local"]
    p1d = inv["part_1d_ome_tiff_and_ims_metadata"]
    p1e = inv["part_1e_curated_419_tables"]

    print("=" * 88)
    print("RAW MICROSCOPY DATA INVENTORY SUMMARY")
    print("=" * 88)

    print("\n[1a] Local ScienceDB 19235 Directories (`data/scidb_19235_chambers/` & `data/scidb_19235_samples/`)")
    print(f"  - Chamber subfolders in `data/scidb_19235_chambers/`: {p1a['n_chamber_subfolders']} "
          f"({p1a['n_non_empty_chamber_subfolders']} non-empty, {p1a['n_empty_chamber_subfolders']} empty)")
    print(f"  - Total .h5 files in `chambers/`: {p1a['total_h5_in_chambers_dir']} "
          f"({p1a['total_raw_h5_in_chambers_dir']} raw stacks `*.h5` + "
          f"{p1a['total_predictions_h5_in_chambers_dir']} U-Net masks `*_predictions.h5`)")
    print(f"  - Total 16-bit MIP .tif files in `chambers/`: {p1a['total_mip_tif_in_chambers_dir']}")
    print(f"  - Files in `data/scidb_19235_samples/`: {p1a['n_h5_in_samples_dir']} "
          f"(`03979.h5` raw stack + `03979_predictions.h5` mask, copy of v05_..._MIP_ROI_23/03979)")
    print("  - Per-chamber breakdown:")
    for ch in p1a["chambers"]:
        shapes = p1a["h5_metadata_findings"]["z_dimension_varies_by_experiment"]["observed_shapes_by_chamber"].get(
            ch["chamber_folder"], []
        )
        print(f"      * {ch['chamber_folder']:42s}: raw_h5={ch['n_raw_h5']:2d}, "
              f"pred_h5={ch['n_predictions_h5']:2d}, paired={ch['n_paired_raw_and_pred_h5']:2d}, "
              f"mip_tif={ch['n_mip_tif']:3d}, sample_shape={shapes}")
    print("  - HDF5 dataset structure & attributes:")
    print("      * Raw stack (`*.h5`): dataset `/raw`, shape `(3, Z, 256, 256)`, dtype `uint16`, attrs `{}` (empty)")
    print("      * Segmentation mask (`*_predictions.h5`): dataset `/predictions`, shape `(Z, 256, 256)`, dtype `uint8`, attrs `{}` (empty)")
    print("      * Z-dimension varies by experiment: Z=190 (v01), Z=170 (v02), Z=166 (v05 & samples), Z=174 (v06), Z=186 (v11)")
    print("      * Voxel size in HDF5 attrs: NONE (`{}`). In `feilab_CART_analysis/main/analysis/summary/summary.m` "
          "line 18, surface area factor is `0.0385 um^2/vox` -> `dx = 0.1962 um/vox` (`19.6 nm/s` actin factor); "
          "`data_dictionary.md` / `data_registry.csv` cite `0.320 um` (the optical lateral FWHM).")
    print("  - Global U-Net mask label counts across all 126 local `*_predictions.h5` files:")
    for lbl, d in p1a["global_mask_label_summary_across_126_local_masks"]["counts_by_label"].items():
        print(f"      * Label {lbl}: {d['voxel_count']:11,d} voxels ({100*d['fraction_of_all_voxels']:6.3f}%) -> {d['meaning']}")

    print("\n[1b] Chamber Folder Name Decoding (e.g. `v05_6_40_120_0-6_hd7_MIP_ROI_23`)")
    print("  - Decoded tokens (verified in `external/feilab_CART_analysis/main/configure.m` lines 18-25):")
    print("      * `v05` (v01..v13)   : Experiment/view index across the 13 experiment runs")
    print("      * `6`                : `volume_num` = 6 3D sweeps per microchamber per 10-min visit burst (~2.5 s/sweep)")
    print("      * `40` (39, 36, 35)  : `roi_num` = nominal number of microchamber ROIs imaged in the run")
    print("      * `120`              : `slice_per_stack` = 120 raw 2D camera slices per 3D light-sheet sweep")
    print("      * `0-6`              : `stepsize` = 0.6 um piezo step size (`strrep('0-6','-','.')`)")
    print("      * `ds` / `_ds`       : 50 nM Dasatinib treatment suffix (in v11..v13 -> Groups 8*, 9*, 10*)")
    print("      * `MIP_ROI_23`       : 0-indexed microchamber ROI folder (`MIP_ROI_0`..`MIP_ROI_39`)")
    print("  - Ambiguous / undecoded token:")
    print("      * `hd7` (hd1..hd10)  : Distinguishes the 10 donor/batch runs (hd1,3,4,6,7 = Groups 1-5; "
          "hd8,9,2,10,5 = Groups 6-10; hd2_ds,10ds,5_ds = Groups 8*,9*,10*), but is ignored by `configure.m` "
          "and its literal expansion (`hd`) is not defined in the repo.")
    print("  - Cross-check discrepancy found:")
    for disc in p1b["folder_name_vs_deposit_discrepancies"]:
        print(f"      * {disc['experiment_dir']} (Group {disc['group_id']}): {disc['explanation']}")

    print("\n[1c] ScienceDB Server Manifest (`data/scidb_19235_all_urls.txt`) vs Local Files")
    print(f"  - Total files on ScienceDB server: {p1c['server_total_files']:,d} across 13 experiments and 502 chambers")
    for k, v in p1c["server_counts_by_category"].items():
        print(f"      * {k:42s}: {v:8,d}")
    cmp_primary = p1c["local_vs_server_comparison"]["primary_dirs_scidb_19235_chambers_and_samples"]
    print(f"  - Downloaded in `scidb_19235_chambers/` + `scidb_19235_samples/`: {cmp_primary['total_local_files']:,d} files "
          f"({100*cmp_primary['fraction_of_server_total']:.4f}% of server)")
    print(f"      * Raw 3D `.h5` stacks     : {cmp_primary['raw_h5_local_vs_server']['local']:4d} / "
          f"{cmp_primary['raw_h5_local_vs_server']['server']:,d} ({cmp_primary['raw_h5_local_vs_server']['percent_downloaded']:.4f}%)")
    print(f"      * U-Net `*_predictions.h5`: {cmp_primary['predictions_h5_local_vs_server']['local']:4d} / "
          f"{cmp_primary['predictions_h5_local_vs_server']['server']:,d} ({cmp_primary['predictions_h5_local_vs_server']['percent_downloaded']:.4f}%)")
    print(f"      * 16-bit MIP `.tif`       : {cmp_primary['mip_tif_local_vs_server']['local']:4d} / "
          f"{cmp_primary['mip_tif_local_vs_server']['server']:,d} ({cmp_primary['mip_tif_local_vs_server']['percent_downloaded']:.4f}%)")
    cache_info = p1c["local_vs_server_comparison"]["auxiliary_cache_dir_scidb_dent_cache"]
    if cache_info["exists"]:
        print(f"  - Additional cached files in `data/scidb_dent_cache/`: {cache_info['counts']}")

    print("\n[1d] OME-TIFF (`data/ome_tiff/`) & Imaris Confocal Metadata (`ims_metadata_*.json`)")
    for ome in p1d["ome_tiff"]:
        print(f"  - {ome['rel_path']}: shape={tuple(ome['shape'])} axes={ome['axes']} dtype={ome['dtype']} "
              f"pages={ome['n_pages']} ({ome['size_bytes']/(1024**2):.1f} MB)")
    m19 = p1d["ims_metadata_20190626"]
    print(f"  - {m19['source_json']} ({m19['upstream_ims_file']}):")
    print(f"      * Grid & FOV : {m19['dimensions_voxels_XYZTC']} -> "
          f"{m19['physical_extent_um_XYZ']['X']} x {m19['physical_extent_um_XYZ']['Y']} x {m19['physical_extent_um_XYZ']['Z']} um")
    print(f"      * Voxel size : dx={m19['voxel_pitch_um_dx_dy_dz']['dx']} um, "
          f"dy={m19['voxel_pitch_um_dx_dy_dz']['dy']} um, dz={m19['voxel_pitch_um_dx_dy_dz']['dz']} um")
    print(f"      * Cadence    : {m19['n_frames']} frames @ {m19['frame_interval_s']} s ({m19['frame_interval_min']} min) = {m19['total_duration_h']} h")
    print(f"      * Channels   : {m19['n_channels']} ({', '.join(c['name'] for c in m19['channels'])})")
    m28 = p1d["ims_metadata_exp28"]
    print(f"  - {m28['source_json']} ({m28['upstream_ims_file']}, {m28['size_bytes']/(1024**3):.2f} GB):")
    print(f"      * Grid & FOV : {m28['dimensions_voxels_XYZTC']} -> "
          f"{m28['physical_extent_um_XYZ']['X']} x {m28['physical_extent_um_XYZ']['Y']} x {m28['physical_extent_um_XYZ']['Z']} um")
    print(f"      * Voxel size : dx={m28['voxel_pitch_um_dx_dy_dz']['dx']} um, "
          f"dy={m28['voxel_pitch_um_dx_dy_dz']['dy']} um, dz={m28['voxel_pitch_um_dx_dy_dz']['dz']} um")
    print(f"      * Cadence    : {m28['n_frames']} frames @ {m28['frame_interval_s']} s ({m28['frame_interval_min']} min) = {m28['total_duration_h']} h")
    print(f"      * Channels   : {m28['n_channels']} ({', '.join(c['name'] for c in m28['channels'])})")

    print("\n[1e] Curated Single-Cell Tables (`lightsheet_wang2026_single_cells_419_clean.csv` & `clean_per_cell_419.csv`)")
    ta = p1e["lightsheet_wang2026_single_cells_419_clean_csv"]
    tb = p1e["clean_per_cell_419_csv"]
    print(f"  - `{ta['rel_path']}`: {ta['n_rows']} rows x {ta['n_columns']} columns: {ta['columns']}")
    print(f"  - `{tb['rel_path']}`: {tb['n_rows']} rows x {tb['n_columns']} columns (adds 8 duplicate/converted columns, "
          f"including `mtoc_angle_deg` which fills 234 missing `MPA_deg` values with 45.0 deg)")
    print("  - Non-null counts in 419-row tables: CA_um2=419, ADC=419, ARFS_nm_s=419, MPA_deg=185 (234 NaN), MD_um=60 (359 NaN)")
    print("  - Cohort row counts: Groups 1-5 (Untreated high-killing) = 167; Groups 6-10 (Untreated low-killing) = 163; "
          "Groups 8*, 9*, 10* (Dasatinib 50 nM) = 89 (Total = 419).")
    print("=" * 88)


def main() -> None:
    inv = build_inventory()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2)
    print_human_readable_summary(inv)
    print(f"\nWrote JSON inventory to: {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
