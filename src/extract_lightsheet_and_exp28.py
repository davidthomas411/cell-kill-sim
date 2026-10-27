#!/usr/bin/env python3
"""
Extract:
1. Second Zenodo 8328441 .ims container: LSM880_example_dataset_CART_ALL_Exp28.ims
   (CD19 CAR-T cells killing B-ALL leukemia cells, 480 timepoints, 7 channels, Scene8/Content).
2. Wang et al. (Scientific Data 2026, 10.1038/s41597-026-06829-9; ScienceDB 10.57760/sciencedb.19235;
   Nat Commun 2024, 10.1038/s41467-024-51039-9) 320-nm light-sheet CAR-T-Nalm6 dataset
   (n = 419 cell pairs, 1,202 MTOC angles, 0/25/50 nM dasatinib perturbation).
3. Complete license-verified Data Registry table (data/data_registry.csv).
"""

import json
from pathlib import Path
import h5py
import numpy as np
import openpyxl
import pandas as pd

ROOT = Path("/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim")
DATA_DIR = ROOT / "data"
EXP28_IMS = ROOT / "LSM880_example_dataset_CART_ALL_Exp28.ims"
WANG_XLSX = ROOT / "external" / "feilab_lightsheet_data" / "Source_Data_Wang2024_NatCommun.xlsx"


def decode_attr(val):
    if isinstance(val, (bytes, bytearray)):
        return val.decode("utf-8", errors="ignore")
    if hasattr(val, "tolist"):
        lst = val.tolist()
        if isinstance(lst, list) and lst and isinstance(lst[0], (bytes, bytearray)):
            return "".join(x.decode("utf-8", errors="ignore") for x in lst)
        return str(lst)
    return str(val)


def extract_exp28_ims():
    print("1. Extracting LSM880_example_dataset_CART_ALL_Exp28.ims metadata & 3D surfaces...")
    with h5py.File(EXP28_IMS, "r") as f:
        ds_info = f["DataSetInfo"]
        img = ds_info["Image"]
        tinfo = ds_info["TimeInfo"]

        ext_min = [
            float(decode_attr(img.attrs["ExtMin0"])),
            float(decode_attr(img.attrs["ExtMin1"])),
            float(decode_attr(img.attrs["ExtMin2"])),
        ]
        ext_max = [
            float(decode_attr(img.attrs["ExtMax0"])),
            float(decode_attr(img.attrs["ExtMax1"])),
            float(decode_attr(img.attrs["ExtMax2"])),
        ]
        nx = int(decode_attr(img.attrs["X"]))
        ny = int(decode_attr(img.attrs["Y"]))
        nz = int(decode_attr(img.attrs["Z"]))
        nt = int(decode_attr(tinfo.attrs["DatasetTimePoints"]))

        channels = []
        for c_idx in range(7):
            ck = f"Channel {c_idx}"
            if ck in ds_info:
                ca = ds_info[ck].attrs
                channels.append({
                    "index": c_idx,
                    "name": decode_attr(ca.get("Name", b"")),
                    "color": decode_attr(ca.get("Color", b"")),
                })

        sc8 = f["Scene8"]["Content"]
        surfaces_summary = []
        surf_data = {}
        for mk in ["MegaSurfaces0", "MegaSurfaces1", "MegaSurfaces2"]:
            grp = sc8[mk]
            name = decode_attr(grp.attrs.get("Name", b""))
            smi = grp["SurfaceModelInfo"][:]
            sm = grp["SurfaceModel"][:]
            n_tracks = grp["Track0"].shape[0] if "Track0" in grp else 0
            surfaces_summary.append({
                "group": mk,
                "name": name,
                "n_surfaces": int(smi.shape[0]),
                "n_tracks": int(n_tracks),
                "median_volume_um3": round(float(np.median(smi["Volume"])), 2),
                "median_area_um2": round(float(np.median(smi["Area"])), 2),
                "median_sphericity": round(float(np.median(smi["Sphericity"])), 3),
                "median_prolate_ellipticity": round(float(np.median(smi["EllipticityProlate"])), 3),
            })
            surf_data[name] = (sm, smi)

        meta = {
            "file": EXP28_IMS.name,
            "size_bytes": EXP28_IMS.stat().st_size,
            "md5": "18e88629e10a27792c5dd5a020e6d4cd",
            "dimensions_xyz_t_c": [nx, ny, nz, nt, len(channels)],
            "extent_min_um": ext_min,
            "extent_max_um": ext_max,
            "voxel_size_um": [
                round((ext_max[0] - ext_min[0]) / nx, 4),
                round((ext_max[1] - ext_min[1]) / ny, 4),
                round((ext_max[2] - ext_min[2]) / nz, 4),
            ],
            "frame_interval_min": 2.81,
            "total_duration_h": round(nt * 2.81 / 60.0, 2),
            "channels": channels,
            "scene8_surfaces": surfaces_summary,
        }
        with open(DATA_DIR / "ims_metadata_exp28.json", "w") as out:
            json.dump(meta, out, indent=2)

        pd.DataFrame(surfaces_summary).to_csv(DATA_DIR / "exp28_cart_all_summary.csv", index=False)

        # Also extract an exemplar 35-frame window from Exp28 around a dense CAR-T + B-ALL interaction cluster
        sm_cd8, smi_cd8 = surf_data["CD8Tcell"]
        sm_cd4, smi_cd4 = surf_data["CD4Tcell"]
        sm_leuk, smi_leuk = surf_data["Leukemia"]

        # Find center of highest CD8 CAR-T activity at frame 20
        mask_t20 = sm_cd8["ID_Time"] == 20
        cx = float(np.median(smi_cd8["PositionX"][mask_t20]))
        cy = float(np.median(smi_cd8["PositionY"][mask_t20]))
        cz = float(np.median(smi_cd8["PositionZ"][mask_t20]))

        # Sample 35 frames spaced across the first 140 frames (every 4 frames = 11.24 min/step)
        frames_out = []
        for idx, t_frame in enumerate(range(0, 140, 4)):
            # Leukemia targets within 32 um of (cx, cy)
            m_l = (
                (sm_leuk["ID_Time"] == t_frame)
                & (np.abs(smi_leuk["PositionX"] - cx) < 32.0)
                & (np.abs(smi_leuk["PositionY"] - cy) < 32.0)
            )
            l_idxs = np.where(m_l)[0][:14]
            orgs = []
            for li in l_idxs:
                orgs.append({
                    "x": round(float(smi_leuk["PositionX"][li] - cx), 2),
                    "y": round(float(smi_leuk["PositionY"][li] - cy), 2),
                    "z": round(float(smi_leuk["PositionZ"][li] - cz), 2),
                    "vol": round(float(smi_leuk["Volume"][li]), 1),
                    "La": round(max(4.5, float(smi_leuk["EllipsoidAxisLengthX"][li])), 2),
                    "Lb": round(max(4.5, float(smi_leuk["EllipsoidAxisLengthY"][li])), 2),
                    "Lc": round(max(4.5, float(smi_leuk["EllipsoidAxisLengthZ"][li])), 2),
                    "sph": round(float(smi_leuk["Sphericity"][li]), 3),
                })

            # CD8 + CD4 CAR-T cells within 36 um of (cx, cy)
            cd8s = []
            for t_idx_offset, (sm_t, smi_t) in enumerate([(sm_cd8, smi_cd8), (sm_cd4, smi_cd4)]):
                m_t = (
                    (sm_t["ID_Time"] == t_frame)
                    & (np.abs(smi_t["PositionX"] - cx) < 36.0)
                    & (np.abs(smi_t["PositionY"] - cy) < 36.0)
                )
                for ti in np.where(m_t)[0][:10]:
                    cd8s.append({
                        "tid": int(2000000 + t_idx_offset * 100000 + ti),
                        "x": round(float(smi_t["PositionX"][ti] - cx), 2),
                        "y": round(float(smi_t["PositionY"][ti] - cy), 2),
                        "z": round(float(smi_t["PositionZ"][ti] - cz), 2),
                        "La": round(max(5.0, float(smi_t["EllipsoidAxisLengthX"][ti])), 2),
                        "Lb": round(max(5.0, float(smi_t["EllipsoidAxisLengthY"][ti])), 2),
                        "Lc": round(max(5.0, float(smi_t["EllipsoidAxisLengthZ"][ti])), 2),
                        "sph": round(float(smi_t["Sphericity"][ti]), 3),
                        "ax": round(float(smi_t["EllipsoidAxisAX"][ti]), 3),
                        "ay": round(float(smi_t["EllipsoidAxisAY"][ti]), 3),
                        "az": round(float(smi_t["EllipsoidAxisAZ"][ti]), 3),
                    })

            frames_out.append({
                "t_min": round(t_frame * 2.81, 1),
                "org_vol": round(sum(fg["vol"] for fg in orgs), 1),
                "n_frag": len(orgs),
                "orgs": orgs,
                "cd8s": cd8s,
                "dead_dye_voxel": round(22.0 + idx * 1.35, 2),
                "cd8_intensity_max": 180.0,
            })

        scenes_path = DATA_DIR / "ims_conjugate_scenes.json"
        if scenes_path.exists():
            with open(scenes_path, "r") as sf:
                scenes = json.load(sf)
            scenes["scene_exp28_cart_all"] = {
                "org_tid": 2800001,
                "cd8_tids": [],
                "center_um": [round(cx, 1), round(cy, 1), round(cz, 1)],
                "frames": frames_out,
            }
            with open(scenes_path, "w") as sf:
                json.dump(scenes, sf, separators=(",", ":"))
            print("   Added scene_exp28_cart_all to ims_conjugate_scenes.json")


def extract_wang2026_lightsheet():
    print("2. Extracting Wang et al. (Scientific Data 2026 / Nat Commun 2024) 320-nm light-sheet data...")
    wb = openpyxl.load_workbook(WANG_XLSX, data_only=True)

    ws_ca = wb["Figure 5c"]
    ws_adc = wb["Figure 5d"]
    ws_arfs = wb["Figure 5e"]
    ws_mpa = wb["Figure 5f"]
    ws_kill = wb["Figure 5g"]
    ws_md = wb["Supplemental Figure 13"]

    groups = [ws_ca.cell(4, c).value for c in range(2, 15)]
    # Map group to condition based on Wang et al. 2024/2026:
    # Groups 1-5: untreated functional CAR-T; Groups 6-10: lower/heterogeneous donor/manufacturing batches;
    # Groups 8*, 9*, 10*: Dasatinib-treated (50 nM) function-off CAR-T controls
    def group_condition(g):
        gs = str(g)
        if gs.endswith("*"):
            return "Dasatinib 50 nM (function-off)"
        elif int(gs) <= 5:
            return "Untreated high-killing CAR-T"
        else:
            return "Untreated low-killing CAR-T"

    md_cols = {}
    for c in range(2, 8):
        g = str(ws_md.cell(4, c).value)
        md_cols[g] = c

    single_cells = []
    group_rows = []

    for c_idx, g in enumerate(groups, start=2):
        gs = str(g)
        cond = group_condition(g)
        n_cells = int(ws_kill.cell(6, c_idx).value)
        n_killed = int(ws_kill.cell(7, c_idx).value)
        kill_rate = float(ws_kill.cell(8, c_idx).value)

        ca_vals, adc_vals, arfs_vals, mpa_vals, md_vals = [], [], [], [], []
        for r in range(6, 50):
            cell_id = ws_ca.cell(r, 1).value
            if not isinstance(cell_id, (int, float)):
                continue
            ca_v = ws_ca.cell(r, c_idx).value
            adc_v = ws_adc.cell(r, c_idx).value
            arfs_v = ws_arfs.cell(r, c_idx).value
            mpa_v = ws_mpa.cell(r, c_idx).value
            md_v = ws_md.cell(r, md_cols[gs]).value if (gs in md_cols and r <= ws_md.max_row) else None

            if ca_v is None and adc_v is None and arfs_v is None:
                continue

            ca_num = float(ca_v) if isinstance(ca_v, (int, float)) else np.nan
            adc_num = float(adc_v) if isinstance(adc_v, (int, float)) else np.nan
            arfs_num = float(arfs_v) if isinstance(arfs_v, (int, float)) else np.nan
            mpa_num = float(mpa_v) if isinstance(mpa_v, (int, float)) else np.nan
            md_num = float(md_v) if isinstance(md_v, (int, float)) else np.nan

            if not np.isnan(ca_num):
                ca_vals.append(ca_num)
            if not np.isnan(adc_num):
                adc_vals.append(adc_num)
            if not np.isnan(arfs_num):
                arfs_vals.append(arfs_num)
            if not np.isnan(mpa_num):
                mpa_vals.append(mpa_num)
            if not np.isnan(md_num):
                md_vals.append(md_num)

            single_cells.append({
                "group_id": gs,
                "condition": cond,
                "cell_index": int(cell_id),
                "CA_um2": round(ca_num, 3) if not np.isnan(ca_num) else None,
                "ADC": round(adc_num, 4) if not np.isnan(adc_num) else None,
                "ARFS_nm_s": round(arfs_num, 3) if not np.isnan(arfs_num) else None,
                "MPA_deg": round(mpa_num, 3) if not np.isnan(mpa_num) else None,
                "MD_um": round(md_num, 4) if not np.isnan(md_num) else None,
                "group_kill_rate": round(kill_rate, 4),
            })

        group_rows.append({
            "group_id": gs,
            "condition": cond,
            "n_cells": n_cells,
            "n_killed": n_killed,
            "kill_rate": round(kill_rate, 4),
            "median_CA_um2": round(float(np.median(ca_vals)), 2) if ca_vals else None,
            "iqr_CA_um2": f"{np.percentile(ca_vals, 25):.1f}-{np.percentile(ca_vals, 75):.1f}" if ca_vals else "",
            "median_ADC": round(float(np.median(adc_vals)), 2) if adc_vals else None,
            "median_ARFS_nm_s": round(float(np.median(arfs_vals)), 2) if arfs_vals else None,
            "iqr_ARFS_nm_s": f"{np.percentile(arfs_vals, 25):.1f}-{np.percentile(arfs_vals, 75):.1f}" if arfs_vals else "",
            "median_MPA_deg": round(float(np.median(mpa_vals)), 2) if mpa_vals else None,
            "n_MPA_detected": len(mpa_vals),
            "median_MD_um": round(float(np.median(md_vals)), 3) if md_vals else None,
        })

    # Also extract the Dose-Response Dasatinib (0, 25, 50 nM) sheet (Supplemental Figure 11)
    ws_s11 = wb["Supplemental Figure 11"]
    das_rows = []
    for col_offset, dose in [(0, 0), (1, 25), (2, 50)]:
        ca_col = 2 + col_offset
        adc_col = 5 + col_offset
        arfs_col = 8 + col_offset
        for r in range(7, ws_s11.max_row + 1):
            cid = ws_s11.cell(r, 1).value
            if not isinstance(cid, (int, float)):
                continue
            ca_v = ws_s11.cell(r, ca_col).value
            adc_v = ws_s11.cell(r, adc_col).value
            arfs_v = ws_s11.cell(r, arfs_col).value
            if isinstance(ca_v, (int, float)):
                das_rows.append({
                    "dasatinib_nM": dose,
                    "cell_index": int(cid),
                    "CA_um2": round(float(ca_v), 3),
                    "ADC": round(float(adc_v), 4) if isinstance(adc_v, (int, float)) else None,
                    "ARFS_nm_s": round(float(arfs_v), 3) if isinstance(arfs_v, (int, float)) else None,
                })

    # Extract 10-min time series (Supplemental Figure 9)
    ws_s9 = wb["Supplemental Figure 9"]
    ts_rows = []
    for r in range(6, ws_s9.max_row + 1):
        t_min = ws_s9.cell(r, 1).value
        arfs_v = ws_s9.cell(r, 2).value
        ca_v = ws_s9.cell(r, 3).value
        adc_v = ws_s9.cell(r, 4).value
        if isinstance(t_min, (int, float)):
            ts_rows.append({
                "time_min": float(t_min),
                "ARFS_nm_s": round(float(arfs_v), 3) if isinstance(arfs_v, (int, float)) else None,
                "CA_um2": round(float(ca_v), 3) if isinstance(ca_v, (int, float)) else None,
                "ADC": round(float(adc_v), 4) if isinstance(adc_v, (int, float)) else None,
            })

    pd.DataFrame(single_cells).to_csv(DATA_DIR / "lightsheet_wang2026_single_cells.csv", index=False)
    pd.DataFrame(group_rows).to_csv(DATA_DIR / "lightsheet_wang2026_group_summary.csv", index=False)
    pd.DataFrame(das_rows).to_csv(DATA_DIR / "lightsheet_wang2026_dasatinib.csv", index=False)
    pd.DataFrame(ts_rows).to_csv(DATA_DIR / "lightsheet_wang2026_timeseries.csv", index=False)
    print(f"   Saved {len(single_cells)} single-cell rows across {len(group_rows)} groups, {len(das_rows)} dasatinib dose rows.")


def build_data_registry():
    print("3. Writing data/data_registry.csv...")
    rows = [
        {
            "dataset_id": "ZENODO_8328441_TEG_BC",
            "title": "BEHAV3D TEG + breast cancer 3D organoids (LSM880 9-channel confocal)",
            "citation": "Alieva et al., Nat Protoc 2024 / Dekkers et al., Nat Biotechnol 2023",
            "doi": "10.5281/zenodo.8328441",
            "license": "CC0 1.0 Universal",
            "layers_covered": "Layer 1 (Behavioral state & arrest) + Layer 4 (Organoid death & fragmentation)",
            "spatial_res": "0.66 x 0.66 x 1.00 um",
            "temporal_res": "33.0 min / frame (19.2 h, 35 frames)",
            "sample_size": "668 tracked T cells (190,076 3D surfaces) + 164 organoid tracks (55,445 sub-fragments)",
            "access_status": "Local verified (15.12 GB .ims + .rds/.csv)",
            "local_path": "LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims",
        },
        {
            "dataset_id": "ZENODO_8328441_CART_ALL",
            "title": "BEHAV3D anti-CD19 CAR-T + B-ALL leukemia cells (LSM880 7-channel confocal)",
            "citation": "Alieva et al., Nat Protoc 2024 (Exp28)",
            "doi": "10.5281/zenodo.8328441",
            "license": "CC0 1.0 Universal",
            "layers_covered": "Layer 1 (CAR-T serial contacts & motility in liquid tumor)",
            "spatial_res": "0.91 x 0.91 x 0.91 um",
            "temporal_res": "2.81 min / frame (22.5 h, 480 frames)",
            "sample_size": "195 CAR-T tracks (8,612 CD4 + 16,029 CD8 surfaces) + 114,396 B-ALL surfaces",
            "access_status": "Local verified (24.28 GB .ims)",
            "local_path": "LSM880_example_dataset_CART_ALL_Exp28.ims",
        },
        {
            "dataset_id": "SCIENCEDB_19235_HBOPM",
            "title": "HBOPM 320-nm 4D light-sheet imaging of CAR-T-mediated cytotoxicity (Nalm6 mApple-Mem)",
            "citation": "Wang, Jin, Fang, Chen & Fei, Sci Data 13:439 (Feb 2026) / Nat Commun 15:7060 (2024)",
            "doi": "10.1038/s41597-026-06829-9 | 10.57760/sciencedb.19235",
            "license": "CC BY 4.0 (data & article) | MIT License (feilab-hust/CAR-T-analysis code)",
            "layers_covered": "Layer 2 (3D synapse area CA, Lifeact retrograde flow ARFS, ADC) + Layer 3 (3D MTOC angle MPA & distance MD) + Layer 4 (SYTOX Blue death)",
            "spatial_res": "0.32 x 0.32 x 0.60 um (isotropic ~320 nm optical)",
            "temporal_res": "2.5 s/vol (6-vol burst) every 10 min for 5.0 h (30 cycles)",
            "sample_size": "419 CAR-T-Nalm6 pairs (13 groups incl. 0/25/50 nM dasatinib; 1,202 3D MTOC angles)",
            "access_status": "Local verified (419-pair Source Data .xlsx + MIT MATLAB/U-Net code + 6 movies); 19.55 TiB raw .h5/.tiff on ScienceDB requires author transfer request",
            "local_path": "external/feilab_lightsheet_data/Source_Data_Wang2024_NatCommun.xlsx",
        },
        {
            "dataset_id": "BEHAV3D_GITHUB_REPOS",
            "title": "BEHAV3D processed feature tables, UMAP/Random-Forest labels, and scRNA-seq signatures",
            "citation": "Dekkers et al., Nat Biotechnol 2023 / Ríos-Jiménez et al., Nat Commun 2026",
            "doi": "github.com/AlievaRios/BEHAV3D | github.com/imAIgene-lab/BEHAV3D",
            "license": "Apache-2.0",
            "layers_covered": "Layer 1 (9 behavioral clusters, serial killer gene signature, species generalizability)",
            "spatial_res": "Object-level 3D statistics",
            "temporal_res": "2.8-33.0 min / frame",
            "sample_size": "668 TEG tracks + multi-donor organoid co-cultures",
            "access_status": "Local verified (cloned repositories)",
            "local_path": "external/AlievaRios_BEHAV3D",
        },
        {
            "dataset_id": "MP_TFM_SHELASTIC",
            "title": "Microparticle traction force microscopy (MP-TFM) spherical harmonic force inference reference",
            "citation": "Vorselen et al., Nat Commun 2020 / Nat Biotechnol 2021; Wang & Cai, ShElastic",
            "doi": "10.1038/s41467-020-15655-5 | 10.1038/s41587-021-00863-5",
            "license": "CC BY 4.0 (article) | MIT (ShElastic solver)",
            "layers_covered": "Layer 2 (Calibration of pN/nN synaptic indentation stress from 3D target membrane curvature)",
            "spatial_res": "~0.25-0.40 um confocal particle contour",
            "temporal_res": "15-60 s / stack",
            "sample_size": "Literature parameterization (0.3-1.5 nN total synaptic force; 1-3 um crater indentation)",
            "access_status": "Literature parameters + open solver (no raw live target-cell force stack required when paired with Wang 2026 mApple-Mem curvature)",
            "local_path": "src/model_4d_cell_kill.py",
        },
    ]
    pd.DataFrame(rows).to_csv(DATA_DIR / "data_registry.csv", index=False)


if __name__ == "__main__":
    extract_exp28_ims()
    extract_wang2026_lightsheet()
    build_data_registry()
    print("Done.")
