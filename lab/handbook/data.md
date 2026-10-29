# Dataset Layers (`lab/handbook/data.md`)

All dataset layers in this project derive from the Fei lab's light-sheet single-cell CAR-T / Nalm6 cytotoxicity dataset deposited at ScienceDB (`19235`; Wang et al., *Nat. Commun.* 2024 / *Sci. Data* 2026) and tracked in `lab/registry.csv`.

## 1. Experimental Cohort Overview

- **13 experiments (`v01`–`v13`)** spanning three biological regimes:
  - **Untreated high-killing CAR-T (`5` donors, Groups 1–5)**: `v01_6_40_120_0-6_hd1`, `v02_6_40_120_0-6_hd3`, `v03_6_40_120_0-6_hd4`, `v04_6_40_120_0-6_hd6`, `v05_6_40_120_0-6_hd7` (`200` chambers total; `167` with `trace_num == 1`).
  - **Untreated low-killing CAR-T (`5` donors, Groups 6–10)**: `v06_6_40_120_0-6_hd8`, `v07_6_39_120_0-6_hd9` (`22` deposited ROIs), `v08_6_40_120_0-6_hd2`, `v09_6_40_120_0-6_hd10`, `v10_6_40_120_0-6_hd5` (`182` chambers total; `160` with `trace_num == 1`).
  - **Dasatinib 50 nM CAR-T (`3` paired donors `hd2, hd10, hd5`, Groups 8\*–10\*)**: `v11_6_36_120_0-6_hd2_ds`, `v12_6_35_120_0-6_hd10ds`, `v13_6_40_120_0-6_hd5_ds` (`120` chambers total; `89` with `trace_num == 1`).
- **Temporal sampling**: `502` total microgrid chambers (`MIP_ROI_*`), imaged for `20–22` visits at `10 min` intervals (`200–220 min` total) with `5` actin sweeps per visit at `15 s` intervals (`10,510` complete chamber-visits; `52,545` complete sweeps; `63,053` total deposited sweep files including partial ROIs).

## 2. Registered Dataset Layers (`lab/registry.csv`)

| Dataset ID | Layer | Path Pattern | Items | Gate | Current Status |
|---|---|---|---:|---|---|
| `lab_masks` | Raw segmentation | `data/scidb_19235_masks/<exp>/<roi>/<stem>_predictions.h5` | 63,053 | G1 | `unverified` |
| `raw_volumes` | Raw microscopy | `data/scidb_19235_raw/<exp>/<roi>/<stem>.h5`; `data/scidb_19235_chambers/*/<stem>.h5`; `data/scidb_19235_mips/<exp>/<roi>/<stem>_MIP*_405.tif` | 950 (3D) + 21,036 (MIPs) | G1/G2 | `unverified` (partial 3D download) |
| `step3_reproduction` | Step 3 reproduction | `data/step3_tracked/per_chamber_all.csv`; `data/step3_actin/per_chamber_actin_summary.csv`; `results/chamber_to_row_map_tracked.csv` | 502 | G0 | `verified` (external check) |
| `step4_death_labels_v2` | Step 4 death labels | `data/step4/per_chamber_death_labels_all_v2.csv`; `data/step4/per_visit_death_labels_all_v2.csv` | 502 | G2 | `unverified` |
| `step5_descriptors` | Step 5a descriptors | `data/step5/per_chamber_descriptors.csv`; `data/step5/per_visit_descriptors.csv`; `data/step5/per_sweep_descriptors.csv` | 502 | G3 | `unverified` |
| `qc_flags` | Mask QC | `data/qc/per_visit_qc.csv` | 4,523 | G1 | `unverified` |
| `clean_subset_v0.1` | Clean subset | `data/clean_subset/clean_visits.csv` | 1,213 | G1 | `unverified` |
| `step5b_associations` | Step 5b associations | `data/step5b/test1_cox_death.csv`; `data/step5b/summary_step5b.json` | 366 | G4 | `unverified` |
| `step6b_fits` | Step 6b mechanics | `data/step6b/visit_fits.csv`; `data/step6b/gateB_visit_fits.csv` | 3,475 | G5 | `rejected` (void) |
| `step7_focused_pushing` | Step 7 focused pushing | `data/step7/summary.json`; `data/step7/test7*.csv` | 365 | G4 | `unverified` |

## 3. Layer Details

1. **`lab_masks` (`*_predictions.h5`)**:
   3D `uint8` array `/predictions` of shape `(Z, 256, 256)` (`Z in [161, 195]`) output by the Fei lab's 3D U-Net:
   - `0`: background
   - `1`: CAR-T cell interior / nucleus
   - `2`: Nalm6 target cell interior / nucleus
   - `3`: CAR-T cell membrane / cortex
   - `4`: Nalm6 target cell membrane / cortex
2. **`raw_volumes` (`*.h5` and `*_MIP*_405.tif`)**:
   - 3D raw `.h5` files store `/data` of shape `(3, Z, 256, 256)` in `uint16`: Channel 0 = `405 nm` SYTOX Blue dead-cell dye, Channel 1 = `488 nm` Lifeact-eGFP F-actin, Channel 2 = `561 nm` mApple-Mem target membrane.
   - 2D MIPs (`*_MIPxy_405.tif`, `*_MIPxz_405.tif`) are exact `uint16` projections (`raw[0].max(axis=0).T` and `raw[0].max(axis=1).T`) at sweep 1 of each visit (`10,518` XY and `10,518` XZ images).
3. **`step3_reproduction`**:
   Exact Python port of the Fei lab's instance segmentation (`step2_instance_seg.m`), contact surface extraction (`cl_factor.m`), T-cell trajectory tracker (`synapse_tracking.m` -> `trace_num`), and 2-chamber raw actin depletion / retrograde flow (`ADC`, `ARFS`). Passed Gate `G0` external check against the published paper tables.
4. **`step4_death_labels_v2`**:
   Per-visit and per-chamber target death calls using the contacted target instance (`trace_num == 1` Track 1) and excluding the `MIP_ROI_0` `visit >= 19` saturation band. Blocked at Gate `G2`.
5. **`step5_descriptors`**:
   33 macro-scale (`>= 1.4 um`) 3D geometric descriptors of the contact patch, target convex-hull / reference-sphere cup indentation, and T-cell protrusion, computed at `z = 0.1962 um` and `z = 0.2400 um`. Blocked at Gates `G1`/`G3`.
6. **`qc_flags` & `clean_subset_v0.1`**:
   Pre-outcome mask quality metrics (`src/mask_qc.py`) and the 1,213-visit candidate clean subset defined in `clean-cell-rule@v1`. Pending visual verification on gallery `clean-2026-10-28`.
