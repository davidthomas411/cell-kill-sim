# Open Questions for the Fei Lab (`lab/handbook/open-questions.md`)

Technical questions regarding the ScienceDB `19235` deposition and the analysis pipeline in Wang et al. (*Nat. Commun.* 2024 / *Sci. Data* 2026):

## 1. Post-Deskew Axial Voxel Pitch (`dz`)

- In `summary.m` (line 18), contact area is scaled by `0.0385 um^2` per voxel face (`sqrt(0.0385) = 0.1962 um` lateral pitch), and `ARFS` is scaled by `19.6 nm/s`.
- Across all uncontacted Nalm6 target cells in `*_predictions.h5`, the 3D bounding-box ratio is `Delta Z / Delta XY = 0.807` (median `Z` span ~19% fewer slices than `X` or `Y` pixels).
- **Question**: After light-sheet deskewing and rotation into the `(Z, 256, 256)` ROI crop, is the true axial voxel step `dz = 0.1962 um` (implying targets are physically oblate on the microgrid floor) or `dz ≈ 0.240–0.243 um`?

## 2. Un-deposited ROIs in Group 7 (`v07_6_39_120_0-6_hd9`)

- Across 12 of the 13 experimental groups (`v01`–`v06`, `v08`–`v13`), our port of `synapse_tracking.m` (`trace_num == 1` in `src/reproduce_contact_area_tracked.py`) matches the published row count in `summary.csv` / `Source_Data` exactly (`395 / 395` rows across 12 groups).
- In Group 7 (`v07_6_39_120_0-6_hd9`), the published table contains `39` rows, whereas ScienceDB `19235` contains only `22` ROI folders (`MIP_ROI_0`..`MIP_ROI_21`, yielding `15` chambers with `trace_num == 1`, a deficit of `24` rows).
- **Question**: Were additional ROIs (`MIP_ROI_22`..`MIP_ROI_55`) imaged for `v07_6_39_120_0-6_hd9` that were omitted from the ScienceDB `19235` upload?

## 3. U-Net Segmentation Quality in Donor `hd10` (`v09` and `v12`) and Dasatinib (`v11`)

- Panel-by-panel visual review (`figures/review/`) and pre-outcome QC (`src/mask_qc.py`) show that while `28–40%` of contact visits in `v01`–`v05`, `v08`, and `v10` have single, solid, un-truncated cell masks, only `5%` of visits in `v09_6_40_120_0-6_hd10` (`18` visits) and `4%` in `v11_6_36_120_0-6_hd2_ds` pass the same single-cell criteria; most target masks in `v09` and `v12` are fragmented into multiple pieces or hollow shells.
- **Question**: Did `v09`, `v11`, and `v12` have lower raw 561-nm mApple-Mem expression or different illumination settings that degraded the 3D U-Net predictions (`classes 2, 4`), and were any manual QC exclusions applied in `Figure 5`?

## 4. `MIP_ROI_0` Late 405-nm Left-Edge Saturation Band (`visit >= 19`)

- In every one of the 13 experiments (`39 / 39` visits at `visit >= 19`, `t >= 190 min`), `MIP_ROI_0` (and no other ROI) displays a vertical saturation band along the left edge (`X = 0..W-1`, peaking at `14,505–25,317 ADU` vs `< 3,000 ADU` in all other `10,471` chamber-visits).
- **Question**: Does `MIP_ROI_0` sit at the physical edge of the microgrid chip where the 405-nm excitation sheet or stage homed at `t >= 190 min`?

## 5. 405-nm Dead-Dye Threshold Calibration (`398 ADU`) and `Figure 5g` Kill Counts

- In `step2_instance_seg.m` (line 45), the 3D dead-dye mask thresholds the `[3,3,1]` Gaussian-smoothed 405-nm volume at `> 398 ADU`, while camera background median is `~396–400 ADU` and clearly dead targets read `410–490 ADU`.
- Furthermore, in `dead_count.m`, any overlap with the 10-voxel dilated dye mask (`d_dil10_3d`) can be triggered by a dead bystander cell within `2 um` of the contacted target.
- **Question**: Were the per-condition kill fractions in `Figure 5g` (`57.5%` high-killing, `27.0%` low-killing, `4.5%` drug-treated) taken directly from `summary.m` (`C = max(B, [], 2)` on `dead_count.m`), or were false kills from adjacent bystander cells and `MIP_ROI_0` curated manually?
