---
id: death-calling
version: v1
date: "2026-10-28"
status: draft
---

# Protocol: Target Cell Death Calling (`death-calling-v1`)

## Purpose

Determine per-visit and per-chamber target cell death labels and death times across all 502 chambers (`10,510` chamber-visits in 13 experiments) from deposited 2D 405-nm SYTOX Blue maximum-intensity projections (`*_MIPxy_405.tif`, `*_MIPxz_405.tif`) and 3D U-Net segmentation masks (`*_predictions.h5`), as currently implemented in `src/death_labels.py` (v2).

**Status**: `draft` (unverified; blocked at Gate `G2` pending gallery/label mismatch resolution, 3D vs 2D neighbour-overlap verification, and threshold sensitivity analysis).

## 1. Reference 3D Rule (Fei Lab MATLAB Code)

In `external/feilab_CART_analysis/main/step2_instance_seg.m` and `analysis/summary/dead_count.m`:

1. Raw 3D Channel 0 (`405 nm` SYTOX dead-cell dye, `uint16`) is smoothed with an anisotropic 3D Gaussian filter (`sigma = [3, 3, 1]` voxels in `(Y, X, Z)`, kernel size `[7, 7, 3]`, replicate padding) and thresholded at `> 398.0 ADU`.
2. The binary mask is closed with a 6-connected radius-3 sphere (`strel('sphere', 3)`), area-opened to keep 26-connected components with `>= 20` voxels (`bwareaopen(..., 20, 26)`), and dilated by radius `10` voxels (`d_dil10_3d`).
3. A segmented target instance `b_inst == i_id` that touches the tracked T-cell synaptic contact surface (`s_mask & sphere_dilate_edt(bi, 1.0)`) is called dead at that sweep if `any(bi & d_dil10_3d)` is non-empty.
4. Verified on all 20 local 3D raw volumes (`data/step4/calibration_20_raw_volumes.csv`): deposited 2D MIPs satisfy `MIPxy_405 == raw[0].max(axis=0).T` and `MIPxz_405 == raw[0].max(axis=1).T` with `0 ADU` maximum difference.

## 2. Current 2D Projection Implementation (`src/death_labels.py` v2)

Because 3D raw `.h5` volumes were initially available for only 20 sweeps whereas 2D 405-nm MIPs (`*_MIPxy_405.tif`, `*_MIPxz_405.tif`) are deposited for all `10,510` visits, `src/death_labels.py` implements a 2D projection rule:

### Step 2.1: Contacted Target Instance Extraction (v2 update)
- On sweeps `1..5` of each visit, 3D target instances are segmented via `instance_seg_matlab_xyz(pred == 2, pred == 4)` (`bwareaopen(nucleus, 3000, 26)` + Voronoi membrane assignment in MATLAB `(X, Y, Z)` order + 26-connected hole fill).
- For the `416` chambers with a single tracked T cell (`trace_num == 1` from `src/reproduce_contact_area_tracked.py`), only the target instance(s) contacted by Track 1 at that visit are retained. If no contact occurs on that visit, the chamber's primary contacted target instance (nearest centroid to the tracked target trajectory) is projected.
- For the `86` chambers with `trace_num != 1`, the target instance contacted by any T cell is used and flagged (`included_by_trace_rule = False`).
- The 3D contacted target mask is projected to match the MIP transpose convention:
  - `b_xy = target_3d.max(axis=0).T`
  - `b_xz = target_3d.max(axis=1).T`
- *(Legacy v1 columns `*_all_targets_v1` projected all raw target voxels `(pred == 2) | (pred == 4)` without instance filtering).*

### Step 2.2: `MIP_ROI_0` Late Stage-Origin Saturation Band Exclusion
- In all `13 / 13` experiments, `MIP_ROI_0` at `visit >= 19` (`t >= 190 min`, `39 / 39` visits) contains a left-edge vertical 405-nm saturation band across columns `0..W-1` (`W in [31, 121] px`, column-0 median `548–915 ADU`, peak `14,505–25,317 ADU`).
- Any visit satisfying `(np.median(mip_xy[:, 0]) > 404.0) and (mip_xy.max() > 5000)` is flagged as `roi0_artifact_flag = True` and excluded from triggering a positive death call (`9` artifact-only false-positive chamber deaths removed).

### Step 2.3: 2D Dead-Dye Mask (`compute_2d_dead_mask`)
- Applied independently to `mip_xy` and `mip_xz`:
  1. 2D Gaussian filter (`scipy.ndimage.gaussian_filter`, `sigma = 3.0`, `mode = 'nearest'`, `truncate = 2.0`).
  2. Threshold strictly above `THRESH_2D_ADU = 404.0 ADU` (compensating for the `+4.6` to `+5.5 ADU` extreme-value shift of taking the maximum over `Z = 166–190` or `Y = 256` slices relative to the 3D threshold `398.0 ADU`).
  3. 2D binary closing with a radius-3 disk (`disk(3)`).
  4. 8-connected area opening keeping components with `>= MIN_COMPONENT_AREA_2D = 20` pixels.

### Step 2.4: Per-Visit and Per-Chamber Death Calls
- **Per-visit positive call (`dead_visit`)**:
  `overlap_xy_px = sum(d_xy & b_xy) >= 20` AND `overlap_xz_px = sum(d_xz & b_xz) >= 20` (and not `roi0_artifact_flag`).
- **Per-chamber definitions** (computed over both the **Full Recording** of `20–22` visits and the **Fixed 200-Minute Window** of `visits 1–20`):
  - **Any-Visit rule (`ever_dead_any_visit_200m`)**: `>= 1` positive visit (matching `summary.m` `C = max(B, [], 2)`). `death_visit` is the first positive visit (`death_time_min = death_visit * 10.0`).
  - **Sustained rule (`ever_dead_200m`)**: `>= 2` consecutive positive visits. `death_visit` is the first visit of the first sustained run.

## 3. Open Issues Blocking Gate `G2`

1. **Gallery vs table mismatch (`label_mismatch`)**: In `figures/review/death_labels/`, panels rendered raw `(pred == 2) | (pred == 4)` outlines while the v2 table used contacted target instances, causing `death_surviving_19`, `death_surviving_24`, and `death_killed_07` to disagree between panel overlay text and chamber label.
2. **Possible overlap discrepancy**: `death_dasatinib_30` visit 5 reports `27 px` overlap while the dye contour appears outside the target outline in both views.
3. **2D projection neighbour overlap (`neighbour_overlap`)**: A dye-positive bystander cell lying above/below or behind the contacted target can overlap in both XY and XZ projections (`death_surviving_17`, `death_killed_05`, `death_killed_09`).
4. **Narrow threshold margin**: Background median is `~400 ADU` vs threshold `404.0 ADU`; borderline calls (`death_killed_01` at `404.3 ADU`) require threshold sensitivity curves across `403–410 ADU` and `20–50 px`.
