---
id: clean-cell-rule
version: v1
date: "2026-10-28"
status: draft
---

# Protocol: Gate G1 Clean-Cell Subset Rule (`clean-cell-rule-v1`)

## Purpose

Filter `trace_num == 1` contact visits (`n = 4,523` across 13 experiments) using strictly pre-outcome mask and geometry quality criteria (`data/qc/per_visit_qc.csv`, `src/mask_qc.py`) so that downstream cup-depth (`G3`) and killing-association (`G4`) analyses run only on visits where the tracked T cell and its contacted target are each one whole, separate, un-truncated cell.

No death labels (`step4`) or outcome associations (`step5b`, `step7`) were consulted when defining these criteria.

## 1. Inclusion Criteria (Per Contact Visit, `trace_num == 1`)

A contact visit is classified as **clean** (`is_clean_visit == True`) if and only if all five conditions hold simultaneously:

1. **Single connected extracted objects**:
   `target_n_components == 1` and `tcell_n_components == 1`
   Both our extracted contacted target mask and our extracted tracked T-cell mask consist of a single 26-connected 3D component.
2. **Cohesive deposited U-Net target mask**:
   `target_raw_largest_frac >= 0.9`
   In the Fei lab's raw deposited U-Net predictions (`(pred == 2) | (pred == 4)` within the contacted target region), the largest connected component contains at least 90% of the target voxels. This rejects visits where the U-Net itself segmented the target into disconnected fragments.
3. **Not truncated by the field-of-view boundary**:
   `touches_fov_edge == False`
   Neither the tracked T cell nor the contacted target touches the X, Y, or Z boundary of the 3D ROI crop.
4. **Plausible single-cell Nalm6 target volume**:
   `200 <= target_volume_um3 <= 1500`
   Target volume at nominal voxel pitch (`0.1962 um`) lies between `200 um^3` (rejecting tiny subcellular debris fragments such as `dasatinib_16` at `37 um^3` or `dasatinib_13` at `154 um^3`) and `1,500 um^3` (rejecting two-target merges such as `cup_01` at `2,821 um^3` and `cup_03` at `2,550 um^3`).
5. **Chamber-level target volume stability**:
   `0.7 <= target_vol_ratio_to_own_median <= 1.4`
   The target's volume at the current visit is within `0.7x` to `1.4x` of the median target volume across visits in the same chamber, rejecting sudden single-visit merges or splits with neighbouring cells.

## 2. Deliberately Excluded Criteria

- **Target solidity and sphericity (`target_sphericity`) are NOT used as filters.**
  A genuine deep synaptic indentation lowers target solidity and sphericity (`Spearman rho = -0.26` with `cup_depth_hull_p95_um` and `-0.34` with `cup_depth_over_radius`). Filtering on sphericity or solidity would selectively discard real deep cups and bias the primary biological quantity under study.

## 3. Resulting Cohort Size (from Masks Only)

- **Overall**: `1,213` of `4,523` `trace_num == 1` contact visits (`26.8%`) pass all five criteria; `243` chambers have `>= 1` clean contact visit and `111` chambers have `>= 4` clean contact visits.
- **Clean fraction by experiment**:
  - Untreated high-killing: `v01` (`hd1`) `0.28`, `v02` (`hd3`) `0.34`, `v03` (`hd4`) `0.34`, `v04` (`hd6`) `0.40`, `v05` (`hd7`) `0.29`
  - Untreated low-killing: `v06` (`hd8`) `0.19`, `v07` (`hd9`) `0.16`, `v08` (`hd2`) `0.38`, `v09` (`hd10`) `0.05`, `v10` (`hd5`) `0.31`
  - Dasatinib 50 nM: `v11` (`hd2_ds`) `0.04`, `v12` (`hd10ds`) `0.16`, `v13` (`hd5_ds`) `0.14`
- **Analytical scope**: The clean subset provides sufficient sample size for within-experiment analyses in `v01`–`v05`, `v08`, and `v10`, but leaves donor `hd10` (`18` untreated clean visits) and pooled dasatinib (`60` clean visits) too thin to support the 5-vs-5 donor comparison or the dasatinib control on their own without mask repair or re-segmentation from raw volumes.

## 4. Verification Requirement (`G1`)

This subset remains `status: draft` (`unverified` in `lab/registry.csv`) until the `clean-2026-10-28` gallery (`figures/review_clean/`, containing stratified random clean visits, extreme clean visits, and 40 rejected visits across the 5 failure rules with raw 561-nm membrane images under every panel) is reviewed panel by panel under `gallery-review@v1`.
