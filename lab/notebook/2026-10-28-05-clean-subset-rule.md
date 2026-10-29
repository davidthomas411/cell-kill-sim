---
id: 2026-10-28-05-clean-subset-rule
date: "2026-10-28"
title: "Gate G1 clean-cell subset rule fixed from mask properties before outcomes (UNVERIFIED)"
gate: G1
status: draft
author: "agy (backfilled)"
reviewers:
  - claude
signoff: null
code_commit: "71251e7, a0b0a94"
datasets:
  - lab_masks
  - qc_flags
  - clean_subset_v0.1
protocols:
  - clean-cell-rule@v1
  - gallery-review@v1
galleries:
  - review-2026-10-28
  - clean-2026-10-28
supersedes: null
superseded_by: null
---

## Question

Can we define an objective, pre-outcome mask quality rule (`clean-cell-rule@v1`) from mask properties alone (`data/qc/per_visit_qc.csv`, without consulting death labels or associations) that selects contact visits where both the tracked T cell and contacted target are single, whole, un-truncated cells, without biasing cup depth by filtering on target solidity or sphericity?

** Verification Notice: Both `qc_flags` and `clean_subset_v0.1` are UNVERIFIED (`status: unverified` in `lab/registry.csv`) until gallery `clean-2026-10-28` is reviewed panel by panel with raw membrane overlays.**

## What was run

```bash
python3 src/mask_qc.py
```

Commits:
- `71251e7` (`Clean-cell subset criteria, fixed from mask properties before any outcome`)
- `a0b0a94` (`Figure: clean-cell rule failures by experiment, cup depth clean vs rejected`)

## What was looked at

- `status/clean_subset_criteria.md` and `figures/qc_clean_subset_overview.png` (`commit a0b0a94`).
- Gallery `clean-2026-10-28` (`figures/review_clean/`, being rendered by `src/review/make_clean_gallery.py` with raw 561-nm membrane and deposited U-Net classes 0–4 on stratified random clean visits, extreme clean visits, and 40 rejected visits across the 5 failure rules).

## Observations

1. **Five pre-outcome criteria (`clean-cell-rule@v1`)**:
   - (1) `target_n_components == 1` and `tcell_n_components == 1`
   - (2) `target_raw_largest_frac >= 0.9`
   - (3) `touches_fov_edge == False`
   - (4) `200 <= target_volume_um3 <= 1500`
   - (5) `0.7 <= target_vol_ratio_to_own_median <= 1.4`
   - Target solidity/sphericity is deliberately excluded because a real synaptic indentation lowers sphericity (`Spearman rho = -0.26` with cup depth).
2. **Subset yield from masks (`UNVERIFIED`)**:
   - `1,213` of `4,523` `trace_num == 1` contact visits (`26.8%`) pass all five criteria across `243` chambers (`111` chambers with `>= 4` clean visits).
   - Clean fraction by experiment: `v01` `0.28`, `v02` `0.34`, `v03` `0.34`, `v04` `0.40`, `v05` `0.29`, `v06` `0.19`, `v07` `0.16`, `v08` `0.38`, `v09` (`hd10`) `0.05`, `v10` `0.31`, `v11` (`hd2_ds`) `0.04`, `v12` (`hd10ds`) `0.16`, `v13` (`hd5_ds`) `0.14`.
3. **Consequences for downstream analyses**:
   - The candidate clean subset supports within-experiment analyses in `v01`–`v05`, `v08`, and `v10`, but leaves donor `hd10` (`18` untreated visits) and pooled dasatinib (`60` visits) too thin to support the 5-vs-5 donor comparison or the dasatinib control on its own.

## Decision

 Criteria locked in `status/clean_subset_criteria.md` and `lab/protocols/clean-cell-rule-v1.md`. Dataset `clean_subset_v0.1` remains **UNVERIFIED** (`status: unverified`) until every panel of gallery `clean-2026-10-28` is reviewed against raw 561-nm membrane images.

## Open issues

- Complete panel-by-panel review of `clean-2026-10-28` in `tools/review_app.py`.
- Quantify per-experiment failure rates by layer: (a) U-Net segmentation failure while raw cell is whole, (b) extraction failure while U-Net labels are sound, and (c) biological blebbing/fragmentation of the raw cell itself.
