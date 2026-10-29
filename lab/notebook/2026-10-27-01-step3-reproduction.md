---
id: 2026-10-27-01-step3-reproduction
date: "2026-10-27"
title: "Step 3: Reproduce the Fei lab's published chamber counts, contact area, and actin metrics"
gate: G0
status: draft
author: "agy (backfilled)"
reviewers: []
signoff: null
code_commit: "e1330b3, 5b373cd, 72c4f0c, d3c2408, ba2d963"
datasets:
  - lab_masks
  - raw_volumes
  - step3_reproduction
protocols: []
galleries: []
supersedes: null
superseded_by: null
---

## Question

Can we reproduce the Fei lab's published per-chamber row counts (`summary.csv` / `Source_Data`, `n = 419` across 13 groups), per-chamber peak synaptic contact area (`CA`), and raw 488-nm actin depletion (`ADC`) and retrograde flow (`ARFS`) directly from the deposited ScienceDB `19235` 3D U-Net masks (`*_predictions.h5`) and local 3D raw volumes?

## What was run

```bash
python3 src/reproduce_contact_area.py --all-experiments --workers 6
python3 src/reproduce_actin_measures.py
python3 src/reproduce_contact_area_tracked.py --all-experiments --workers 6
python3 src/map_chambers_to_paper_rows.py
```

Commits:
- `e1330b3` (`Reproduce the lab's contact area from deposited masks (group 1)`)
- `5b373cd` (`Map chambers to paper rows for every processed group`)
- `72c4f0c` (`Reproduce the lab's actin flow and actin depletion exactly (2 chambers)`)
- `d3c2408` (`Chamber-to-row map for all 13 groups`)
- `ba2d963` (`Step 3: port lab T-cell tracker; single-track rule reproduces paper chamber selection`)

## What was looked at

- External comparison against published summary tables (`external/feilab_CART_analysis/main/analysis/summary/summary.csv`) across all 13 groups (`502` deposited chambers, `52,545` sweeps).
- Diagnostic figures `figures/step3_contact_surface_sanity_slice.png`, `figures/step3_contact_area_reproduction.png`, `figures/step3_tracked_vs_masks_only.png`, and `figures/step3_actin_reproduction.png`.
- No stage-gate gallery existed at the time of this run (visual review of the underlying masks was subsequently assigned to Gate `G1` on gallery `review-2026-10-28`).

## Observations

1. **Single-track chamber selection (`trace_num == 1`, commit `ba2d963`)**: Porting the lab's 3D T-cell centroid tracker (`synapse_tracking.m`) in MATLAB `(X, Y, Z)` dimension order yields `416` chambers with `trace_num == 1` (`25` with `0`, `56` with `2`, `5` with `3` across `502` chambers). In all `12 / 12` completely deposited experimental groups (`v01`–`v06`, `v08`–`v13`), `trace_num == 1` matches the paper's published row count (`395 / 395` rows, `0` difference in every group). Group 7 (`v07_6_39_120_0-6_hd9`) has `22` deposited ROIs (`15` with `trace_num == 1`) vs `39` rows in the paper (`24` ROIs not in the ScienceDB upload).
2. **Masks-only contact area (`CA`, commits `e1330b3`, `d3c2408`, `ba2d963`)**: Excluding the peripheral radius-5 rim (`s_edge = -j` in `step2_instance_seg.m` lines 89–90) and matching MATLAB `(X, Y, Z)` `bwdist` tie-breaking maps `415 / 419` paper rows firmly/probably (`395 / 395` in the 12 complete groups), with masks-only `CA` within `0.5%` of the published number for `369 / 419` rows (`median |rel_err| = 0.08%`).
3. **Exact reproduction with raw 488-nm actin (`commit 72c4f0c`)**: On the two Group 1 chambers with local 3D raw volumes (`v01/MIP_ROI_32` and `v01/MIP_ROI_2`), running the full `cl_factor.m` actin-slope component selection and `optical_flow.m` reproduces all six published numbers (`CA = 129.114` and `93.547 um^2`, `ADC = 0.1807` and `1.0965`, `ARFS = 53.877` and `137.110 nm/s`) to `0.00%` error.

## Decision

Gate `G0` (external numerical reproduction of the Fei lab's pipeline and published tables) **passed**. `step3_reproduction` is marked `verified` by external check in `lab/registry.csv`. However, matching the lab's numbers does not validate the 3D U-Net masks themselves (since the lab used the same masks); visual verification of cell identity is required under Gate `G1`.

## Open issues

- `24` rows of Group 7 (`v07_6_39_120_0-6_hd9`) are absent from the ScienceDB `19235` deposition (logged in `lab/handbook/open-questions.md`).
- Visual quality of the underlying U-Net masks (`lab_masks`) must pass Gate `G1` before downstream geometric measurements are trusted.
