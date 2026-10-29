---
id: 2026-10-28-02-step6b-runs-void
date: "2026-10-28"
title: "Step 6b: Two-shell mechanical model runs and Gate A/B checks — VOID / UNVERIFIED"
gate: G5
status: draft
author: "agy (backfilled)"
reviewers:
  - claude
signoff: null
code_commit: "40486f6, 62770f3, 6d76475"
datasets:
  - lab_masks
  - step5_descriptors
  - step6b_fits
protocols:
  - cup-measurement@v1
  - gallery-review@v1
galleries:
  - review-2026-10-28
supersedes: null
superseded_by: null
---

## Question

Does a coupled two-shell cortical mechanics solver (`src/mech/shell_model.py`, fitting dimensionless active push $\Pi = p_{\text{act}} R_{\text{target}} / T_{\text{target}}$ per contact visit) pass analytical/numerical validation and beat proportional geometric baselines on held-out shapes (`status/step6b_heldout_design.md`)?

** Verification Notice: All Step 6b model runs (`data/step6b/`) are VOID / REJECTED (`status: rejected` in `lab/registry.csv`) and parked at Gate `G5`.**

## What was run

```bash
python3 src/mech/run_step6b.py --workers 6
python3 src/mech/run_gateA_check.py --workers 6
python3 src/mech/run_gateB_check.py --workers 6
```

Commits:
- `40486f6` (`Step 6b: write down model and held-out test before building`)
- `62770f3` (`Plan: step 6b first run rejected on review, rerun in progress`)
- `6d76475` (`Step 6b: two-shell solver with live contact, converged solves, validation benchmarks; addendum before final held-out run`)

## What was looked at

- Pre-specified held-out design and Gate A/B addenda in `status/step6b_heldout_design.md`.
- Reports `.agy_step6b_report.md`, `.agy_step6b_gateA.md`, `.agy_step6b_gateB.md`, and `data/step6b/gateB.json`.
- Real-data visual review gallery `review-2026-10-28`:
  - `fit_reachable`: `fit_reach_01`..`16`
  - `fit_unreachable`: `fit_unreach_01`..`16`
  - `initial_overlap`: `overlap_01`..`12`
  - `mesh_size`: `mesh_size_01`..`06`

## Observations

1. **First run (`commit 40486f6` -> `62770f3`) rejected on code review**:
   - The T-cell shell was not included as a live contacting body during optimization (`kappa_c = 0`), L-BFGS-B was capped at `maxiter = 60` (`0%` convergence), and `84%` of visits saturated at the grid ceiling `Pi = 4.0`.
2. **Gate A run (`.agy_step6b_gateA.md`) failed coverage gate**:
   - Enforcing live two-shell contact (`kappa_c = 50`) and full L-BFGS-B convergence reduced maximum attainable indentation on concave pre-contact meshes to `~1.0–1.5 um`, leaving `69.9%` of visits at the `Pi = 12` ceiling (`30.1%` in-range vs `>= 95%` requirement).
3. **Gate B run (`commit 6d76475`, `.agy_step6b_gateB.md`) passed synthetic benchmarks (`1.63%` error vs analytical spherical cap; `0.00%` two-shell symmetry error) but FAILED visual review on real data (`review-2026-10-28`)**:
   - **Unpinned rigid-body drift (`drift`)**: Because nothing pins rigid-body translation of the target shell, the synaptic push translates the solved target `2–3 um` off the real target mask (`fit_reach_03`, `fit_reach_04`, `fit_reach_05`, `fit_unreach_10`, `fit_unreach_12`).
   - **Path-dependent equilibria (`drift`)**: Re-solving a visit afresh at its fitted `pi_solved` disagrees with the 9-node continuation curve in `9 / 32` fit panels (`fit_reach_01`: `0.37` vs `0.88 um`; `fit_reach_03`: `1.61` vs `2.00 um`; `fit_reach_07`: `1.10` vs `1.51 um`; `fit_reach_08`: `0.98` vs `0.75 um`; `fit_reach_09, 10, 11, 13`; `fit_unreach_09`: `1.44` vs `0.08 um`).
   - **Reference target misalignment & huge initial overlap (`misaligned`)**: Placing the reference target from another visit by centroid and volume alone creates initial overlaps up to `11.0 um` (`overlap_01`: `11.0 um`, `overlap_03`: `7.7 um`, `overlap_12`: `6.3 um`); pushing apart then shifts the T cell `~4 um` off its mask (`fit_unreach_03, 10, 12, 13, 15`).
   - **Curve inversion bug (`other`)**: In `fit_reach_14`, the continuation curve stays near `0.05 um` up to `Pi = 10`, yet `pi_solved = 6.75` is reported for an observed depth of `0.50 um`.
   - **Corrupted target masks**: Several "unreachable" visits (`fit_unreach_01, 06, 07, 08, 14, 16`) had bogus observed cup depths from fragmented/merged/hollow masks.

## Decision

All Step 6b runs are **VOID / REJECTED** (`status: rejected` in `lab/registry.csv`). The final 365-chamber run was stopped, and Gate `G5` is **parked**. Mechanical modelling will only resume if Gate `G4` (associations rerun on `G1`–`G3` verified data) still shows a biological effect that requires a mechanical explanation.

## Open issues

- If Gate `G5` is reopened after `G4`: (1) pin rigid-body motion on the non-contact target hemisphere, (2) register reference shape orientation and cap initial overlap, (3) enforce path-independent re-solve reproducibility, (4) fix curve inversion in `fit_reach_14`, and (5) pass a full real-data fit gallery review.
