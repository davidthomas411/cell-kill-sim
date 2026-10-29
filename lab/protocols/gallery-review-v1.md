---
id: gallery-review
version: v1
date: "2026-10-28"
status: reviewed
---

# Protocol: Gallery Review (`gallery-review-v1`)

## Purpose

No dataset stage feeds the next until a real-image gallery for that stage has been inspected panel by panel and passes its stage gate (`G1`–`G5`). Matching summary numbers in a published table does not validate segmentation masks or extracted measurements.

## 1. Panel Selection (Stratified Random + Extremes + Borderlines)

Every stage-gate gallery renders real experimental data (`*_predictions.h5`, raw 561-nm membrane / 488-nm actin `.h5` volumes fetched via `src/download_raw.py --file-list`, and 405-nm SYTOX MIPs) across three mandatory strata using a fixed random seed (`20261028`):

1. **Stratified random examples**: sampled across all three biological conditions (Untreated high-killing CAR-T, Untreated low-killing CAR-T, Dasatinib 50 nM CAR-T) and across donors (`hd1`–`hd10`).
2. **Extremes of each measured quantity**: maximum and minimum values (top 2–3 and bottom 2–3) of the stage's primary metrics (e.g., `cup_depth_hull_p95_um`, `cup_depth_over_radius`, `target_volume_um3`, `target_raw_largest_frac`, `pi_solved`, `push_apart_um`).
3. **Borderline threshold cases**: visits or chambers closest to decision boundaries (e.g., within a few pixels of the 20-px 405-nm overlap cutoff, near the 404.0 ADU dye threshold, or closest to each of the five clean-cell QC thresholds).

## 2. Visual Standard Across All Panels

1. **Raw images underneath**: Every 3D slice panel must show the raw fluorescence image (561-nm target membrane and/or 488-nm F-actin) beneath our extracted outlines, alongside the deposited U-Net labels (classes 0–4) when diagnosing segmentation failures.
2. **Fixed colour scheme**:
   - Tracked target outline: blue solid (`#1F4E79`)
   - Tracked T-cell outline: orange solid (`#C0561F`)
   - Other cells in field: grey fill/outline (explicitly labelled)
   - Contact patch voxels: purple squares (`#6B4C9A`)
   - Target convex hull / reference surface: green dashed (`#2E7D5B`)
   - Measured cup depth vector / dead-dye contour: red (`#B3261E`)
   - Solved model shapes: dashed outlines in fixed plot colours (`#6B4C9A` target, `#111111` T cell)
   - A complete legend/key and all numerical metrics from `index.csv` must appear on every panel.

## 3. Verdict Scale & Controlled Tag Vocabulary

Each reviewer inspects every panel at full resolution (via `tools/review_app.py` or contact sheets) and records a row in `lab/reviews/<gallery_id>.csv` with:

- **Verdict (`verdict`)**:
  - `ok` — Cell identity, segmentation contour, overlay alignment, and computed measurement are physically sound and match the underlying raw image.
  - `problem` — A segmentation error, truncation, merge, hollow shell, projection overlap artefact, label mismatch, or solver failure invalidates the measurement on this panel.
  - `unsure` — Ambiguous boundary, faint dye signal near threshold, or missing raw channel requiring secondary check.
- **Controlled tags (`tags`, semicolon-separated)**:
  - `fragmented` — Target or T cell is split into multiple disconnected pieces.
  - `merged` — Two distinct cells are fused into a single mask object.
  - `hollow` — Ring-shaped or broken-shell mask with unfilled interior.
  - `edge` — Cell body is truncated by the field-of-view border.
  - `tiny` — Implausibly small cell fragment (`< 200 um^3`).
  - `misaligned` — Reference or starting mesh does not align with the real cell.
  - `drift` — Solved mechanical mesh translates off the real cell mask (`2–3 um` rigid-body drift) or re-solve disagrees with continuation curve.
  - `label_mismatch` — Panel overlay numbers disagree with the chamber/visit table label.
  - `neighbour_overlap` — Dye-positive bystander cell overlaps target projection in XY/XZ without dye inside the contacted target.
  - `real_death` — Confirmed target death with SYTOX dye appearing inside the contacted target and persisting across visits.
  - `clean` — Whole, single, solid, well-segmented T cell and target away from FOV edges.
  - `other` — Issue not covered above (must be explained in `note`).

## 4. Independent Second Reviewer & Gate Signoff

1. **Primary complete pass (`claude`)**: Reviews 100% of panels in the gallery, logs per-panel findings, and summarizes failure rates by layer:
   - *(a)* U-Net labels wrong while raw cell is whole (lab segmentation error),
   - *(b)* U-Net labels sound but extracted object wrong (our extraction code bug),
   - *(c)* Raw cell itself blebbing/fragmenting (real biology / possible death signal).
2. **Independent second reviewer (`user` / PI)**: Reviews panels independently in `tools/review_app.py` on key stage gates (`G1`, `G2`, `G3`). The review app and static site display all reviewers' verdicts side by side and highlight any panel where reviewers disagree.
3. **Iterate and lock**: Code or subset rules are fixed to resolve identified problems, the same panels are re-rendered, and the gallery is re-reviewed until pass criteria hold and the notebook entry is signed off.
