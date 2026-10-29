---
id: 2026-10-28-06-cup-depth-measurement-validated-on-synthetic-sha
date: '2026-10-28'
title: Cup-depth measurement validated on synthetic shapes
gate: G3
status: reviewed
author: claude
reviewers: [claude]
signoff: null
code_commit: 36b9484
datasets: [step5_descriptors]
protocols: [cup-measurement@v1]
galleries: [cup-validation-2026-10-28]
supersedes: null
superseded_by: null
---

## Question

Is `cup_depth_hull_p95_um` (src/contact_descriptors.py:709-715) biased? In
particular, does restricting the distance to T-cell-facing hull facets inflate
depth (suspected from qc_clean_02, a flat-looking contact recorded at 2.87 um)?

## What was run

```bash
python3 src/validate_cup_depth.py      # synthetic cases a-f + real-panel re-measure
python3 -m unittest tests.test_cup_depth
```

Run by agy (Gemini 4 Argon High); report `.agy_cuptest_report.md`.

## What was looked at

- figures/cup_depth_validation.png (summary)
- figures/review/cup_validation/: b_dent_R5_a3_d2, case_f_hollow_shell,
  f_hollow_R5_d1p0_filled_mask (reviewed by Claude).

## Observations

- **No facing-facet bias.** On a flat cut (7 cases, expected 0) the result is
  0.000. A spherical dent of 2 um measures 1.96 um; a 3 um dent measures
  2.75-2.94 um; a tilted synapse axis (10-30 deg) gives no change; a wide bowl
  with 0.3 um sag measures 0.196 um (one voxel). All-facets distance matches the
  recorded value in 17/22 real panels. My concern was wrong.
- **Merged targets inflate depth.** Two spheres with the T cell between them
  (expected cleft 0) give 2.3-4.0 um. This matches gallery panels deep_07,
  qc_edge_01, qc_edge_02 and dasa_vs_untr_untreated_02.
- **Visit means are contaminated by single bad sweeps.** qc_clean_02 visit 2:
  the displayed sweep measures 0.71 um; the visit mean of 2.87 um includes
  sweeps 3-4 where a second target merged (2,450 um^3; 5.6-5.8 um "cups").
  Per-visit descriptors average 5 sweeps, so QC must run per sweep.
- The sphere-reference depth (`cup_depth_sphere_p95_um`) is badly biased: 0.3-3.0
  um on flat contacts. It should not be used.
- Hollow-shell case f: the shell without filling measures 0.98 um (expected 1.0).
  After `fill_solid_cell_mask` it measures 0.196 um, and the figure shows the cup
  filled in. How the synthetic shell was built is not clear, so this is an open
  question, not a finding. Solid dented cells (case b) are measured correctly.

## Decision

Keep the hull-depth definition. G3 work: QC per sweep (not per visit), exclude
merged-target sweeps, drop the sphere-reference depth. Recompute visit
descriptors from clean sweeps only.

## Open issues

- Confirm whether fill_solid_cell_mask can erase real cups on real hollow masks
  (compare filled vs unfilled depth on the hollow panels dasatinib_09, 11, 14).
- RESOLVED (handbook definitions.md): channel 2 is mApple-Mem membrane (561 nm), as the captions say. Earlier reading: its signal is
  brighter in the target core than at the rim (e.g. v01 ROI_2 stem 00495: core
  560, rim 490, background 401), so it is likely a whole-cell label. Channel 1
  marks the T cell, channel 0 the death dye. Correct the captions.
