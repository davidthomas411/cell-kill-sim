---
id: 2026-10-27-03-step5a-descriptors
date: "2026-10-27"
title: "Step 5a: 3D mask contact and cup descriptors across 502 chambers (UNVERIFIED)"
gate: G3
status: draft
author: "agy (backfilled)"
reviewers: []
signoff: null
code_commit: "88199ec, b3be277"
datasets:
  - lab_masks
  - step3_reproduction
  - step5_descriptors
protocols:
  - cup-measurement@v1
galleries:
  - review-2026-10-28
supersedes: null
superseded_by: null
---

## Question

What surface detail do the deposited 3D U-Net masks retain compared with raw 561-nm membrane images, and can we compute macro-scale (`>= 1.4 um`) 3D contact patch and target cup indentation descriptors across all 502 chambers (`10,510` visits, `52,545` sweeps)?

** Verification Notice: All Step 5a descriptor tables (`data/step5/`) reported in this entry are UNVERIFIED and blocked at Gates `G1` and `G3`.**

## What was run

```bash
python3 src/mask_resolution_check.py
python3 src/contact_descriptors.py --workers 6
```

Commits:
- `88199ec` (`Check how much surface detail the deposited masks keep`)
- `b3be277` (`Step 5a: mask contact descriptors for all chambers; pre-register the 5b association check`)

## What was looked at

- Resolution benchmark output `data/step5_resolution/` and `figures/step5_mask_resolution_check.png` (20 sweeps across 7 chambers with local 3D raw `.h5` volumes).
- Descriptor overview `figures/step5_descriptors_overview.png` and `data/step5/descriptor_dictionary.csv`.
- Subsequently inspected on real-data gallery `review-2026-10-28`:
  - `cup_measurement`: `cup_01`..`cup_16`
  - `dasatinib_vs_untreated`: `dasa_vs_untr_untreated_01`..`08`, `dasa_vs_untr_dasatinib_09`..`16`
  - `deep_vs_shallow`: `deep_vs_shallow_deep_01`..`08`, `deep_vs_shallow_shallow_09`..`16`

## Observations

1. **Mask surface resolution cutoff (`commit 88199ec`)**: On 20 raw-vs-mask volumes, the 3D U-Net mask dilates the target boundary outward by `+0.72 um` (`+3.69` voxels) relative to the raw 561-nm membrane half-maximum contour and attenuates `> 50%` of surface roughness power below spatial wavelength `lambda_50% = 1.38 um` on the contact patch (`1.45 um` away from contact). Sub-micron microvilli (`0.1–0.5 um`) are erased; only macro-scale descriptors (`>= 1.4 um`) are retained.
2. **Axial pitch check (`commit 88199ec`, `b3be277`)**: Uncontacted targets have bounding-box ratio `Delta Z / Delta XY = 0.807` (`0.1962 / 0.807 = 0.243 um`). All 33 descriptors were computed at both `z = 0.1962 um` and `z = 0.2400 um` (`_z024`) across `52,545` sweeps (`22,808` contact sweeps, `5,447` contact visits, `502` chambers).
3. **Severe mask-quality artefacts found in visual review (`review-2026-10-28`, UNVERIFIED)**:
   - **Fragmented targets**: When a target mask is split into multiple pieces (`cup_01`: `10.15 um` depth, `2,821 um^3`; `cup_03`: `9.18 um`, `2,550 um^3`; `cup_09`; `cup_10`: `5.10 um`, `depth/radius = 2.75`; `deep_01`), the 3D convex hull bridges the gap between pieces rather than measuring a synaptic dent. Across `4,523` `trace_num == 1` contact visits, median `cup_depth_hull_p95_um` is `1.35 um` for `target_sphericity < 0.5` (`n = 1,925`) vs `0.72 um` for `target_sphericity >= 0.8` (`n = 292`; Spearman `rho = -0.26`).
   - **Merged targets**: Two target cells fused into one mask (`dasa_vs_untr_untreated_02`, `deep_06`, `deep_07` at `5.60 um`, `deep_08`) cause the inter-cell cleft to be measured as a deep cup.
   - **Hollow targets**: Ring-shaped or broken-shell masks (`dasa_vs_untr_dasatinib_09, 11, 14, 15`; `shallow_10, 11`) produce `5.8–6.8 um` bogus cup depths.
   - **FOV edge truncation**: Cells cut by the crop border (`cup_05`, `cup_13`) distort hulls and volumes.
   - **Donor `hd10` (`v09`, `v12`)**: Dominated by fragmented and hollow target masks.

## Decision

`step5_descriptors` is **UNVERIFIED** (`status: unverified` in `lab/registry.csv`, blocked at Gates `G1` and `G3`). Descriptors must be recomputed/filtered on Gate `G1`-verified clean cells (`clean-cell-rule@v1`) and visually verified before any downstream use.

## Open issues

- Complete Gate `G1` clean-cell verification (`clean-2026-10-28`), then recompute and re-verify `G3` cup descriptors on whole, single, solid cells only.
