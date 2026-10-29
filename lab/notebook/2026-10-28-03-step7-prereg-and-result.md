---
id: 2026-10-28-03-step7-prereg-and-result
date: "2026-10-28"
title: "Step 7: Pre-registered focused-pushing test and inconclusive result (UNVERIFIED)"
gate: G4
status: draft
author: "agy (backfilled)"
reviewers: []
signoff: null
code_commit: "6fa449f, 6bf7ee0"
datasets:
  - step5_descriptors
  - step4_death_labels_v2
  - step7_focused_pushing
protocols:
  - cup-measurement@v1
  - death-calling@v1
galleries:
  - review-2026-10-28
supersedes: null
superseded_by: null
---

## Question

Under the pre-registered focused-pushing hypothesis (`status/step7_preregistration.md`, committed in `6fa449f` before running the test, with `cup_depth_over_radius` as the sole confirmatory metric), do CAR-T cells from high-killing donors push in a more focused way than those from low-killing donors (Test 7A), does focused pushing predict faster killing within donors beyond contact area and duration (Test 7B), and does 50 nM dasatinib lower the focus metric in `3/3` paired donors (Test 7C)?

** Verification Notice: All Step 7 results (`data/step7/`) are UNVERIFIED and on hold at Gate `G4` because the underlying masks and descriptors failed Gate `G1`/`G3` visual audit (in particular, outlier donor `hd10` is dominated by fragmented/hollow masks).**

## What was run

```bash
python3 src/step7_focus_test.py
```

Commits:
- `6fa449f` (`Step 7 pre-registration: focused-pushing test, committed before analysis`)
- `6bf7ee0` (`Step 7: focused-pushing test per pre-registration (result: inconclusive, control failed on primary metric)`)

## What was looked at

- Pre-registration `status/step7_preregistration.md` and report `.agy_step7_report.md`.
- Output tables `data/step7/test7a_donor_focus.csv`, `data/step7/test7b_within_donor_cox.csv`, `data/step7/test7c_dasatinib_control.csv`, `data/step7/stiffness_route_b_summary.csv`, and figure `figures/step7_focus.png`.
- Visual review gallery `review-2026-10-28` (`dasa_vs_untr_untreated_01`..`08`, `dasa_vs_untr_dasatinib_09`..`16`, `deep_vs_shallow_deep_01`..`08`, `deep_vs_shallow_shallow_09`..`16`).

## Observations

1. **Test 7C — Positive control (`dasatinib 50 nM` in `hd2, hd10, hd5`, `UNVERIFIED`) FAILED on primary metric**:
   - On `cup_depth_over_radius` (`d / a`), dasatinib lowered the donor median in only `1 / 3` paired donors (`hd10`: `0.500 -> 0.400`; whereas `hd2`: `0.303 -> 0.383` and `hd5`: `0.316 -> 0.345`; donor-stratified one-sided $p_{\text{lower}} = 0.749$).
   - Secondary metrics `cup_depth_hull_p95_um` (`d`) and `tcell_protrusion_frac` fell in `3 / 3` donors (`p = 0.0049` and `p = 0.0022`), but because dasatinib simultaneously shrank patch radius `a` (`patch_equiv_radius_um`) by `27–38%` in `hd2` and `hd5`, the ratio `d / a` did not decrease.
2. **Test 7A — Between-donor test (`5` high vs `5` low untreated donors, `UNVERIFIED`) NULL**:
   - High-killing donor median `cup_depth_over_radius` (`0.299`) vs low-killing (`0.336`): exact two-sided permutation `p = 0.246` (Summary B) and `p = 0.651` (Summary A), driven by untreated low-killing donor `hd10` (`v09`, donor median `0.500`).
3. **Test 7B — Within-donor Cox model (`UNVERIFIED`, restatement of Step 5b on same data)**:
   - Adjusted for `contact_area_um2`: `HR = 1.375` per SD (`p = 0.0022`, Summary A) and `HR = 1.656` per SD (`p = 6.9e-6`, Summary B).
4. **Visual review finding (`status/visual_review_2026-10-28.md`)**:
   - Donor `hd10` (`v09` untreated and `v12` dasatinib)—which produced the `0.500` outlier in Test 7A and was the only donor where dasatinib lowered `d / a` in Test 7C—is mostly fragmented or hollow masks (`v09` clean fraction `0.05`, `v12` clean fraction `0.16`).

## Decision

Under the pre-registered decision rule, Step 7 returned the failed-control sentence (*"The focus metric does not respond to a known perturbation of T-cell pushing, so step 7 is inconclusive"*), and the entire result is marked **UNVERIFIED** (`status: unverified` in `lab/registry.csv`, blocked at Gate `G4`).

## Open issues

- Wait for Gates `G1`–`G3` to complete; note that `clean-cell-rule@v1` leaves only `18` untreated clean visits in `hd10` (`v09`) and `60` pooled dasatinib clean visits, which cannot support the 5-vs-5 donor test or dasatinib control without mask repair or re-segmentation from raw volumes.
