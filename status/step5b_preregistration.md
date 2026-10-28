# Step 5b pre-registration: do contact descriptors relate to killing or donor?

Written 2026-10-27, after the step 5a descriptors were computed and before any
descriptor was compared with death, donor or condition. Committed before the
analysis runs, so the git history records the plan ahead of the result.

## Question

If the contact-shape descriptors (including the indentation/topography ones)
show no association with death or donor, a mesh model of this dataset is
unlikely to explain killing. It would still estimate force relative to
stiffness, but it would not predict who kills.

## Data

- Unit: chamber. Primary set: the 416 chambers with `trace_num == 1` (the
  paper's inclusion rule). Sensitivity: all chambers with a contact.
- Outcome labels: step 4 v2, contacted target only, any-visit rule (the lab's
  definition), 200-minute window (visits 1-20). Sustained (>= 2 visits) rule as
  a sensitivity check.
- Descriptors: `data/step5/per_chamber_descriptors.csv` and the per-visit file.

## Avoiding reverse causation

A dying target shrinks and fragments, which changes every shape descriptor.
Descriptors are therefore taken only from visits **before** the death visit:

- (A) the first-contact visit;
- (B) the mean over contact visits strictly before the death visit (all
  contact visits for survivors, up to visit 20).

Peak-contact-area values are not used, because the peak can fall after death.
Chambers whose first contact is the death visit have no pre-death data and are
dropped from (B) (count reported).

## Descriptors

- Primary family (ICC >= 0.5 in step 5a, z-robust or z-scaled consistently):
  `cup_depth_hull_p95_um`, `cup_depth_over_radius`, `cup_volume_hull_um3`,
  `tcell_protrusion_frac`, `tcell_protrusion_vol_um3`, `patch_equiv_radius_um`,
  `contact_area_um2`, `target_volume_um3`, `target_sphericity`,
  `target_elongation`, `tcell_volume_um3`, `tcell_sphericity`.
- Secondary family (topography, low ICC, flagged as noisy):
  `cup_asymmetry_norm`, `zernike_tilt_amp_um`, `zernike_astig_amp_um`,
  `zernike_coma_amp_um`, `zernike_defocus_um`, `zernike_spherical_um`,
  `rough_above_cutoff_um2`, `patch_elongation`. A null here is weak evidence,
  because their noise attenuates any association.
- Outliers: values are heavy-tailed (e.g. hull cup volume up to ~4000 um^3,
  larger than a target). All tests are rank-based; chambers where a cup or
  protrusion volume exceeds the target's own volume are flagged and the result
  is repeated without them.

## Tests

1. **Death (chamber level).** Cox model of time from first contact to death,
   censored at 200 min, **stratified by experiment** so that between-experiment
   (donor/batch) differences cannot drive it. One descriptor at a time, rank-
   transformed (normal scores) within experiment. Report HR per SD and 95% CI.
   Repeat adjusted for contact area (is shape informative beyond size?).
   Benjamini-Hochberg within each family.
2. **Death (prediction, held out).** Logistic model of death by 200 min,
   leave-one-experiment-out. Baseline: contact area (A) + condition. Added:
   primary family. Report held-out AUC with bootstrap CI for baseline and
   baseline + descriptors.
3. **Donor.** In untreated experiments each donor is one experiment (one
   imaging day), so donor and batch cannot be separated, and there are only 10
   donors (5 high-killing, 5 low-killing). Tests: per-donor median of each
   descriptor, (a) Mann-Whitney high- vs low-killing donors (n = 5 vs 5; the
   smallest possible two-sided p is 0.0079), (b) Spearman correlation of donor
   median with donor kill rate (n = 10). Also Kruskal-Wallis across the 10
   donors at chamber level, reported as donor-or-batch variation.
4. **Positive control (latrunculin).** Same donors with and without
   latrunculin A (v08 vs v11 hd2, v09 vs v12 hd10, v10 vs v13 hd5). Actin
   inhibition should reduce indentation and protrusion. If the descriptors
   cannot see this, they are unlikely to see subtler effects, and a null in
   tests 1-3 is uninformative rather than negative.

## Decision rule

- **Association present** if any primary descriptor has BH q < 0.05 in test 1
  (unadjusted) and its direction is the same in at least 2 of the 3 condition
  blocks (high-killing, low-killing, latrunculin), **or** test 2 raises held-out
  AUC by >= 0.03 with the bootstrap CI of the gain excluding 0, **or** a primary
  descriptor separates high- and low-killing donors in test 3(a) at p < 0.01
  with |Spearman| >= 0.6 in 3(b).
- **No association** if none of these hold **and** the positive control works
  (latrunculin lowers `cup_depth_hull_p95_um` or `tcell_protrusion_frac`,
  paired direction in 3/3 donors). Then a mesh model will not explain killing
  from this dataset.
- **Inconclusive** if none hold and the positive control fails.
- Whether shape adds anything beyond contact area (test 1 adjusted) is
  reported separately; it does not change the gate, since contact area itself
  is one of the descriptors a mechanical model would use.

## Erratum (added 2026-10-27, after the analysis ran)

Groups 8*-10* (v11, v12, v13; folder suffix `_ds`) are **dasatinib 50 nM**, a
kinase inhibitor that blocks CAR/TCR signalling, not latrunculin A. The name
above was copied from an earlier report without checking. No test or rule is
changed: the control still asks whether switching off the signalling that
drives actin pushing lowers indentation and protrusion in the same donors.
