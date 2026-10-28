# Step 7 pre-registration: do high-killing donors push in a more focused way?
Written 2026-10-28, before running the step 7 analysis, so the focus metrics,
adjustments, and stiffness conversion rules are fixed ahead of the result.

## Sharp question
Do T cells from high-killing donors push on target cells in a **more focused**
way than T cells from low-killing donors, and do cells that push in a more
focused way kill faster within the same donor/experiment?

## Focus metrics (from step 5a reliable, z-pitch-robust descriptors)
Mask topography below 1.4 um wavelength is sweep-to-sweep noise (step 5a ICC <
0.42). "More focused" pushing is defined at the macro-synapse scale using the
three indentation/protrusion descriptors that passed both reliability (ICC >=
0.5) and z-pitch robustness (median |rel. shift| <= 10% between z = 0.1962 and
0.240 um):
1. **Primary focus metric:** `cup_depth_over_radius` = `cup_depth_hull_p95_um` /
   `patch_equiv_radius_um` (ICC = 0.60, visit ANOVA ICC = 0.78, z-shift = -2.8%).
   Indentation depth per unit patch radius: higher values mean pushing deeper
   for a given contact footprint rather than spreading flat.
2. **Secondary 1 (target indentation depth):** `cup_depth_hull_p95_um` (ICC =
   0.69, visit ANOVA ICC = 0.84, z-shift = +5.6%), adjusted for
   `patch_equiv_radius_um` (or `contact_area_um2`), measuring depth at fixed
   patch size.
3. **Secondary 2 (T-cell invasive focus):** `tcell_protrusion_frac` (fraction of
   T-cell volume protruding inside the target's 3D convex hull; ICC = 0.61,
   visit ANOVA ICC = 0.84, z-shift = 0.0%), a strictly z-invariant measure of
   how much of the T cell concentrates into the indentation.

## Units of analysis, cohorts, pre-death rule, and adjustments
- **Cohorts:** Primary cohort is `trace_num == 1` chambers (n = 416 total: 327
  untreated across 10 donors [5 high-killing: hd1, hd3, hd4, hd6, hd7; 5
  low-killing: hd8, hd9, hd2, hd10, hd5] and 89 dasatinib 50 nM across 3 paired
  donors [hd2, hd10, hd5]). Sensitivity: all 502 chambers, and excluding volume-
  outlier visits (`cup_volume_hull_um3 > target_volume_um3`).
- **Pre-death visits only (avoiding reverse causation):** Only contact visits in
  visits 1-20 strictly **before** `first_any_dead_visit_200m` enter (3,108
  untreated pre-death contact visits across 305 `trace_num == 1` chambers; 492
  dasatinib visits across 61 chambers). Both Summary A (first-contact pre-death
  visit, n = 316 untreated) and Summary B (pre-death chamber mean, n = 305
  untreated) are evaluated.
- **Hierarchy (visit -> cell/chamber -> donor):** Each untreated donor is one
  experiment (imaging day), so donor and batch are confounded and n = 10 donors
  is the effective sample size for between-donor inference.
- **Adjusting for contact duration and patch size:**
  - *Visit level:* Linear mixed models include fixed effects for contact duration
    (`visit - first_contact_visit`) and patch size (`patch_equiv_radius_um`),
    with nested random intercepts `(1 | donor / chamber)`.
  - *Chamber level:* Summary B models adjust for `patch_equiv_radius_um` (or
    `contact_area_um2`) and pre-death visit count
    `n_pre_death_contact_visits_any_200m` (Summary A needs no duration adjustment).

## Confirmatory tests, effect sizes, CIs, and alpha
Primary alpha = 0.05 (two-sided) on `cup_depth_over_radius` alone; this is the
only confirmatory metric. The two secondaries are reported with Benjamini-Hochberg
q across the 3 focus metrics but cannot by themselves support a decision.
1. **Test 7A — Between-donor focus (high vs low killing, n = 5 vs 5 donors):**
   - *Donor-level permutation test:* Compute per-donor chamber medians (Summary A
     and B, unadjusted and residualized on `patch_equiv_radius_um` and pre-death
     visit count). Compare 5 high vs 5 low donors via exact two-sided permutation
     over all $\binom{10}{5} = 252$ donor-label assignments on group-mean
     difference and Mann-Whitney U (minimum two-sided p = 2/252 = 0.00794).
     Report Hodges-Lehmann median difference, Hedges' g with exact permutation
     95% CI, and Spearman $\rho$ with donor kill rate (n = 10, bootstrap 95% CI).
   - *Mixed-model check:* Fit `focus ~ group_high + patch_equiv_radius_um + time_in_contact + (1 | donor / chamber)` across all 3,108 untreated pre-death visits; test `group_high` by permuting the 10 donor labels (252 permutations) alongside Satterthwaite df (df <= 8).
2. **Test 7B — Within-donor cell-level killing (experiment-stratified):**
   - *Overlap with step 5b:* 5b already fitted an experiment-stratified Cox
     model to these descriptors on the same visits (HR 1.42 per SD for
     `cup_depth_over_radius`). 7B repeats it with the adjustments below, so it
     is a restatement on the same data, not an independent confirmation.
   - Cox model of time from first contact to death (censored at 200 min),
     stratified by `experiment` (donor), on within-experiment normal-score ranks
     of each focus metric, adjusted for `contact_area_um2` and (in Summary B)
     `n_pre_death_contact_visits_any_200m`. Report HR per SD, profile-likelihood
     95% CI, Wald p, and mixed-effects logistic/frailty `(1 | donor)` check.
3. **Test 7C — Positive control (dasatinib 50 nM in hd2, hd10, hd5):**
   - Dasatinib lowered depth and protrusion in 3/3 donors in step 5b (Summary B).
     Test whether dasatinib lowers each focus metric and inferred push $\Pi$ in
     3/3 paired donors and by donor-stratified rank Mann-Whitney (one-sided
     alpha = 0.025, paired bootstrap 95% CI).

## Expressing the answer in units of stiffness
Which route maps focus to dimensionless push $\Pi = p R_{\text{target}} / T_{\text{target}}$
is decided **solely by the step 6b gate**, not by step 7 results:
- **Route (a) — Step 6b mechanical model** (if the step 6b solver fix passes its
  gate: <5% analytical error, >=95% in-range coverage without extrapolation, and
  held-out test win over proportional baseline): use per-visit solved $\Pi$
  ($\alpha_{\text{syn}} = 2$, frozen training $\chi$).
- **Route (b) — Simple-description paper relation** (if step 6b fails its gate):
  Laplace normal-stress balance on a spherical target shell (radius
  $R_{\text{target}}$, tension $T_{\text{target}}$) indented to depth $d$
  (`cup_depth_hull_p95_um`) over patch radius $a$ (`patch_equiv_radius_um`),
  with bowl curvature $\kappa_{\text{cup}} \approx 2d/(a^2+d^2) \approx 2d/a^2$
  for $d \ll a \ll R_{\text{target}}$ (uniform isotropic $T_{\text{target}}$,
  quasi-static cap balance):
  $\Pi_{\text{simple}} = p R_{\text{target}} / T_{\text{target}} \approx 2 + 4(R_{\text{target}}/a)(d/a)$
  and net indentation push $\Delta\Pi_{\text{indent}} = 4(R_{\text{target}}/a)\,\text{cup\_depth\_over\_radius}$.
- **Physical units (`data/step6b/afm_to_tension_conversion.json`):** Multiply by
  nominal $T_{\text{target}} \in [49, 163]\text{ pN}/\mu\text{m}$ (full range
  $[18, 430]\text{ pN}/\mu\text{m}$; T-cell $T_{\text{T}} \in [570, 1790]\text{ pN}/\mu\text{m}$
  from Jung et al. 2021 AFM) to report active pressure $p = \Pi T_{\text{target}} / R_{\text{target}}$
  (Pa) and total force $F = p A_{\text{contact}}$ (nN) as conditional intervals.

## Decision sentences
- **Donor-level focus supported** (Test 7A permutation p < 0.05 on the primary
  metric, Test 7C passes): *"T cells from high-killing donors push in a
  significantly more focused way than T cells from low-killing donors."*
- **Within-donor cell-level effect only** (Test 7A null, Test 7B adjusted BH
  q < 0.05 in Summary A and B, Test 7C passes): *"Within a donor/experiment,
  more focused pushing predicts faster killing beyond contact size and duration,
  but between-donor differences do not exceed batch variation (n = 5 vs 5)."*
- **No effect** (Tests 7A and 7B null, Test 7C passes): *"Focused pushing does
  not explain killing either between or within donors."*

- **Positive control fails** (Test 7C does not show lower primary focus under
  dasatinib in 3/3 donors): *"The focus metric does not respond to a known
  perturbation of T-cell pushing, so step 7 is inconclusive."* Tests 7A and 7B
  are still reported.

Added on review by Claude before any step 7 analysis (2026-10-28): the primary-only
confirmatory rule, the step 5b overlap note, and the failed-control sentence.

## Secondary / exploratory analyses (not part of the decision)
- **Step 5c raw-image extension (secondary):** On the 20 in-depth chambers
  (`results/step5_indepth_chambers.csv`), extract sub-voxel 561 nm membrane and
  488 nm F-actin topography/enrichment to test concordance with mask focus.
- **Exploratory:** Unadjusted donor medians, Zernike modes, and within-visit SD.
