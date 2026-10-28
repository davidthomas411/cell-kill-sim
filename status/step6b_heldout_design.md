# Step 6b design: first mechanical model and its held-out test

Written 2026-10-28, before the model is built or fitted. Committed first so
the history records the test ahead of the result.

## Model

Two closed triangulated shells (T cell, target) from the segmented masks,
coarse meshes (edge about 1 um, the resolution limit from step 5). Each shell
has cortical tension, area stretching, bending and a volume constraint. A
contact penalty stops interpenetration. Inside the synapse, the T-cell shell's
tension is alpha_syn x its tension elsewhere, and an active pressure p acts on
those faces only, pointing into the target. Solved quasi-statically.

- Stiffness parameter: cortical tension T (pN/um). Literature AFM values
  (Young's modulus E, Pa) are converted with a stated formula and assumptions,
  and the result is reported as a range, not one number.
- Identifiable from shape: only the dimensionless push
  Pi = p R_target / T_target, plus a weakly identified tension ratio
  chi = T_T,back / T_target. Absolute force needs an outside anchor.
- alpha_syn = 2 (Jung et al. 2021, AFM on antibody-coated glass, a different
  setup). Ablation: alpha_syn = 1.

## Solver validation (before any fit)

On synthetic spheres, compare the solver with a known result (indentation of a
tense shell or liquid drop under a patch load), and check mesh convergence at
edge lengths about 0.7, 1.0 and 1.4 um. Report the error.

## Reference (unloaded) shape

Every contact frame is already deformed. Preferred reference: the target's
shape on the last visit before first contact, where one exists. Fallback: the
contact frame with the cup filled in. Whatever reference is used must not
contain the quantity being predicted. Results are reported separately for the
two reference types.

## Data

Chambers with trace_num == 1, contact in visits 1-20, at least 4 pre-death
contact visits (so early and late halves each have 2 or more). Only pre-death
visits are used.

## Held-out tests

Chambers are split 50/50, stratified by experiment. chi is fitted on the
training half only and then frozen. All metrics are reported on the test half.

1. **Later visits from early visits.** For each test chamber, fit Pi on the
   first half of its pre-death contact visits. Freeze it, then predict cup
   depth (`cup_depth_hull_p95_um`) on the second half, using only that visit's
   measured geometry (patch radius, target size) as input.
   - Baseline: depth / patch radius from the early visits, times the late
     visit's patch radius.
2. **Shape away from the synapse.** Fit Pi to cup depth only. Predict, without
   further fitting, the change from the reference shape in (a) target
   sphericity and elongation and (b) T-cell sphericity and elongation.
   - Baseline: no change, and the training-half mean change.
3. **Experiment-level repeat.** Repeat tests 1 and 2 with a
   leave-one-experiment-out split for chi.

Error metric: median absolute error per chamber. Comparison: paired bootstrap
over chambers (2,000 resamples).

## Decision rule

- The model **adds something** if it beats the baseline in test 1, or beats
  both baselines in test 2(a) or 2(b), with the 95% CI of the paired difference
  excluding 0, in the 50/50 split and in the same direction in test 3.
- Otherwise the mesh model does no better than measured cup depth plus a
  proportional rule, and step 7 should use the simpler description.
- alpha_syn = 2 vs 1 is reported for every test. It does not change the
  decision.

## Reported but not part of the decision

- Fitted Pi per chamber and visit, and its relation to death within experiment
  (exploratory, compared against raw cup depth).
- Pi in dasatinib vs untreated chambers from the same donors.

## Addendum, 2026-10-28, before the final held-out run

Written after solver gates A and B (`.agy_step6b_gateA.md`, `.agy_step6b_gateB.md`)
and before any held-out result from the fixed solver. Both earlier held-out
runs are void: the first never solved per visit, and the second used a solver
with stale contact pairs and unconverged solves.

- Solver: contact pairs rebuilt during the solve, initial overlap pushed apart
  along the synapse axis, solves run to convergence, volume error < 1%.
  Benchmarks (a)-(c) plus a rigid-indenter contact benchmark (d) and a
  two-shell symmetry test pass with no solver-fitted constants.
- Pi grid 0-12, no extrapolation, no cap. A visit whose observed cup depth is
  deeper than the model reaches at Pi = 12 is "unreachable". Unreachable visits
  are not used to fit Pi. In test 1 they are still predicted and scored (the
  model's prediction is its solved depth at the frozen Pi). A chamber with no
  reachable early visit is dropped from test 1 and counted. Primary depth is
  net indentation d(Pi) - d(0).
- Known limits, reported with the result: about 20% of visits are unreachable
  on the 20 in-depth chambers; fitted Pi changes by up to about 40% between
  1.0 and 0.5 um meshes on real visits. The mesh stays at 1.0 um for cost.
- Step 7 route (a) required >= 95% in-range coverage. Gate B gives about 80%,
  so step 7 stiffness units stay on route (b) whatever this run shows.
- Tests, baselines, split, bootstrap and decision rule are unchanged.
