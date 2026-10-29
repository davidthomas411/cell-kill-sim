# Visual review of real-example panels, 2026-10-28

Panels: `figures/review/` (gallery: `figures/review/gallery.html`). Reviewed by
Claude: all 128 panels (33 contact sheets of 4), in two passes.

## Measurement (affects steps 5, 5b, 7)

1. **Fragmented or merged target masks inflate cup depth.** The "target" object
   is often several separate pieces (cup_01, cup_03, cup_09, cup_10,
   fit_unreach_01, deep_01). The 3D convex hull bridges the pieces, so the
   hull-to-contact distance measures the gap between pieces, not a dent.
   Extremes: cup_01 10.2 um depth (target 2,821 um^3), cup_03 9.2 um (2,550
   um^3), cup_10 5.1 um, depth/radius 2.75. Across 4,523 trace_num==1 contact
   visits, median cup depth falls from 1.35 um (target sphericity < 0.5, n =
   1,925) to 0.72 um (sphericity >= 0.8, n = 292); Spearman rho = -0.26
   (depth) and -0.34 (depth/radius).
   - Risk: if targets fragment as they start to die, before dye uptake,
     fragmentation could inflate "cup depth" just before death and create part
     of the step 5b / 7B association.
2. **Cells cut off by the field-of-view edge** (cup_05, cup_13). Volumes, hulls
   and cup depths are wrong for truncated cells.
3. **Two target cells merged into one mask.** Two round lobes with the T cell
   in the groove between them (dasa_vs_untr_untreated_02; deep_06, deep_07,
   deep_08). The groove is then measured as a deep cup (deep_07: 5.6 um).
4. **Hollow target masks.** Ring-shaped or broken-shell targets whose interior
   is not filled (dasatinib_09, 11, 14, 15; shallow_10, shallow_11). Hollow
   masks give 5.8-6.8 um "cups".
5. **Mask quality differs by experiment.** Donor hd10 (v09 untreated, v12
   dasatinib) is mostly fragmented or hollow (untreated_01, 03, 06, 07;
   dasatinib_09, 11, 14, 15). hd10 was the step 7 outlier (focus 0.50) and the
   one donor where dasatinib lowered depth/radius, so both may be artefact.
   Tiny target fragments also appear (dasatinib_16: 37 um^3; dasatinib_13:
   154 um^3).
6. Clean, whole, separate cells measure sensibly, and where raw images exist
   the masks follow the membrane well (cup_14-16, shallow_12, shallow_13,
   deep_04, dasa_vs_untr_untreated_04, 05, 08). In mesh_size_06 the raw cell
   bodies look larger than the mask outlines; to check on more raw images.

## Mechanical model (affects step 6b, including the final run in progress)

4. **Solved cells drift.** The solved target (purple) often sits 2-3 um from
   the real target (fit_reach_03, 04, 05; fit_unreach_10, 12). Nothing in the
   model pins rigid-body motion, so the push translates the target.
5. **Re-solving is not reproducible.** Solving a visit afresh at its fitted Pi
   gives a different depth from the gate-B continuation curve: fit_reach_01
   0.37 vs 0.88 um, fit_reach_03 1.61 vs 2.0, fit_reach_07 1.10 vs 1.51,
   fit_reach_08 0.98 vs 0.75, fit_unreach_09 1.44 vs 0.08. The equilibrium
   depends on the solve path (multiple equilibria or incomplete convergence).
6. **The starting (reference) target is misaligned.** It comes from another
   visit and is placed by centroid and volume only, but the target moves,
   rotates and changes shape between visits (fit_unreach_03, 10, 12).
7. **Push direction can miss the target.** In fit_unreach_03 the solved T cell
   slides past the target and the curve stays near 0.2 um.
8. Several "unreachable" visits have bogus observed depths from items 1-4
   (fit_unreach_01, 06, 07, 08, 14, 16).
9. **The fitted Pi can disagree with its own curve.** fit_reach_14: the curve
   stays near 0.05 um up to Pi = 10, yet the fitted Pi is 6.75 for an observed
   0.50 um. Possible bug in the curve inversion.
10. **Cells sometimes never touch in the model** (fit_unreach_13: final gap
    1.43 um, curve flat at 0) or the starting target is placed ~4 um from the
    real one (fit_unreach_15).
11. **The starting overlap is often far larger than any cup** (overlap_01:
    11.0 um, overlap_03: 7.7 um, overlap_12: 6.3 um). That is a misplaced
    reference, and pushing apart then moves the T cell ~4 um off its mask.
    Sensible cases overlap by about the cup depth (overlap_06, 08, 11).
12. Mesh size: 1.0 and 0.5 um outlines agree closely in slices; depth differs
    4-36%. Same drift and misplacement as above in every mesh panel.
13. Re-solve mismatches seen in 9 of 32 fit panels (fit_reach_01, 03, 07, 08,
    09, 10, 11, 13; fit_unreach_09). Fits that look sound: fit_reach_06, 12,
    15, 16.

## Death labels (affect steps 4, 5b, 7)

9. **The dye signal is small.** Background is about 400 ADU; the threshold is
   404; killed targets read 404-416 and a clearly dye-positive neighbour about
   410-490. Calls near threshold (death_killed_01: 404.3) need a threshold
   sensitivity check.
10. **Possible overlap bug:** death_dasatinib_30 v5 reports 27 px overlap while
    the dye speck appears outside the target outline in both views.
12. **Gallery overlap numbers disagree with the chamber labels.** Visits shown
    as dead on consecutive visits in chambers labelled surviving with no
    positive visit at all: death_surviving_24 (v11 and v12: 296/251 and
    104/236 px, "any-visit pos: False"); death_surviving_19 (v12-v14,
    including a bright blob at v14, labelled single-visit). death_killed_07 is
    dead at v7 in the panel but first positive at v8 in the label. Most likely
    the panel overlaps against every target in the field while the labels use
    only the contacted target; or the labels have a bug. Either way, one of
    them is wrong, and it must be resolved before step 4 is trusted.
13. Real deaths look right where dye appears inside the target and grows over
    visits (death_killed_03, 04, 06, 08, 10). Possible false kills from a
    dye-positive neighbour overlapping in projection: death_killed_05,
    death_killed_09.
11. **Projection overlap can come from a neighbour.** In death_surviving_17 a
    dye-positive neighbouring cell lies under the target in both projections
    at v19 (304 / 928 px). death_surviving_18 v18 has a large transient smear.
    The sustained-death rule rejects both, but single-visit positives are
    projection artefacts, and a sustained neighbour overlap would be a false
    kill.

## Status of earlier results

- Step 5b / 7B within-donor association: unconfirmed until rerun with
  fragmented and edge-truncated visits flagged or excluded.
- Step 6b: all model fits, including the final run, are provisional until
  items 4-7 are fixed.
- Step 4 death labels: need the overlap-bug check and a threshold sensitivity
  analysis; 3D (not projection) overlap would remove item 11.

## Addendum: QC job results and its 18 panels (reviewed by Claude, all 18)

Source: `.agy_qc_report.md`, `data/qc/`, `figures/review/qc_flagged/`, `figures/review/qc_clean/`.

- **Death-label mismatch resolved for death_dasatinib_30.** It was a gallery bug:
  `src/review/make_gallery.py` overlapped dye with every target in the field (a
  71-voxel U-Net speck here). `src/death_labels.py` v2 uses the contacted target
  only and records no overlap. death_surviving_19, death_surviving_24 and
  death_killed_07 are presumably the same, and are still to be confirmed.
- **Death calls are threshold-sensitive.** Sustained kills (trace_num==1, 200 min)
  at 404 ADU / 20 px: 81 of 416. At 406 ADU: 64 (-17). At 410 ADU: 38 (-43).
  Low-killing group: 11.2% at 404, 5.6% at 406, 0.6% at 410. Most low-killing
  kills are faint, near-threshold calls; they need panel-by-panel review (G2).
- **Lab-label fragmentation rises before death.** flag_fragmented_raw, last 3
  pre-death visits vs earlier: 43% vs 35% (any-visit rule, 92 chambers,
  p = 0.020); 43% vs 31% (sustained, p = 0.017). This is the confound risk for
  the cup-death association.
- Excluding the literal flags barely moves the step 5b HRs (cup depth/radius
  1.60 -> 1.51). But the literal flag catches only 3% of visits, so this does
  not clear the association.
- qc_frag_01-06: all genuine small fragments (41-173 um^3). In qc_frag_06
  (raw) the target pieces sit where the membrane image shows almost no
  cell: a possible U-Net false target.
- qc_edge_01-06: cells genuinely cut at the border. qc_edge_01 and 02 are also
  merged targets (about 2,000 um^3, two lobes), with 5-6 um "cups".
- **New concern, cup measurement definition (qc_clean_02, 04).** A visually
  flat contact records 2.87 um (qc_clean_02, patch radius 6.6 um). The code
  (src/contact_descriptors.py:709-715) takes each patch point's min distance
  to the planes of T-cell-facing hull facets only. The true distance to the
  hull is the min over all facets, so excluding facets can overestimate depth
  near the rim of wide or curved contacts. Synthetic validation with known
  answers is running (.agy_cuptest_report.md).
