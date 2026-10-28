# Visual review of real-example panels, 2026-10-28

Panels: `figures/review/` (gallery: `figures/review/gallery.html`). Reviewed by
Claude: cup_measurement 1-16, fit_reachable 1-8, fit_unreachable 1-4 and 9-12,
death_labels 5-8, 25-30 and 13-18, deep_vs_shallow deep 1-4. Not yet reviewed:
initial_overlap, mesh_size, dasatinib_vs_untreated, the rest of deep/shallow and
fit sheets.

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
3. Real-looking contacts measure sensibly (cup_14, cup_15, cup_16, deep_04).

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
8. Several "unreachable" visits have bogus observed depths from item 1.

## Death labels (affect steps 4, 5b, 7)

9. **The dye signal is small.** Background is about 400 ADU; the threshold is
   404; killed targets read 404-416 and a clearly dye-positive neighbour about
   410-490. Calls near threshold (death_killed_01: 404.3) need a threshold
   sensitivity check.
10. **Possible overlap bug:** death_dasatinib_30 v5 reports 27 px overlap while
    the dye speck appears outside the target outline in both views.
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
