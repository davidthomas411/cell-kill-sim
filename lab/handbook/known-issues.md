# Known Issues (`lab/handbook/known-issues.md`)

All issues below were identified by visual review of real-data gallery `review-2026-10-28` (`status/visual_review_2026-10-28.md`) and technical audits. Every issue links to the notebook entry where it was recorded or addressed.

## 1. Cell Identity & Cup Measurement Issues (Gates `G1` & `G3`)

1. **Fragmented target masks inflate 3D convex-hull cup depth**
   - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-27-03-step5a-descriptors](../notebook/2026-10-27-03-step5a-descriptors.md), [2026-10-28-05-clean-subset-rule](../notebook/2026-10-28-05-clean-subset-rule.md)
   - **Panels**: `cup_01` (`10.15 um` depth, target `2,821 um^3`), `cup_03` (`9.18 um`, `2,550 um^3`), `cup_09`, `cup_10` (`5.10 um`, `depth/radius = 2.75`), `fit_unreach_01`, `deep_01`.
   - **Mechanism & risk**: When the target mask consists of multiple disconnected pieces, the 3D convex hull bridges across the pieces and `cup_depth_hull_p95_um` measures the inter-piece gap rather than a synaptic dent. Across `4,523` `trace_num == 1` contact visits, median cup depth is `1.35 um` when `target_sphericity < 0.5` (`n = 1,925`) vs `0.72 um` when `target_sphericity >= 0.8` (`n = 292`; Spearman `rho = -0.26`). If dying targets fragment before SYTOX uptake, pre-death fragmentation could create part of the Step 5b / 7B Cox association ([2026-10-27-04-step5b-result](../notebook/2026-10-27-04-step5b-result.md)).
2. **Cells truncated by the field-of-view boundary (`edge`)**
   - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-28-05-clean-subset-rule](../notebook/2026-10-28-05-clean-subset-rule.md)
   - **Panels**: `cup_05`, `cup_13`.
   - **Mechanism**: Cell bodies cut by the X/Y/Z crop border yield corrupted volumes, convex hulls, and indentation depths. Excluded by criterion 3 (`touches_fov_edge == False`) in `clean-cell-rule@v1`.
3. **Two target cells merged into one mask (`merged`)**
   - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-28-05-clean-subset-rule](../notebook/2026-10-28-05-clean-subset-rule.md)
   - **Panels**: `dasa_vs_untr_untreated_02`, `deep_06`, `deep_07` (`5.60 um` depth), `deep_08`.
   - **Mechanism**: Two adjacent round target cells fuse into a bilobed object with the T cell sitting in the cleft between them; the 3D hull bridges the cleft and records a spurious deep cup.
4. **Hollow ring-shaped or broken-shell target masks (`hollow`)**
   - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-28-05-clean-subset-rule](../notebook/2026-10-28-05-clean-subset-rule.md)
   - **Panels**: `dasa_vs_untr_dasatinib_09`, `dasa_vs_untr_dasatinib_11` (`6.75 um` depth), `dasa_vs_untr_dasatinib_14`, `dasa_vs_untr_dasatinib_15`, `deep_vs_shallow_shallow_10`, `deep_vs_shallow_shallow_11`.
   - **Mechanism**: Target membrane class (`pred == 4`) forms an open ring or broken shell with missing interior (`pred == 2`), producing `5.8–6.8 um` false cup depths.
5. **Severe experiment-specific mask degradation in donor `hd10` and `tiny` debris fragments**
   - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-28-03-step7-prereg-and-result](../notebook/2026-10-28-03-step7-prereg-and-result.md), [2026-10-28-05-clean-subset-rule](../notebook/2026-10-28-05-clean-subset-rule.md)
   - **Panels**: `dasa_vs_untr_untreated_01`, `03`, `06`, `07`; `dasa_vs_untr_dasatinib_09`, `11`, `13` (`154 um^3`), `14`, `15`, `16` (`37 um^3`).
   - **Mechanism**: Donor `hd10` (`v09` untreated clean fraction `0.05`, `v12` dasatinib clean fraction `0.16`) is predominantly fragmented or hollow, which drove the `hd10` donor outlier (`cup_depth_over_radius = 0.50`) and the single-donor dasatinib reduction in Step 7 ([2026-10-28-03-step7-prereg-and-result](../notebook/2026-10-28-03-step7-prereg-and-result.md)).

## 2. Death Label Issues (Gate `G2`)

6. **Gallery target-outline mismatch with v2 chamber labels (`label_mismatch`)**
   - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-27-02-step4-v2-labels](../notebook/2026-10-27-02-step4-v2-labels.md)
   - **Panels**: `death_surviving_24` (visits 11–12 show `296/251` and `104/236 px` overlap in panel but chamber label has `ever_dead_any_visit_200m == False`), `death_surviving_19` (visits 12–14 positive in panel, labelled single-visit), `death_killed_07` (positive at v7 in panel, v8 in table).
   - **Mechanism**: Gallery Case 6 projected all raw target voxels `(pred == 2) | (pred == 4)` whereas `per_chamber_death_labels_all_v2.csv` projected only the contacted target instance (`instance_seg_matlab_xyz`).
7. **Possible overlap calculation discrepancy (`other`)**
   - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-27-02-step4-v2-labels](../notebook/2026-10-27-02-step4-v2-labels.md)
   - **Panels**: `death_dasatinib_30` (visit 5 reports `27 px` overlap while the dye contour appears outside the target outline in both XY and XZ views).
8. **2D projection overlap from dye-positive bystander cells (`neighbour_overlap`)**
   - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-27-02-step4-v2-labels](../notebook/2026-10-27-02-step4-v2-labels.md)
   - **Panels**: `death_surviving_17` (v19 `304/928 px` overlap from a dead bystander underneath), `death_surviving_18` (v18 transient smear), `death_killed_05`, `death_killed_09`.
9. **Narrow 405-nm signal-to-background margin (`404.0 ADU` vs `~400 ADU`)**
   - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-27-02-step4-v2-labels](../notebook/2026-10-27-02-step4-v2-labels.md)
   - **Panels**: `death_killed_01` (peak Gaussian intensity `404.3 ADU` vs threshold `404.0 ADU`).
10. **`MIP_ROI_0` late 405-nm left-edge saturation band (`visit >= 19`)**
    - **Notebook entries**: [2026-10-27-02-step4-v2-labels](../notebook/2026-10-27-02-step4-v2-labels.md)
    - **Scope**: All `39 / 39` visits at `visit >= 19` (`t >= 190 min`) in `MIP_ROI_0` across all 13 experiments exhibit a `14,505–25,317 ADU` left-edge band. Excluded in `src/death_labels.py` v2 (`roi0_artifact_flag`).

## 3. Mechanical Model Issues (Gate `G5` — Parked / Void)

11. **Unpinned rigid-body drift (`drift`)**
    - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-28-02-step6b-runs-void](../notebook/2026-10-28-02-step6b-runs-void.md)
    - **Panels**: `fit_reach_03`, `fit_reach_04`, `fit_reach_05`, `fit_unreach_10`, `fit_unreach_12`.
    - **Mechanism**: Without rigid-body pinning on the non-contact target hemisphere, synaptic pushing translates the entire target shell `2–3 um` away from its real mask.
12. **Path-dependent equilibria / re-solve non-reproducibility (`drift`)**
    - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-28-02-step6b-runs-void](../notebook/2026-10-28-02-step6b-runs-void.md)
    - **Panels**: `9 / 32` fit panels (`fit_reach_01` `0.37` vs `0.88 um`, `fit_reach_03` `1.61` vs `2.00 um`, `fit_reach_07` `1.10` vs `1.51 um`, `fit_reach_08` `0.98` vs `0.75 um`, `fit_reach_09`, `fit_reach_10`, `fit_reach_11`, `fit_reach_13`, `fit_unreach_09` `1.44` vs `0.08 um`).
13. **Misaligned starting reference target and excessive initial overlap (`misaligned`)**
    - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-28-02-step6b-runs-void](../notebook/2026-10-28-02-step6b-runs-void.md)
    - **Panels**: `fit_unreach_03`, `fit_unreach_10`, `fit_unreach_12`, `fit_unreach_13`, `fit_unreach_15`, `overlap_01` (`11.0 um` push-apart), `overlap_03` (`7.7 um`), `overlap_12` (`6.3 um`).
14. **Curve inversion discrepancy (`other`)**
    - **Notebook entries**: [2026-10-28-04-visual-review-2026-10-28](../notebook/2026-10-28-04-visual-review-2026-10-28.md), [2026-10-28-02-step6b-runs-void](../notebook/2026-10-28-02-step6b-runs-void.md)
    - **Panels**: `fit_reach_14` (continuation curve flat near `0.05 um` up to `Pi = 10`, yet `pi_solved = 6.75` for observed depth `0.50 um`).
