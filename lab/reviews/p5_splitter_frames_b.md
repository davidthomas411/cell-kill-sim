# P5 independent image review, group 2 (v03_MIP_ROI_30, v04_MIP_ROI_24, v05_MIP_ROI_8, v06_MIP_ROI_28, v07_MIP_ROI_2)

Method: every visit frame JPEG (01..20 or 01..22; there is no 00) was opened and viewed, RIGHT column (C5) judged against raw colours and the LEFT column (C4). Only what is visible is described. "Identical" means the C4 and C5 outlines look the same and the printed contact/volume numbers match. Grey-fragment counts are visual impressions only. No project file was edited.

Rating key per visit: BETTER / SAME / WORSE for the C5 BLUE target outline vs C4; MIXED = C5 fixes one thing and breaks another, or both outlines are poor and I cannot say which is closer to the cell.

---------------------------------------------------------------------
## v03_MIP_ROI_30 (20 visits)  - clean chamber, one big T cell and one big round tumour cell, one far-away small T cell

(a) Per-visit notes
- T cell (orange) is the real large green cell in all 20 visits in both C4 and C5. Target (blue) is the one large magenta cell the T cell touches; ID is tr=1/tr=1 in every visit in both columns. The small bright green elongated cell at the top of the frame is never outlined (correct, it is a different small cell).
- Visits 1-6, 8-14, 16-20: C4 and C5 identical (same outlines, same contact/volume numbers). SAME.
- Visit 7: the magenta cell is pushed into a kidney/two-lobe shape around the T cell (left lobe + lower lobe). C4 draws one blue outline round both (824 um^3, similar to neighbours 877 at v6 and 799 at v8). C5 cuts it with a straight diagonal line: blue = lower lobe only (451 um^3), left lobe becomes grey. Given the volume continuity (877 -> 824 -> 799) and that v6 and v8 show a single cell, this looks like an over-split of ONE cell by the T cell pressing in. Image is ambiguous (two lobes are visible with green T-cell material between them in XY), but I rate WORSE (probable). The straight cut also is not along an obvious gap.
- Visit 15: single round magenta cell next to the T cell. C4 outline = whole cell (929 um^3). C5 cuts it with a jagged vertical line in the middle of the cell: blue = left part, right part grey (668 um^3). Clear over-split of one round cell, WORSE (definite). Also visible in XZ as a vertical cut through the cell.
- Visit 11: both columns have a small spurious blue loop inside the T-cell overlap region (identical in C4/C5).
- Visit 18: blue has a notch at top (the cell is partly dim) in both; identical.
- Visits 19-20: target stays outlined in C5 after the cell dies (visit 20 filled with yellow SYTOX); outline kept in C5, nothing lost. No cell wrongly removed; no non-cell objects removed or added that I could see.
- Merges still present in C5: none visible (no two-cell merges in this chamber besides the v7 case where C4 was the "merge" only if you believe it is two cells).

(b) Verdict
- Tracking correct? YES, 20/20 (T cell and target both right and consistent).
- C5 vs C4 overall: identical in 18/20 visits; C5 only worse (over-splits) at 7 (probable) and 15 (definite). It never improves anything here.
- Worst frames: 15 (jagged vertical cut through a round cell), 7 (diagonal cut, 824 -> 451 um^3).

---------------------------------------------------------------------
## v04_MIP_ROI_24 (20 visits)  - clean, one big tumour cell and one T cell, approach, contact, late partial death (yellow dots at v19-20)

(a) Per-visit notes
- T cell (orange) is the real large green cell in all 20 visits. Target (blue) is the single large magenta cell the T cell approaches (v1-v8, contact 0 to a few um^2) and then contacts (v9 onward). One whole cell in every frame.
- C5 vs C4: outlines are visually identical in ALL 20 visits (same shapes, same volumes, same contact numbers). The only difference is the printed track ID labels: C4 shows T cell tr=2 at visits 6, 7, 8, target tr=2 at 17 and T cell tr=3 at 18 and 19 (the outline is on the same real cell each time, so these are label flips only); C5 shows tr=1/tr=1 throughout. So C5 is slightly better on identity consistency, but the picture is the same. Rated SAME for outline.
- Shared defects (same in C4 and C5): visit 5 blue has a small pointed tail at the right; visits 10, 17, 18, 19, 20: blue extends into the T cell overlap region with notches/loops (visit 18 blue has a small separate interior loop inside the T cell outline; v17 blue wraps a tongue into the T cell). Visit 11-12 blue is irregular/ragged on the dim tumour cell. Visit 16 blue has a small appendage at upper right.
- Visits 19-20: yellow SYTOX spots appear inside the tumour cell (partial dye uptake); outline retained in both.
- Cells wrongly removed by C5: none. Non-cell objects: none in view. Merges: none.

(b) Verdict
- Tracking correct? YES 20/20 in pixels; C5 also has the consistent ID (C4 flips IDs at 6-8, 17-19).
- C5 vs C4: SAME in all 20 visits (outline), ID labels better in C5.
- Worst frames: none for C5; the shared weak outlines are 17, 18 (blue leaks into T cell).

---------------------------------------------------------------------
## v05_MIP_ROI_8 (20 visits)  - one big T cell, tumour cells partly buried inside the T cell (white overlap), crowded from ~v8, a third cell and debris

(a) Per-visit notes
- Visit 1: T cell correct (large green). Magenta tumour cell at upper right, half hidden under the T cell. C4 blue = hidden overlap region + visible part (1128 um^3). C5 blue omits a lower-right chunk, which becomes a grey fragment (787 um^3). The tumour cell is one cell, so C5 over-splits it. WORSE.
- Visit 2: identical outlines; C4 label target tr=2, C5 tr=1. SAME.
- Visit 3: both blue include the overlap region inside the T cell. C5 adds a large grey sub-region in the middle of the blue (1367 -> 1037 um^3). One cell cut into pieces. WORSE (minor).
- Visit 4: C5 again carves a round grey object out of the lower part of the blue target (1295 -> 1045). WORSE (minor).
- Visit 5: C5 blue has a perfectly straight vertical cut through the magenta cell in XZ (and in XY), left/lower part grey (1366 -> 811 um^3). One cell cut by a straight line. WORSE.
- Visit 6: C5 grey blob cut out of the top of the magenta cell in XY and a straight diagonal line through it in XZ (1345 -> 1023). WORSE (minor).
- Visit 7: C4 and C5 identical (473 um^3). The small green debris object to the right is outlined grey in both (a non-cell object, not removed by C5). SAME.
- Visit 8: C5 carves a grey piece off the top-left of the target and puts a diagonal line in XZ (1292 -> 1063). A separate round magenta cell at lower right is grey in both. WORSE (minor). Small green debris is outlined grey in both.
- Visit 9: BETTER. C4's blue is a bright round magenta cell at upper right that does NOT touch the T cell (contact 0.0), while the tumour cell the T cell is actually touching (left, glowing where T cell meets it) is grey. C5 puts blue on the touched cell (654 um^3, contact 62) and the round cell becomes grey. C5 is the correct target here.
- Visit 10: C4 blue merges TWO tumour cells (left lobe + upper right round cell with a visible nucleus-like circle; 2197 um^3, clearly too big vs ~800-1300 elsewhere). C5 splits them (1170 um^3) but the cut is a jagged diagonal band: part of the round cell stays in blue, part is a thin grey strip along the right. Volume is plausible, the boundary is not along the visible gap. MIXED (better on merge, jagged). A small separate dim magenta cell lower left is outlined grey in C4 but has NO outline in C5 (a real but dim isolated tumour cell removed; harmless for the target).
- Visit 11: tumour cell largely overlapped by the T cell in XY. C4 blue 2157 um^3 (possibly two cells stacked in z, XZ shows a tall column). C5 blue 1032 um^3 with a horizontal jagged cut through it and grey slivers on top and right. Ambiguous; MIXED. A small magenta dot lower left is outlined in neither.
- Visit 12: tumour cell is cut by the left edge of the frame. Blue identical in C4/C5 (label tr=2 vs tr=1). To the right a big magenta cell is grey in both, but C5 additionally cuts it with a diagonal line into two grey pieces (over-split of an untracked cell). Target outline SAME.
- Visit 13: BETTER. C4 blue merges the upper lobe that touches the T cell with the two-lobed round cell at right and a thin strand (1468 um^3). C5 blue = the upper cell only (789 um^3); the right two-lobed cell is grey. Cleaner and matches the touching cell.
- Visit 14: BETTER (moderate). C5 blue = the round tumour cell at top (480 um^3), C4 blue is that cell plus wavy jagged extensions into the T cell (835 um^3). C5 drops the bright mass inside the T cell (grey). Ambiguous whether that bright mass is a second tumour cell; image is crowded. T cell outline in both columns is very large/irregular (spans the whole cluster) and may include more than one T cell (ambiguous).
- Visit 15: WORSE. Left tumour cell touching the T cell. C4 blue = that cell (816 um^3). C5 blue is a diagonal piece of it (224 um^3) and the upper-left of the cell is grey. T cell outline is a very long horizontal sausage in both columns (possibly two T cells merged; ambiguous).
- Visit 16: WORSE. C4 blue = left cell touching the T cell (738, contact 94). C5 blue jumps to a right-half fragment of the lower-right cell cluster (254 um^3, contact 0.0) and the left cell touching the T cell is grey. Target identity changes relative to v15 in C5.
- Visit 17: WORSE. C4 blue = a large merged region that wraps the lower part of the T cell plus the right tumour cell (1585 um^3, with tiny interior loops). C5 blue = thin strip along the right cell (389 um^3, contact 0.0), with a grey band beside it. Both poor; C5 blue does not cover the cell.
- Visit 18: WORSE. C4 blue merges the upper-right cell with a lower dim piece (1107). C5 blue = upper-right cell cut by a straight diagonal line (295 um^3, contact 0.0); a grey ring surrounds the rest. A green debris object inside is grey in both.
- Visits 19-20: identical outlines (target is a dim cell at the upper-left next to the T cell; in v20 blue spans two vertically adjacent things, also identical). A dim magenta cell lower right at v19 is grey in both. SAME.
- Tumour cell with SYTOX yellow fill: none seen in this chamber.
- Non-cell objects: small green debris outlined grey in both C4 and C5 at visits 6-8; C5 did NOT remove them. In visit 4 a green round object at lower-left (probably another small cell) is outlined in both.
- Merges still present in C5: visit 3-4 (hidden overlap region inside the T cell is still in blue), visit 11 (possible two-cell stack, ambiguous), visit 17 (blue spans T-cell interior in C4; C5 no longer).

(b) Verdict
- Tracking correct? YES about 15/20 (visits 1-13, 19, 20); AMBIGUOUS at 14, 15 (huge T outline, partial target); WRONG/hopping at 16-18 (C5 blue on a fragment with contact 0 while the T cell touches other cells).
- C5 vs C4 overall: mostly WORSE. C5 fixes the real merges at 9, 13 (and partly 14, 10) but fragments the target with straight or diagonal cuts at 1, 3, 4, 5, 6, 8, 15, 16, 17, 18 (volumes drop by 25-70 %).
- Worst frames: 15, 16, 17, 18 (target reduced to an arbitrary fragment), 5 (straight vertical cut), 1.

---------------------------------------------------------------------
## v06_MIP_ROI_28 (22 visits)  - very dim, many touching tumour cells and T cells in a clump; yellow dead cells appear from v16. Both methods are poor here; the image is hard to interpret even by eye.

(a) Per-visit notes (T-cell and target identity are largely unverifiable; I note what I can see)
- Visit 1: C4 orange = the green cell overlapped with tumour at lower left; C5 orange = a different green cell at upper right (contact 11.9). C5 blue is split into a long thin diagonal-cut object across the magenta mass (467 vs 675 um^3). WORSE.
- Visit 2: C4 blue is a few ragged fragments at top (155 um^3, nonsense); C5 blue = a clean outline of the left tumour cell (480 um^3), which touches a green cell (grey in C5). C5 target outline is the better one, but C5's orange T cell is the OTHER green cell at upper right, contact 0.0 (the green cell touching the tumour is grey). MIXED.
- Visit 3: C5 blue = left tumour cell, large green cell next to it grey; C5 orange T cell far upper right (contact 0.0). C4 blue = the middle cell touching the C4 orange T cell. MIXED (target outline looks clean in C5, T cell wrong).
- Visit 4: C5 blue is split into 2-3 pieces by straight cuts (327 um^3); C4 blue = one middle cell (319). WORSE. C5 T cell again the distant green cell, contact 0.0.
- Visit 5: C5 blue = lower sliver of the cluster (407, contact 0.0); C4 blue 953 (top and bottom merged). WORSE (fragment).
- Visit 6: blue outlines identical (651 um^3 in both: a long snake-shaped object that looks like at least two merged cells in both). C5 orange now is the right green cell (contact 57). SAME; merge present in both.
- Visit 7: C5 blue = lower-left cell, cleaner than C4's jagged left piece (276 vs 205); C5 orange = distant right green cell, contact 0.0. MIXED.
- Visit 8: C5 blue = a thin diagonal sliver cut from the lower cell by a straight line (350 vs 1564 um^3); C5 orange outline covers the upper magenta cells and weaves through them (T cell outline lying over tumour cells). WORSE.
- Visit 9: C5 blue = left half of the bright magenta cell with a curved cut (465 vs 1469); C5 orange is an elongated strip through the middle of the cell. WORSE.
- Visit 10: C5 blue = upper part of the big magenta blob cut by a straight horizontal line (840 vs 1883, contact 0.0). The blob may be 2 cells; the horizontal line is arbitrary. WORSE.
- Visit 11: blue outlines similar (806 vs 496); C5 adds straight horizontal lines in XZ. SAME / slightly worse.
- Visit 12: C5 blue moves to the dim upper cell and the lower fragment; the BRIGHT magenta cell that the C4 outline covers and that touches the T cell has only grey outlines in C5. WORSE.
- Visit 13: C5 blue 541 vs 1475, left cell cut by an inner curved line with grey parts. WORSE (ambiguous: C4 blue may be 2 cells).
- Visit 14: C5 blue 187 um^3 fragment at left vs 827. WORSE.
- Visit 15: both are a large irregular blue region over the cluster (C5 734 vs C4 1360, C5 drops the upper-left dim area). MIXED; both unreliable.
- Visit 16: SERIOUS. The yellow-filled (dead, SYTOX positive) cell at the left carries blue in C4. In C5 that cell has only grey outlines; blue moves to a dim elongated cell at the right (561 um^3). WORSE; a dead target filled with yellow loses its blue outline (it is still outlined, just grey).
- Visit 17: C5 blue = middle-upper dim fragment (388 um^3), yellow left cell grey. WORSE.
- Visit 18: C5 blue = a long thin diagonal strip crossing the cluster (525 vs 1389). WORSE.
- Visit 19: C5 blue = right cell only (477); the two yellow-filled cells at the left are only grey (C4 blue spans left yellow cells; right cell too, 1308). WORSE (dead cells lose the target outline).
- Visit 20: C5 blue = right tall cell with heavy yellow fill (843 um^3); the yellow cells at lower left are grey. C5 orange outline is a ragged shape lying directly on the yellow-filled tumour cell (T cell outline on a dead tumour cell). MIXED.
- Visit 21: C5 blue = fragmented right part (335) vs C4 1100 over the left yellow cell. WORSE. T cell outline in both columns includes parts of tumour cells.
- Visit 22: C5 blue = the right yellow dead cell (427 vs 933); orange outline in C5 wraps the lower-left yellow dead cell. MIXED.
- Non-cell objects: a few small bright green specks and green debris are not outlined in either; at v10 a green faint object at left is grey in both; C5 did not clearly remove or add non-cell objects. Cells wrongly removed by C5: the yellow dead cells at v16-22 lose their blue outline (they become grey, not deleted).
- Merges still present in C5: visit 6 (snake shape), visit 10 (big blob), visit 15 (large cluster), visit 20/22 cluster with several cells under one T-cell outline.

(b) Verdict
- Tracking correct? NO / cannot be confirmed. At most ~9/22 plausible (visits 6, 13, 15-22 are plausible), 0 can be called clearly correct. C5's orange is on a green cell that is NOT touching the target at v1-5, 7 (contact 0.0), and the T outline overlaps tumour cells from v8.
- C5 vs C4: mostly WORSE (14/22), 6 mixed, 2 same, 0 better. C5 produces straight-line fragments and loses the yellow-dead target cells.
- Worst frames: 8, 9, 10, 16, 19, 21 (fragment/sliver target, dead yellow cell no longer blue), 1.

---------------------------------------------------------------------
## v07_MIP_ROI_2 (22 visits)  - two touching tumour cells at the start (left round + right lobed); one T cell arrives; second dim tumour cell far upper right; target disappears at v22

(a) Per-visit notes
- T cell (orange) is the real large green cell in visits 1-21 in both columns (the smaller dim green cell at the bottom is grey in both at v1-v4). In visit 22 only a T cell at the upper-left corner is outlined; no tumour cell visible, target tr=0 in both columns (identical).
- Visit 1: BETTER. C4 blue merges the two touching tumour cells (1261 um^3). C5 blue = left round cell only (979 um^3). The right cell is grey, but C5 cuts it with a straight vertical line (over-split of an untracked cell).
- Visit 2: BETTER (modest). C5 blue = left cell (885 vs 1332 merged); extra grey ribbon fragments cross the lower part of both cells.
- Visit 3: BETTER. Clean split along the neck between the two cells, jagged but following the gap (544 vs 1309).
- Visit 4: MIXED. The cells have rearranged into a curved arrangement. C4 blue = whole curved blob (647); C5 blue = its lower right part (312) cut by a near-vertical line, left part becomes a mess of grey concentric fragments (also present in C4 as ring contours). I cannot tell where the true gap is.
- Visit 5: SAME (identical, 554 um^3). Blue is a dumbbell: an upper ragged cell connected through a narrow neck to a lower lobe at the T cell: still merged in both.
- Visit 6: MIXED. C5 splits the dumbbell: blue = upper cell (363), grey = lower lobe next to the T cell. C4 blue = whole dumbbell (580). The merge is removed but the lobe touching the T cell is dropped from the blue target. Contact prints 0.0 in both despite the cells visibly touching.
- Visits 7, 8, 9, 10, 12, 16, 20, 21: blue target identical in C4 and C5 (same volumes, same contact). Visit 7: blue includes an odd jagged region inside the T cell in both. In visits 7, 8, 9 the faint dim magenta cell at the upper right is outlined by C4 with a ragged grey outline; C5 has NO outline on it (a real but dim tumour cell removed; it was not the target, and the C4 outline was incomplete/ragged anyway). Visit 10 the upper magenta cell (grey in both) is intact.
- Visit 11: WORSE (modest). C5 blue 588 vs 966; diagonal cuts across the target where it overlaps the T cell; grey sub-region in XZ. The cell looks single.
- Visit 12: SAME (identical, C4 label tr=2, C5 tr=1). Both blue include speckled interior contours in the T cell region.
- Visit 13: WORSE. The magenta cell is one cell (visible as one bright magenta body) but C5 cuts it by a diagonal/vertical line (497 vs 948), left strip grey.
- Visit 14: SAME. Target = upper round tumour cell + some overlapped region; C5 additionally carves a grey sub-region in the T-cell overlap (924 vs 1215).
- Visit 15: MIXED. Two tumour cells at the T cell. C4 blue = lower-left bright cell (923, contact 77). C5 blue = upper dim cell (515, contact 175), whose lower boundary in XZ is a straight horizontal line across the T cell; the lower-left cell is grey. C5 is consistent with v14 (upper cell) while C4 switches; which is the intended target is ambiguous.
- Visit 17: MIXED. C5 blue = upper dim fragment (222 um^3, contact 2.3); C4 blue = lower-right sliver (245). Both fragments of dim cells; image very dim/noisy.
- Visit 18: MIXED. C5 blue = the left tumour cell at the left end of the long T cell (611, contact 0.2); C4 blue = lower-right cell (501, contact 432). Both are real tumour cells touching the long T cell; C5 identity changes from v17 and C4 differs. Ambiguous.
- Visit 19: SAME in quality: C5 blue = left whole cell (800) and C4 blue = right whole cell (538). Both are single cells; they disagree on which cell is the target.
- Visit 22: SAME; no tumour cell visible, both show target 0.
- Dead (yellow) cells: none in this chamber. Non-cell objects: not clearly removed in any visit; the dim grey objects (green blobs) are retained in both.

(b) Verdict
- Tracking correct? T cell correct 21/21 where present. Target plausible in ~14/22 (visits 1-3, 7-14, 19-21), questionable in 4, 5, 6 (merged/dumbbell/lobe dropped), 15-18 (identity hopping between the two cells at the T cell), no target in 22.
- C5 vs C4 overall: BETTER at 1-3 (separates two touching cells), identical in 12 visits, WORSE at 11 and 13 (cuts a single cell), 5 MIXED.
- Worst frames: 13, 11 (over-split of the target); 6, 15, 17, 18 (target choice).

---------------------------------------------------------------------
## Overall summary table

| Chamber | Tracking OK visits / total (C5) | C5 BETTER | SAME | MIXED | WORSE | Serious problems |
|---|---|---|---|---|---|---|
| v03_MIP_ROI_30 | 20 / 20 | 0 | 18 | 0 | 2 (v7 probable, v15 definite) | v15 straight jagged vertical cut through a round cell; v7 diagonal cut of one deformed cell (824 -> 451 um^3) |
| v04_MIP_ROI_24 | 20 / 20 (C4 label flips at 6-8, 17-19, C5 consistent) | 0 | 20 | 0 | 0 | none; outlines identical to C4 (shared: blue leaks into T cell at 17-20) |
| v05_MIP_ROI_8 | ~15 / 20 (2 ambiguous at 14-15, 3 wrong at 16-18) | 3 (v9, v13, v14) | 5 | 2 (v10, v11) | 10 | straight/diagonal cuts fragment target (v1, 3-6, 8, 15-18); v16-18 target becomes a fragment with contact 0.0; v15 target 816 -> 224 |
| v06_MIP_ROI_28 | 0 confirmed, ~9 / 22 plausible | 0 | 2 | 6 | 14 | C5 T cell on a green cell not touching the target (v1-5, 7); target reduced to slivers (v5, 8-10, 14, 18); yellow dead cells lose blue outline (v16, 19, 21); T outline over tumour cells (v8, 9, 20, 22); chamber very dim and crowded, both methods poor |
| v07_MIP_ROI_2 | ~14 / 22 | 3 (v1-3) | 12 | 5 | 2 (v11, v13) | v13, v11 over-split a single target cell; v4-6, 15-18 target identity ambiguous (merged dumbbell, hopping between two tumour cells); target disappears v22 (both) |

Totals for the five chambers (104 visit frames): C5 BETTER 9, SAME 57, MIXED 13, WORSE 28 (counts for the BLUE target outline only).

Cross-chamber observations
- C5's useful behaviour: separates touching tumour cells where there is a real neck (v07 visits 1-3, v05 visits 9, 13). It is identical to C4 where there is a single clean cell (v03, v04).
- C5's recurring failure: it cuts single cells (and cells overlapped by the T cell) with straight or diagonal lines or jagged near-vertical cuts, giving a smaller blue fragment and a grey remainder (v03 v15; v05 v1, 5, 15, 18; v07 v13; v06 many). These cuts usually do not follow a visible gap.
- C5 also over-splits untracked cells into grey pieces (v05 v12, v07 v1) and kept most non-cell debris (v05 v6-8 green blobs remain grey in C5). Removals seen: a dim real tumour cell grey outline removed at v07 v7-9 and v05 v10; no dead (yellow) cell was deleted outright, but in v06 v16-22 the yellow-filled dead cells lose their blue outline and are only grey.
- Where the T cell is in contact with the target, C4/C5 outlines both leak blue into the T-cell overlap region (v04 v17-20, v07 v7, v12, v05 v1-5); C5 does not fix this.
