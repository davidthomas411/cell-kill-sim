# P5 independent image review, group 3 (C4 left column vs C5 right column)

Chambers: v07_MIP_ROI_26, v08_MIP_ROI_24, v08_MIP_ROI_38, v09_MIP_ROI_29, v09_MIP_ROI_38.
Every visit frame of every chamber was opened and examined (22 + 20 + 20 + 22 + 22 = 106 frames). Everything below is what is visible in the panels. Where the image is ambiguous this is stated. "Identical" means the outlines and the printed volume/contact are the same in both columns.

Note on the "C5 target outline vs C4" grading: it grades the BLUE outline only (whole single cell, follows cell border, correct cell). T-cell (orange) problems are reported separately.

---------------------------------------------------------------------
## 1. v07_MIP_ROI_26 (22 visits)

Scene: two tumour cells lying end to end (large, left and right), one T cell above/right. From about v8 the cells rearrange into a 2-3 cell cluster that the T cell wraps around, and the tumour ends yellow (dead, SYTOX filled) at v20-22.

Per-visit notes
- v1: T (orange): correct, large green cell. C4 blue = BOTH tumour cells in one outline (merge, 1390 um3). C5 blue = only the right-hand cell (the one nearer the T cell), rounded, 515 um3; left cell grey. BETTER (clean split on a real neck). The grey outlines inside C5 left cell show some over-splitting of the left cell into slivers, but not the target.
- v2: Same as v1. C4 merged pair (1567); C5 right cell only (476), grey slivers over the left cell. BETTER.
- v3: C4 blue = whole pair merged (1410, smooth). C5 blue is a jagged, snaking region that runs along the middle of both cells, crossing the neck between them and ignoring the cell borders (411 um3). WORSE (C4 is merged but tidy; C5 is irregular and cuts across cells in both XY and XZ).
- v4: C4 blue jagged, covering the left cell partly plus some of the right (merge/partial, 1062). C5 blue is a thin lens-shaped sliver on the right edge of the right cell (253 um3), far smaller than the cell. Both bad. SAME (both bad); C5 arguably the worse shape (sliver).
- v5: C4 merged pair (1428). C5 blue is a horizontal jagged band across the lower half of both cells (485), not following any cell border. WORSE.
- v6: C4 merged pair (1518) with a hole. C5 blue = right-hand cell only, smooth, 514; a vertical grey strip cuts the left cell in two. BETTER (target). Over-splitting of the non-target left cell (strips).
- v7: C4 merged pair (1486). C5 blue = right cell only, smooth, 568. BETTER. Grey horizontal sliver objects inside the left cell (over-split, not target).
- v8: T has moved to touch the upper edge of the right cell. C4 blue = left cell only (743 um3, contact 0, wrong cell, the T cell is not touching it). C5 blue = right cell (643, contact 17.5), but the left border of blue is a perfectly straight vertical line through the cell; the cell seems cut roughly at its neck, acceptable. BETTER (correct cell; contact detected). Straight artificial cut line.
- v9: Cells have rearranged: T cell (large, below) now touches a small pale upper-right magenta cell. C4 blue = the big left cell (contact 0, not touched). C5 blue = the upper-right small cell but only its lower-left part, cut by a diagonal straight line, plus a bit of the right (241 um3). Contact 30.8. BETTER on identity (touched cell), but outline is a truncated polygon. Identity continuity from v8 is ambiguous (cannot tell that this is the same cell as the v8 target).
- v10: C5 blue = round upper-right cell touched by T, smooth and whole (499). C4 blue = left cell (not touched). BETTER.
- v11: The T cell now overlays a pale left cell; a bright compact magenta cell lies to its right. C5 blue = the bright compact cell (smooth, 238 um3, contact 22); C4 blue = the left pale cell with bleb (359, contact 84). Which is the "touched" cell is ambiguous (T overlaps the left cell in XY; XZ suggests the right cell sits against it too). C5 outline itself is clean. BETTER (outline), identity ambiguous.
- v12: Three overlapping cells under the T cell. Both columns jagged. C5 blue = right bright cell plus a notch into the T region (322). C4 blue = left cell plus notch. SAME (messy).
- v13: C4 blue = large left cell (695, contact 109). C5 blue = small upper-right cell with an intrusion at the T boundary (255). Identity differs from v12 for C4 and is the same as v12 for C5 (C5 consistent). SAME (outline).
- v14: Same pattern as v13; C5 blue the upper-right cell with a jagged wedge into the T cell (352); C4 blue is left cell. SAME.
- v15: Identical in both columns (596 um3). SAME. Blue = pink left blob wrapped around by T, outline plausible.
- v16: C4 blue = whole cluster (901, upper bright + lower green-speckled cell in one outline). C5 blue = lower-left cell only, cut from the upper cell by a diagonal line (582). BETTER (separates two cells), cut is straight.
- v17: Identical (570). Blue is irregular and overlaps the orange T outline; orange T outline has holes/loops around a bright green+magenta area. SAME.
- v18: C4 blue 890 (big, includes a long dim tail). C5 blue 330 = upper dim cell with a horizontal slit plus a small second contour near the T boundary. Both are messy. SAME.
- v19: C4 blue sprawls far down-left past the cell (514); C5 blue less sprawling (470). C5 removed a dim pinkish blob outline at lower left (a pale, diffuse blob; looks like out-of-focus haze/weak cell, ambiguous). BETTER (marginal).
- v20: C4 562 vs C5 351; both blue regions follow the dying cell group with holes; the cell now has yellow (dead) core. SAME.
- v21: Identical (694). Blue encloses the yellow dead cell with an orange T outline looping through it. SAME.
- v22: Identical (519). C5 removed the grey outline of a faint magenta/green-dot cell upper right that C4 had (ambiguous: a faint dim blob with a green spot; could be a weak real cell). SAME (target). Dead yellow target keeps its blue outline.

T cell (orange): the large green T cell in all 22 visits. In v17-22 the orange outline has loops and a lobe that wraps the yellow dead tumour cell at lower left (same in C4 and C5), so the T cell outline is partly polluted by the target's yellow/green overlap.

Chamber verdict
1. Tracking correct? T cell: yes (22/22). Target: correct single tumour cell and clean in about 11/22 (C5 v1, 2, 6, 7, 8, 10, 11, 15, 16, 21, 22 roughly); v3-5 are jagged/cut across cells; v9 and v11-14 identity is ambiguous because the cluster is overlapped and the T cell touches different cells in turn.
2. C5 vs C4: 10 BETTER (v1, 2, 6, 7, 8, 9, 10, 11, 16, 19), 10 SAME, 2 WORSE (v3, v5; v4 also bad). C5 correctly splits the end-to-end pair where it works (v1, 2, 6, 7), but at v3-5 it cuts through both cells with a jagged line.
3. Worst frames: v3, v4, v5 (jagged blue crossing both cells / sliver); v9 (target polygon is half a cell); v18.
- Cells wrongly removed by C5: none clear. Possibly faint dim blobs at v19 and v22 (ambiguous). The yellow dead target keeps its outline.
- Non-cells removed: no clear case. The green debris outlines (grey, lower right v2-v7) are kept by C5.
- Merges still in C5: none in the sense of the original pair; C5 blue at v12-14 mixes pieces of two cells (T-boundary intrusion).

---------------------------------------------------------------------
## 2. v08_MIP_ROI_24 (20 visits)

Scene: one round tumour cell touched by one T cell; clean chamber. A second T cell enters at v9 and v19.

Per-visit notes
- v1: Identical. T correct (elongated green). Blue = one tumour cell with a small notch at the contact. SAME.
- v2: C4 blue = round cell only (324, contact 0) with the small lower magenta lobe (where the T cell touches) outlined grey. C5 blue = round cell plus lobe in one outline (651, contact 84). The lobe is consistent with v1, v3 where it belongs to the target. BETTER (whole cell, contact detected).
- v3-v8: Identical outlines/numbers (v3 662, v4 628, v5 601, v6 679, v7 534, v8 572). Blue = one whole round cell, orange = the T cell. SAME. v7: the T cell lies over the cell (overlap shows yellowish-white contact zone); blue still one cell.
- v9: Identical. A second green T cell (grey outline) lies beside the orange T cell; blue fine. SAME. The orange covers only the upper T cell, grey the lower one: ambiguous which green cell is the attacker, same in both columns.
- v10-v11: Outlines identical (580, 598). C4 labelled T "tr=2" while C5 keeps T "tr=1" (ID consistency better in C5; outline same). SAME for outline; tracking ID BETTER in C5.
- v12: Identical (644). A grey object inside the T cell (second T) in both. SAME.
- v13-v17: Identical (610, 663, 630, 690, 383). SAME. At v17 the cell is small and bright (pink/white, partly overlapping T) and the outline hugs it with a small tail at the lower boundary; fine.
- v18: C4 target 329 um3 with a grey inner outline (target split in two pieces). C5 blue = whole cell, no inner split, 482 um3. BETTER.
- v19: Identical target (532). A big second T cell (grey) on the left. C5 removed a faint diffuse magenta blob under the T cell (grey in C4); it is a very faint out-of-focus smear, so this is a correct removal of a non-cell/haze. SAME for target.
- v20: Identical (466). SAME.

Chamber verdict
1. Tracking correct? Yes: T cell and target correct, one whole tumour cell, identity consistent 20/20. C5 gives a more consistent T ID (v10, v11).
2. C5 vs C4: 2 BETTER (v2, v18), 18 SAME, 0 WORSE.
3. Worst frames: none for C5 (v9 ambiguous second T in both; v17 target small/overlapped).
- Cells wrongly removed: none. Non-cell removed correctly: faint magenta haze v19.
- Merges in C5: none.

---------------------------------------------------------------------
## 3. v08_MIP_ROI_38 (20 visits)

Scene: one tumour cell at the left touching a large green cluster that contains at least two T cells (a smoother left cell and a bright lobed cell at right, later a third green mass). Other tumour cells and a dead (yellow-green) cell appear at the top from v10.

Per-visit notes
- v1: Target: both columns outline the same single left tumour cell (1150), C4 label tr=2 vs C5 tr=1. T cell: C4 orange = one T cell (left-centre of the cluster, grey outlines on the rest). C5 orange encloses the whole green cluster (at least two T cells merged: smooth left cell + bright lobed right cell). C5 WORSE for the T cell (merge); target SAME.
- v2: C4 blue = whole tumour cell (1131). C5 blue is cut by a curved line through the left part of the cell; the cut piece is left grey; 829 um3. The cell looks like one round cell in both views, so this is an over-split. WORSE (target). T: C5 again outlines the whole cluster (merged T cells); C4 only the central cell.
- v3: Target identical (1070) whole cell. T: C5 orange spans the whole cluster; C4 one cell. Target SAME; T worse in C5.
- v4: The tumour cell is now deformed (squeezed against the T). C4 blue = lower lobe only (375, contact 0), grey on the upper-left lobe. C5 blue = both lobes in one outline (542, contact 27). Whether this is one cell or two is ambiguous, but with contact present C5 looks more complete. BETTER (ambiguous). T: C5 merged cluster again.
- v5: Identical (386). Blue = the lower lobe and a bright round magenta dot at far left included inside the blue outline in both (a bright dot is probably debris, small spurious part of the outline). SAME. T: C5 orange is jagged and spans the cluster (merged); C4 one cell.
- v6: C4 blue = lower lobe only (436, contact 15). C5 blue = lower cell larger and smoother (697, contact 77); upper-left round bud stays separate and grey in both. BETTER (target). T: C5 orange covers the entire cluster including the right lobed T cell (merge).
- v7-v20: target volume and contact are identical to C4 at every visit (v7 649, v8 498, v9 574, v10 689, v11 505, v12 555, v13 548, v14 498, v15 518, v16 522, v17 496, v18 526, v19 575, v20 558), outlines look the same. SAME for target. Target is one single tumour cell at the left of the cluster, identity consistent; it develops a bud at the upper left (v11-20), included in the blue outline, fine; yellow dead dye is seen at the contact edge from about v11.
- T cell v7-v20: the orange outline covers a large green mass in both columns. At v10, a second T cell at the lower right (cut by the frame edge) is inside the same orange outline (merge, same in C4 and C5). At v14 and v19 grey outlines show the rest of the green mass (separate T cells).
- Removals (C5 vs C4):
  - v10-v20: top-left of the frame holds a bright yellow-green rounded object with magenta debris around it (looks like a dead cell filled with dye plus apoptotic fragments). C4 had grey outlines on the debris around it (v10 whole cluster; v11-14 partial fragments); C5 has no outline on it at all in v10-v20. Not the tracked target, but a real dead tumour cell lost its outline.
  - v14: bottom-right tumour cell at the frame edge (magenta, textured) is outlined in C4 and has no outline in C5. v15: tumour cell touching the T cell on the right is outlined in C4 and not in C5. v20: tumour cell at right of the T cell outlined in C4 and not in C5. These are real tumour cells wrongly removed.
  - v11: a faint circular ring (pale, vesicle-like) outlined in C4 is removed by C5: correct removal. v13: small grey debris outlines near the top removed: correct (they were small fragments).

Chamber verdict
1. Tracking correct? Target: yes, the single left tumour cell is followed correctly and whole for 20/20 (C5 label consistent). T cell: not for C5 in v1-6 (C5 orange includes neighbouring T cells; clean 14/20); v7-20 identical to C4 (a multi-T mass partly merged in both).
2. C5 vs C4 for the target outline: 2 BETTER (v4, v6), 17 SAME, 1 WORSE (v2 over-split). T outline: WORSE in v1-6.
3. Worst frames: v2 (target cut by an arc), v1-v6 (T-cell merge), v14/v15/v20 (real tumour cells lost), v10-20 (dead yellow cell with no outline).
- Merges in C5: T-cell merges v1-6 (new in C5). v10: second T cell merged into orange in both.

---------------------------------------------------------------------
## 4. v09_MIP_ROI_29 (22 visits)

Scene: a cluster of 2-3 overlapping tumour cells at the lower-left of the frame (the frame edge cuts off the T cell and cells in several visits). Some tumour cells die (yellow-green inclusions) from about v13. One T cell throughout.

Per-visit notes
- v1: C4 blue = the small right cell (234, contact 0, not touching T). C5 blue = the large left cell that the T cell touches (575, contact 21); other cell grey. T: orange correct, cut by frame edge. BETTER.
- v2: Overlapped cluster: a bright green T-cell body at the left and two magenta cells. C4 orange is a small central piece; C5 orange extends to the left green body, which looks like the real T cell. C5 blue 561 vs C4 321: both messy, C5 seems more complete; ambiguous. SAME (ambiguous; T outline arguably better in C5).
- v3: Identical (636). Blue spans right cell; left lobe grey. SAME.
- v4: Identical (597). SAME.
- v5: C4 blue = both cells (1408, label tr=5 / T tr=2: ID jump). C5 blue = lower cell (689), upper cell grey; IDs tr=1/1. BETTER.
- v6: Identical (562). T cut by frame edge. SAME.
- v7: C4 merged (1184). C5 blue = lower cell (647) but with a thin horizontal sliver artefact at the cut (grey band), so the cut is not clean, and the small ring-shaped round object at lower right is inside blue in both. BETTER (with sliver artefact). C5 removed the small grey outline on the round magenta ring next to the T cell (a small ring-like object, could be debris; ambiguous).
- v8: C4 740, C5 380: C5 cut by a straight vertical line separating a left pale cell (grey) from the right bright cell plus an upper bleb. Plausible split of two cells but the line is perfectly straight. BETTER (moderate).
- v9: C5 blue 347 with a diagonal jagged cut and thin strips (XY and XZ); C4 925 is a merge. Cluster is hard to read, the C5 line follows no visible gap. WORSE.
- v10: C5 491 vs C4 773: both messy; C5 has a diagonal line across the left cell. SAME.
- v11: C5 400 vs C4 791: C5 blue = elongated upper arm; the lower-left round cell is grey. Ambiguous. SAME.
- v12: SAME (509 vs 686), both jagged; contact falls to 0 in C5.
- v13: C4 blue = lower-left dye-filled cell plus the upper ring-shaped lobe (808, merge). C5 blue = lower-left cell only (398, smooth, round). BETTER.
- v14: C4 blue = only the upper lobe (116 um3, fragment). C5 blue = lower dye-filled cell and the upper lobe in one smooth outline (795). BETTER vs C4. But note the extent of the C5 target switches between v13 (lower cell only) and v14 (lower + upper lobe): ambiguous whether the upper lobe is part of the same cell.
- v15: Identical (746). SAME.
- v16: C4 blue has jumped to the small lower cell next to the T cell (173, contact 22), a different cell from v15. C5 blue stays on the dye-filled cell + lobe (651, contact 0). BETTER (consistent, whole), but see the identity note below.
- v17: C4 blue = lower cell next to T (291, contact 271); C5 blue = same upper-left cell (667), contact 0. BETTER (whole).
- v18: C4 blue = lower cell near T (405). C5 blue = only the lower part of the elongated upper cell, cut by a horizontal line (170); upper part grey. WORSE (over-split, volume drops from 667 to 170).
- v19: C5 blue = the upper cell whole again (609, round, smooth); C4 blue = lower cell (287). BETTER.
- v20: C5 346 vs C4 356: C5 upper cell whole; C4 lower cell. SAME.
- v21: C5 538: upper cell (yellow-filled) outline plus a lower extension; C4 425 lower cell. SAME.
- v22: C5 313 upper cell; C4 367 lower cell. SAME.
- Dead dye-filled target: its outline is kept in C5 (v13-v22).

Identity note (important): from v12 on, C5 target and the T cell are not touching (contact 0.0 in C5 v12-v22), while the T cell is touching a different tumour cell (the one lower in the frame, which becomes bright yellow, dead, at v20-22; it has a grey outline in C5). C4 switches to that cell at v16-22 (contact > 0). So C5 stays consistently on the first cell (which was dye-filled earlier, v13-15, and looks like it was attacked earlier at v8-11) but by v16-22 it is not the cell the T cell contacts. If the analysis needs the cell currently in contact, C5 misses the second kill (cell yellow at v20-22); if it needs identity continuity, C5 is right. Ambiguous from images alone.

T cell (orange): the one T cell, correct in all visits (cut by the frame edge in v1, v5-v9, v18 etc.). v14-22: orange has some loops/holes around the bright overlap region (same in C4/C5).

Chamber verdict
1. Tracking correct? T cell yes (22/22). Target: C5 identity is consistent (tr=1 all visits) and whole in about 14/22 (v1, 3, 4, 5, 6, 13-17, 19-22); doubtful v2, 7 (sliver), 8-12 (cluster), 18; and from v12 onward the C5 target is not in contact with the T cell.
2. C5 vs C4: 9 BETTER (v1, 5, 7, 8, 13, 14, 16, 17, 19), 11 SAME, 2 WORSE (v9, v18).
3. Worst frames: v9 (jagged diagonal cut), v18 (cell cut by a horizontal line, volume 170), v7 (sliver artefact), v12-v22 (target not touched by T).
- Cells wrongly removed: only a small ring-like round object at v7 (ambiguous). Non-cells removed correctly: small round magenta debris at v16-17 (left, near the T cell) also lost its outline (ambiguous, could be a small cell).
- Merges in C5: v14 and v15 (upper ring lobe + lower dye-filled cell in one outline) possible; v10-11 messy.

---------------------------------------------------------------------
## 5. v09_MIP_ROI_38 (22 visits)

Scene: three tumour cells (upper, lower-right, later a left one) around one T cell that wanders; one small green object below. The tracked cell differs between columns: C4 follows the upper cell; C5 follows the lower-right cell, which the T cell reaches only from v7.

Per-visit notes
- v1: C4 blue = upper cell, irregular, with a hole and part missing (208, contact 0). C5 blue = lower-right cell, smooth, whole (646, contact 0). T: orange correct. BETTER (C4 outline incomplete).
- v2: C4 blue = upper cell (559, contact 3, fine outline, touches T). C5 blue = lower-right cell, whole (616, contact 0). Both outlines good; different cells. SAME.
- v3: C4 upper cell (603, contact 4.5); C5 lower-right (627, contact 0). SAME.
- v4: C4 upper (661); C5 lower-right (661). SAME.
- v5: C4 upper cell with a hole (328); C5 lower-right cell, smooth (679). BETTER.
- v6: C4 upper cell jagged (201); C5 lower-right whole (708). BETTER.
- v7: T now touches the lower cell. C4 blue = upper cell (229, contact 0); C5 blue = lower cell (738, contact 12.5). BETTER (the T cell touched cell is now the target).
- v8: Identical (813): blue = the cell touched by T, smooth. SAME. (C4 label tr=2, C5 tr=1.)
- v9: C4 blue = upper cell (299, contact 0, wrong); C5 blue = cell touched by the T cell (708, contact 52). BETTER.
- v10: C4 upper cell with holes (380, contact 0). C5 blue = touched cell, smooth (608, contact 120). BETTER.
- v11: C5 touched cell whole (662, contact 134); C4 upper cell (308). BETTER.
- v12: C5 blue = lower cell (889); in XY the blue outline wraps around the green T region because the tumour cell lies under the T cell in z (XZ shows the magenta cell beneath the green T cell, so it is not a merge but looks odd in XY). BETTER (C4 on upper cell, 352, fragment). Orange outline here includes an upper green-and-magenta region (second T or overlapped region); ambiguous.
- v13: C5 touched cell smooth (593, contact 39); C4 upper cell (652). BETTER (C5 is the cell in contact).
- v14: C5 lower cell below T, whole (668, contact 67); C4 upper cell with holes (287, contact 0). BETTER.
- v15: C5 blue = big left cell next to the T cell, whole (804, contact 54); C4 = upper-right fragment with holes (213). BETTER.
- v16: C5 left/lower-left cell whole (829); C4 upper cell with holes (149). BETTER.
- v17: C5 whole (675, contact 76); C4 upper cell (397). BETTER.
- v18: C5 whole (715); C4 upper cell jagged (318). BETTER.
- v19: Identical (756). T and cells form a bundle; the target appears as two lobes (heart shape) at the top; one or two cells is ambiguous. SAME.
- v20: C5 679 vs C4 918: both have a spurious blue loop lying inside the green T cell at lower right (a faint tumour overlay?) in addition to the left cell, so a second object is merged into the target (same in both columns). C5 additionally splits grey slivers from the left cell. SAME.
- v21: Identical (369). SAME.
- v22: Identical (708): blue = round upper cell with bright core touched by T. SAME. (Whether it is the same cell as v18 is not verifiable.)

Identity: C5 tr=1 for all visits (target ID stable). C4 shows labels tr=1, 2, 3, 2 and moves between cells. C5 follows the lower-right cell from v1 to v14 then the left cell from v15 (the cell positions change as the T cell drags cells around; the move from v14 to v15 involves a big positional jump that I cannot verify as the same cell).

Chamber verdict
1. Tracking correct? T cell correct 22/22. Target: C5 yields a whole smooth single cell at about 19/22 visits (exceptions: v19-v20 two lobes/inner loop; v12 XY looks odd); the target is the cell touched at v7-v18, but v1-v6 it is a cell the T cell does not touch (the upper cell is the one touching T at v2-3, in C4).
2. C5 vs C4: 14 BETTER (v1, 5, 6, 7, 9-18), 8 SAME, 0 WORSE. C4 outlines are repeatedly incomplete (holes, small volumes) and wrong cell.
3. Worst frames for C5: v20 (spurious inner loop), v19 (two lobes), v12 (blue wraps around T in XY).
- Cells wrongly removed: none seen. Non-cell debris: no relevant removals.
- Merges in C5: v20 (blue includes an extra object inside the T cell, in both columns); v19 possible two cells.

---------------------------------------------------------------------
## Overall summary table

Counts are per visit. "Tracking OK" = T cell correct AND target a single whole tumour cell that is plausibly the contacted cell with consistent identity in C5 (my estimate; cluster frames are marked ambiguous).

| Chamber | Visits | Tracking OK (C5) | C5 better / same / worse (target outline) | Serious problems |
|---|---|---|---|---|
| v07_MIP_ROI_26 | 22 | about 11/22 (T cell 22/22) | 10 / 10 / 2 (v3, v5; v4 also bad) | C5 jagged band cutting through both tumour cells at v3-5, sliver at v4, half-cell polygon at v9; target identity ambiguous v9-v14; over-split slivers in non-target left cell |
| v08_MIP_ROI_24 | 20 | 20/20 | 2 / 18 / 0 | none; faint haze correctly removed at v19 |
| v08_MIP_ROI_38 | 20 | target 20/20; T cell 14/20 | 2 / 17 / 1 (v2) | C5 merges neighbouring T cells into one orange outline at v1-6 (new in C5); target over-split by an arc at v2; C5 drops real tumour cells at v14, v15, v20 and the outline of a dead yellow cell (top left) in v10-20 |
| v09_MIP_ROI_29 | 22 | about 14/22 (T cell 22/22) | 9 / 11 / 2 (v9, v18) | jagged cuts and sliver artefacts (v7, v9, v18 cell cut by horizontal line, volume 670 to 170); from v12 the C5 target is not touched by the T cell, the touched cell (which turns yellow v20-22) is untracked |
| v09_MIP_ROI_38 | 22 | about 19/22 (T cell 22/22) | 14 / 8 / 0 | v19-20 two lobes and an extra loop merged into target (both columns); target in v1-6 is not the cell touching the T cell; v12 blue wraps T in XY |

Overall: C5 is clearly better than C4 in v09_38 and v07_26 (mostly where C4 either merged two tumour cells or kept a fragment), neutral in v08_24, and mixed in v08_38 and v09_29. Systematic C5 failure modes seen: (1) straight/jagged cut lines and thin slivers that do not follow a visible gap (v07_26 v3-5; v09_29 v7, v9, v18); (2) over-splitting a single round cell (v08_38 v2; v09_29 v18); (3) merging neighbouring T cells into one orange outline (v08_38 v1-6); (4) occasional loss of real tumour cells at the frame edge or touching a T cell (v08_38 v14, v15, v20) and of dead dye-filled cells (v08_38 top-left cell v10-20). Dead dye-filled tracked targets kept their blue outlines in every chamber.
