# P5 review, group 4 (C4 left vs C5 right). All frames viewed individually.

## v10_MIP_ROI_28 (21 visits)

Scene: one large green T cell contacting a cluster of 2-3 magenta tumour cells (upper-left cell "A", a lower cell "B", and later a dim/pale small cell "C"). No yellow (SYTOX) fill is visible in any visit; no obvious kill.

Per visit
- V01: T cell correct (orange) in both. Two touching tumour cells (upper A, lower B) in raw. C4 blue = A only, B grey (correct). C5 blue = A but with a jagged outline that is notched along its lower edge and wraps a spur along the T cell side; the straight horizontal cut between A and B in C5 is a hard line, not following the narrow gap. XZ: C5 blue has an extra inner grey circle (another fragment) inside/next to A. Volume 561 -> 335. C5 WORSE (jagged, partial A; C4 blue is smoother and whole A). No merge in either.
- V02: both identical; blue = A whole, B grey, T orange whole. SAME.
- V03: identical. SAME. (grey fragment in T cell in XZ in both, small.)
- V04: Dim pale magenta cell C near the T cell: C4 outlines it as a grey fragment with a hole; C5 removes it (no outline). Pale/blurred cell, so removal is acceptable (arguably non-cell). Blue A same. SAME on target.
- V05: identical. SAME. Blue A in XY has a small notch at T cell contact in both.
- V06: C4 has messy grey fragments across the T cell and pale cell C; C5 clean, no fragments. Blue and orange same. C5 BETTER (cleaner; non-cell/overlap fragments removed).
- V07: C5 removes C's grey outline (pale cell, fading); targets same. SAME (benign removal).
- V08: C5 removes grey fragments in T cell area. Same blue/orange. SAME / slightly BETTER (fragment removed).
- V09: C5 blue is split: an extra thin inner contour runs through the upper part of A (A appears cut into a thin upper sliver plus the main body; XZ also shows an inner outline). Volume 802 -> 449. So C5 over-splits/cuts across a round whole cell. C5 WORSE. C4 blue = whole A, correct.
- V10: identical; target A, T cell whole. The pale cell C (no outline in either). SAME.
- V11: identical. SAME. Target A whole, T cell whole.
- V12: T cell now touches A (top) and B (bottom, grey). Both same; blue A whole-ish, B grey. SAME.
- V13: identical; A blue at top, B grey at bottom. SAME.
- V14: identical; blue (top cell) outline irregular/partly under T cell in XZ in both. SAME.
- V15: T cell has moved; blue = upper-right large cell, B grey at lower-left. Identical in C4 and C5. Identity plausibly the same cell A (cannot be 100% sure after T cell shifted). SAME.
- V16: identical. SAME.
- V17: identical. SAME. Blue outline slightly jagged at the T cell side in both.
- V18: identical; blue is a larger irregular shape with a hooked lower edge, same in both. SAME.
- V19: DIFFERENT. C4 blue = small dim fragment (112 um^3) at upper right, the big cell A is grey = C4 target identity jumped to a small pale blob. C5 blue = the large cell A (618 um^3), including a jagged line across the T cell contact; plausibly the real contacted target. C5 BETTER (identity). C5 outline over the pale cell C is gone (pale cell visible unoutlined top-right).
- V20: C4 label target tr=2, C5 tr=1, but both blue on the same round tumour cell with bright nucleus touching T cell; outlines identical. SAME. Possibly a rounded/condensed cell (A?); ambiguous whether it is A.
- V21: C4 blue = small lower-right cell (154 um^3), upper cell grey. C5 blue = the upper, pale/fading cell sitting on the T cell (321 um^3), with an odd inner contour in XZ inside the T cell. Which is the "right" target is ambiguous; C5 identity is continuous with V20 round cell (plausible), C4 jumps to a different cell. C5 outline in XZ is an oval inside the T cell (over-small/inside T cell). Mixed: identity BETTER, outline quality SAME/WORSE.

Wrongly removed by C5: none that matter (only the pale blurry cell C in V04-V08 and the fading cell in V19/V21 lower-right). No dead yellow cell present. Non-cell objects: haze strips in lower-left (large grey irregular object) remain in C5 in both V01-V09 (not removed). Merges remaining in C5: none seen (A and B are always separate in blue vs grey).

Verdict v10
- Tracking: T cell correct all 21. Target correct V01-V18 in both; V19 and V21 C5 better in identity; V20 same.
- C5 vs C4: mostly SAME; BETTER V06, V19 (V21 identity); WORSE V01, V09 (jagged/over-split of A).
- Worst frames: V09 (A cut with inner contour, 802->449 um^3), V01 (jagged A, 561->335), V21 (ambiguous target, inner XZ oval).

## v11_MIP_ROI_15 (22 visits)

Scene: a crowded, hard-to-read chamber. A cluster of 2-3 touching magenta tumour cells plus a green T cell that is elongated/deformable (not a clean round cell) and, in late frames, appears to have green material inside tumour cells and a second green blob (V13, V20-22 may be two T cells; ambiguous). Bright yellow-white patches inside the green/magenta at V20-V22 (top-left blob V20, top blob V22) may be dye-positive (dead) cells; the image is too noisy to be sure. Nearly every frame is ambiguous about which tumour cell is "the" target. Note: C4 label shows T cell tr=2 / target tr=2 in several visits (ID changes in C4), C5 is tr=1/tr=1 throughout, i.e. C5 IDs are consistent by label.

Per visit
- V01: T cell correct (green cell below the cluster) in both. C4 blue = a big shape that spans 2 tumour cells (left bright one + right one) = MERGE (781 um^3). C5 blue = only the right-hand cell (421 um^3), split along a straight vertical line, the left cell grey. The right cell is the one touching the T cell, so C5 target is correct and whole. C5 BETTER (clear merge fixed), cut line is straight/artificial though.
- V02: no contact yet (0.0 um^2 in C5, T cell below). Cluster of 3 cells. C4 blue = a large outline covering two cells (872 um^3) with an inner hole; C5 blue = upper-right cell with irregular lobes, extra grey cells over the left/bottom cell. C5 blue wiggles through the middle of the bright central cell (jagged, notched). Ambiguous which is correct. C5 SAME/slightly WORSE (jagged, inner contours).
- V03: T cell correct (round green). Three tumour cells. C4 blue spans the upper-right cell and part of the left cell (merge, 875 um^3). C5 blue = fragmented irregular outline snaking through T-cell contact region plus left cell, jagged, with grey outlines overlapping in XZ (a straight line cuts through the upper right cell). Ambiguous; C5 outline jagged and cuts across cells. WORSE (jagged, cuts through cell) though merge gone.
- V04: T cell is a narrow green cell between tumour cells. C4 blue = left cell + upper right cell merged (1110 um^3). C5 blue has straight-line polygon-like cuts across the left cell (a horizontal and a diagonal straight line) and wraps around the T cell; target = left cell plus a strip of right cell. Artificial straight cuts, jagged. WORSE (cuts across cells along straight lines, not along gaps). Merge not clearly fixed.
- V05: T cell outline fragmented in both (orange covers only the bright part; a dimmer green region left of it is not outlined). Target blue = right-hand cell in both (390 um^3). Left big cell is grey; C5 cuts it into two (a vertical line), looks like 2 lobes, plausibly two cells. SAME for target. Identical volume.
- V06: T cell (green, elongated left) fine in both. Blue = right elongated cell. C4 has an extra grey circle inside the right cell (partial inner object); C5 does not, blue 688 um^3 whole. C5 splits the upper-left tumour cell into two grey pieces (vertical line; looks like it could be one cell, ambiguous). C5 BETTER on target (no inner fragment).
- V07: very tangled. Blue in both is a long snaking outline across cells above and right of the T cell; C5 has an extra straight diagonal line in XZ. Target volume 712 vs 466. Neither is clearly one cell; C5 appears to have cut off the right grey blob. SAME (both bad).
- V08: identical. Blue = right small bright cell, T cell correct. SAME. Pale lower-left cell grey in both.
- V09: identical. Blue = right cell. T cell large green top-left. SAME.
- V10: identical. SAME. Blue = bright right cell, good.
- V11: identical. Blue = upper right cell; irregular but plausible. SAME.
- V12: T cell now separate (no contact, 0.0 um^2 in C5). C4 blue = large cell right (650 um^3) with a notch; C5 blue = right cell but split by an internal contour at lower right (330 um^3); the grey part at the bottom is part of the same big cell as far as I can see (raw shows one large round magenta cell). C5 over-splits one cell. WORSE.
- V13: T cell is the green upper-left blob; C4 says T cell tr=2 (changed). Blue C4 = right elongated cell plus part of lower-left. C5 blue = lower-left cell (430 um^3) with the right elongated cell grey. The target changed: C5 target is now the left cell, C4 the right cell. Identity ambiguous (both touch T cell). C5 outline stays jagged at T cell contact. SAME/ambiguous.
- V14: T cell (orange) tr=2 in C4. In C5 blue is the small left-lower cell with a tail (273 um^3) AND an inner contour in the T cell region; right big cell with green puncta is grey. C4 blue = the big right cell plus inner outlines. Target identity differs between C4 and C5, both plausible. C5 blue outline has a long tail and inner overlapping contour (jagged). WORSE/ambiguous.
- V15: T cell and tumour cell overlap heavily in the raw (green inside magenta). C4 blue = large outline including the lower-left magenta tail; C5 blue = upper cell with a straight line through it and part of it split off (grey), 443 vs 834 um^3. Ambiguous. C5 has straight-edged cuts again. SAME/WORSE.
- V16: identical mostly. Blue has a long snaking extension; grey objects left. SAME (both poor).
- V17: identical. Blue = a bean-shaped extended outline across the left cells with inner contours; T cell right. SAME.
- V18: T cell outline in C5 has an extra inner contour inside T cell (small closed loop), C4 does not. Blue similar. C5 slightly WORSE (spurious inner contour in the T cell, in both XY and XZ).
- V19: C4 blue = a ragged left shape (193 um^3), many grey; C5 blue = left cell plus a bigger region including green puncta (471 um^3). Raw: pale magenta fragmenting cells; ambiguous. SAME (ambiguous).
- V20: two green blobs: orange outline covers both the lower elongated T cell and a bright top-left green blob in both C4 and C5 (extra T cell merged into orange? or bright yellow-green dead/dye-positive patch). Identical C4/C5. SAME. T cell identity questionable (orange may contain 2 objects).
- V21: identical. Blue = a snaking multi-lobe outline around 2-3 pale magenta fragments (299 um^3); T cell large green bar on top. Target not a single cell. SAME (both bad).
- V22: identical. Messy; bright yellow-green top patch inside orange (likely dye-positive/overlap). SAME.

Wrongly removed by C5: nothing obvious; I could not identify a dead dye-filled target that lost its outline (yellow patches at V20-V22 retain outlines in both). Non-cell objects removed: the small green blob at the upper right in V01-V05 (grey in C4 V01) is outlined in both in V01 and then not outlined; unclear. Merges still present in C5: V02, V03, V07, V15, V17, V21 (blue spans more than one cell or a ragged multi-lobe shape).

Verdict v11
- Tracking: T cell plausibly correct in most visits (V01-V19) but unclear in V13-V22; target identity is ambiguous after V12 (cluster fragmenting, which cell is the target cannot be decided from the picture). C5 labels are stable (tr=1/tr=1), C4 flips to tr=2.
- C5 vs C4: BETTER V01, V06 (merge fixed, inner fragment gone); WORSE V02, V03, V04, V12, V14, V18 (straight-line cuts, over-split, inner contours); the rest SAME (many identical, mostly because both fail).
- Worst frames: V04 (straight-line cuts across left cell), V03, V12 (round cell cut in two), V14, V18 (inner loop in T cell).

## v12_MIP_ROI_23 (22 visits)

Scene: a single small magenta tumour cell that is progressively engulfed by a very large, dim, flat green cell (probably the T cell, spreading over the tumour cell; bright green dots along an arc and a left-side clump are debris/beads, not cells). The magenta signal shrinks, splits in two (V15-V16), becomes a thin streak (V17-V19) and then only faint remnants (V18-V22), i.e. the tumour cell looks like it is being destroyed/consumed. No yellow SYTOX fill is visible. This chamber has the most serious C5 problem: C5 loses the target for most of the series.

Per visit
- V01: T cell (orange) = the large dim green cell covering the magenta cell, same in both. C4 blue = magenta cell (173 um^3). C5: NO blue target (tr=0, vol 0.0), the clearly visible bright magenta cell has no outline at all. C5 WORSE: tumour cell wrongly removed.
- V02: both same: blue = whole magenta cell, orange = dim green cell, labelled contact 6 um^2. SAME. (C5 recovers target here.)
- V03: same in both, blue whole cell with a small notch at the T-cell edge. SAME.
- V04: same in both. SAME.
- V05: C4 has blue on the magenta cell with a ragged lower-left edge; orange (T cell tr=3) is wrongly on the left debris clump (not a cell). C5: no blue (tr=0), no orange; magenta cell without outline; left clump grey. T cell absent in this frame in the picture (the big green cell is off screen/not visible; faint green at left edge). C5 WORSE for target (cell removed) but BETTER for T cell (does not call debris clump a T cell).
- V06: magenta cell blue in both, whole, with ragged lower edge. C4 orange = debris clump (wrong, not a T cell), C5 grey (correct, not tracked as T cell). C5 BETTER (correct to not label debris as T cell). Target same.
- V07: same story: C4 orange on debris clump (wrong), C5 grey. Blue same. C5 BETTER (no false T cell). Also a faint green object near the cell grey in both.
- V08: the large dim green cell is back (orange, right in both). C4 blue = magenta core (153 um^3) with a ragged inner outline; C5 no blue, no target (tr=0, 0 um^3), cell visible bright magenta in raw. C5 WORSE (target removed).
- V09: orange in both covers the large dim green region (and in C4 part of the left debris clump is merged into the orange outline in both C4 and C5, so orange contains debris). C4 blue = magenta core, ragged; C5 none. C5 WORSE.
- V10: T cell round dim green, tumour cell magenta inside. C4 blue = magenta core (72 um^3), C5 none. C5 WORSE.
- V11: same, magenta core now two spots; C4 blue encloses both, C5 none. C5 WORSE.
- V12: magenta cell compact and bright inside the green cell; C4 blue tight around it, C5 none. C5 WORSE.
- V13: same. C5 WORSE.
- V14: same, magenta cell elongated, C4 blue tight; C5 none. C5 WORSE.
- V15: magenta now two lobes (dividing or fragmenting); C4 blue wraps both lobes (86 um^3); C5 none. C5 WORSE.
- V16: two separate magenta bodies; C4 blue on upper one, lower one grey; C5 none for both. C5 WORSE (both bodies unoutlined).
- V17: magenta dumbbell/streak; C4 blue follows it; C5 none. C5 WORSE.
- V18: only a faint magenta streak; neither has a target (tr=0 in both). SAME.
- V19: green cell now long and spread; magenta remnants (a vertical streak top, another small streak lower). C4 blue (21 um^3) on the lower streak; C5 none. Remnants are faint, so ambiguous. SAME/ambiguous.
- V20: both: no target; two faint magenta spots inside the orange cell; T cell outline in both shows a weird shape connecting two cells (long orange object including holes in lower-left). T cell outline may contain 2 green cells (merged). SAME.
- V21: both: no target, orange on the right green object, left green object grey (second large green cell, so >1 T cell in the field). SAME.
- V22: C4 orange on the right object (tr=5); C5 orange swapped to the left object (tr=1), and the right object is grey. So C5 T-cell identity follows the left cell which has the magenta streak at the bottom. Ambiguous which is right. SAME/ambiguous.

Wrongly removed by C5: the tumour cell in V01, V05, V08-V17 (12 of 22 visits), including clean bright magenta cells with no overlap; this would break any measurement of this chamber. Correctly removed: nothing is falsely a T cell in V05-V07 in C5 (C4 tracked debris as T cell). Merges: none for the target; orange (T cell) outline merges debris clump with the big cell in V09 in both.

Verdict v12_23
- Tracking: T cell (the big dim green cell) correct in V01-V04, V08-V19 in C5; V05-V07 the T cell is not visible (C5 correct to show none, C4 wrong: debris tracked). Target tracking in C5 FAILS in 12/22 visits (no target). C4 target ok in V01-V17.
- C5 vs C4: WORSE in V01, V05, V08-V17 (12 visits, target removed), BETTER in V06, V07 (debris not tagged T cell); SAME otherwise.
- Worst frames: V12-V14 (bright clean magenta cell inside T cell completely unoutlined in C5), V01, V08.

## v12_MIP_ROI_32 (22 visits)

Scene: one large magenta tumour cell (lower right at first, later drifting upward) contacted by a big flat/irregular green T cell. Several other small round green cells/blobs (second T cells or debris; some outlined grey) and a ring of bright green debris dots. The tumour cell stays single and whole through the series (no merge, no kill visible; a whitish bright core inside the magenta cell at V19-V22 is overlap of green/magenta, not clearly yellow dye; ambiguous).

Per visit
- V01: C4 orange = a small dim round cell at upper right (NOT the T cell; the real T cell, large green flat cell touching the tumour cell on its left, is grey in C4). C5 orange = the large green T cell (correct), blue = whole tumour cell in both (709 um^3, identical outline). C5 BETTER (T cell identity fixed). C5 also removes the small dim round cell's outline from tracking (it stays grey).
- V02: C4 again no orange on the real T cell (orange small round cell upper right, T cell grey, contact 0.0). C5 orange = big green T cell (contact 31.7) but the C5 orange outline also has an extension up to a smaller green object (merged small cell, a bridge in the upper middle); minor. Blue same in both. C5 BETTER (T cell), slight orange over-merge.
- V03: C4 orange on a small green fragment at the top (wrong), T cell grey. C5 orange = the large green T cell (correct). Blue same. C5 BETTER.
- V04: C4 orange = small fragment at top; the T cell is grey. C5 orange = the T cell (a tall green cell touching the tumour cell). In C5 blue has an inner grey notch along the contact edge (the lower part of the tumour cell outline is bitten by a grey piece) but follows the cell. C5 BETTER overall (T cell), blue slightly irregular at the lower contact edge.
- V05: C4 labels T cell tr=2, orange = the right long green cell; C5 tr=1, the same outline. Identical. SAME. Blue = whole tumour cell.
- V06: identical. SAME (T cell and target correct).
- V07: T cell lies under the tumour cell: both outlines cross (orange border jagged through the blue cell). Identical in C4 and C5. SAME. Blue = whole tumour cell; the T cell outline includes a horizontal green region at left.
- V08: identical, clean. T cell large green on the left, grey sliver inside it in both. SAME.
- V09: identical. A grey inner outline in the T cell in both. SAME.
- V10: identical, clean. SAME.
- V11: identical. SAME.
- V12: identical. SAME.
- V13: identical, clean. SAME. (a faint round dim green cell and a round bright green cell remain unoutlined in both; those are not touching.)
- V14: identical. T cell is the tall narrow green cell, blue whole. SAME.
- V15: identical. The T cell orange outline includes a small round green cell at lower left joined to the big flat cell (probable merge of two green cells into one orange outline). Same in C4 and C5. SAME.
- V16: identical, clean. SAME.
- V17: identical. SAME.
- V18: identical, clean. SAME.
- V19: identical. Orange covers the big green cell plus a bright round green cell on the left connected by a narrow bridge (two green cells in one outline). SAME (merge of T cells in both).
- V20: C4 orange is a jagged fragment overlapping the tumour cell's lower-left (contact 19.7); C5 orange = large green cell at left (cleaner, contact 0.0), with a second grey region. C5 BETTER (T cell outline cleaner and not cutting into the tumour cell). Blue identical.
- V21: identical: orange covers the long green flat cell plus the very bright round green cell at right (two green cells merged). Blue whole, includes bright core. SAME.
- V22: identical. Same merged orange. SAME.

Wrongly removed by C5: none (target kept in every visit). Non-cell objects: C5 and C4 keep the same grey debris/blobs; C5 does not use debris as T cell. Merges: T-cell (orange) merges of two green cells at V15, V19, V21, V22 in both C4 and C5 (not a tumour merge). No tumour merge.

Verdict v12_32
- Tracking: target correct 22/22 in both. T cell correct in C5 22/22 visually except merged-two-green-cell outlines at V15, V19, V21-V22 (and V02 small bridge); C4 has the wrong T cell (a small dim cell or fragment) at V01-V04 (and V20 jagged).
- C5 vs C4: BETTER V01-V04, V20; SAME everything else; never worse.
- Worst frames: V15/V19/V21/V22 (orange merges two green cells, in both C4 and C5), V02 (C5 orange bridge to a small cell).

## v13_MIP_ROI_28 (22 visits)

Scene: a very noisy chamber. One very large, dim magenta tumour mass (looks like one big blebbing cell or 2-3 cells; 2-3 bright spots inside) and one green T cell that is elongated or dumbbell-shaped, often dim, sometimes almost gone. The image is ambiguous throughout about how many tumour cells there are; I can only say where C5 cuts a cell that looks like a single round cell. C5 target volumes swing between about 250 and 1200 um^3 while C4 stays 550-1640 um^3.

Per visit
- V01: T cell = elongated bean-shaped green cell lower left, correct in both, not touching (0.0 um^2). Target = big round magenta cell upper right. C4 blue = the whole cell plus a tail toward the T cell (1018 um^3). C5 blue = the left part only, cut by a curved line running through the cell (562 um^3), right piece grey; in XZ the C5 blue outline is jagged and fragmented. C5 over-splits a round cell: WORSE.
- V02: T cell correct, touching tumour tail. C4 blue = whole cell with tail (1486); C5 blue = same but right ~20% cut off as grey (1202). C5 WORSE (cut across the cell, mild).
- V03: T cell (orange) correct, long narrow shape on the left. Target dim and overlaps the T cell; C4 blue wraps the magenta region; C5 blue slightly smaller with grey pieces at upper right (396 vs 586). Ambiguous; SAME/slightly worse. Marked SAME.
- V04: T cell is a small bright green patch at the left edge of the magenta mass (orange, correct in both). C4 blue = whole wide magenta mass with holes; C5 blue = left 2/3 with a straight vertical edge, right third grey (456 vs 752). Raw shows a bright spot at the right end, possible second cell; ambiguous. Marked WORSE (straight artificial vertical cut).
- V05: identical outlines. T cell small green at bottom. Target is a fragmented-looking mass; blue and grey pieces tile it. SAME (ambiguous).
- V06: C4 orange = a small green fragment to the right (dim object), C5 no T cell (tr=0), the green is grey. The green is dim and not touching, so neither is clearly right. Target: C4 blue = whole mass (1232) with inner grey pieces; C5 blue = middle slice only (628) bounded by two straight vertical lines, left and right parts grey. Raw looks like one large cell with 2-3 bright lobes. C5 WORSE (straight cuts).
- V07: T cell = dim green cell at right (both orange), correct. C4 blue = whole mass (1539), C5 blue = narrow middle slice (473) with straight vertical edges, the left 60% (with two bright lobes) grey. C5 over-splits: WORSE.
- V08: T cell tiny at lower right: C4 orange fragment, C5 none (tr=0). Target: C4 blue whole (1602); C5 blue = a diagonal slice (631) with a jagged boundary. C5 WORSE.
- V09: identical, T cell (bright small green) at the bottom touching the lower magenta blob; blue = lower part, round cell at upper left grey (a separate round cell, plausible). SAME.
- V10: T cell not visible (tr=0 in both). Blue in both = large jagged outline with grey pieces; C5 has an extra horizontal straight cut (593 vs 789). SAME/slightly worse. Marked SAME.
- V11: T cell = long thin green cell at left, correct in both, touching the mass at the lower left. Blue: C4 wraps most of the mass (668), C5 has a thin strip along the upper left and more grey pieces (435). Both jagged. SAME.
- V12: T cell green at left, correct. Raw: one mass with a bright round lobe at upper right. C4 blue = whole mass (994) with holes; C5 blue = mass minus the upper-right lobe (cut by a straight line, 591). C5 WORSE (cuts off the bright lobe).
- V13: T cell dim green at left not touching, correct. C4 blue = whole mass (1051), C5 blue = smaller (625) with a straight horizontal cut in XZ and grey interior. C5 WORSE.
- V14: T cell correct, touching. C4 blue (648) vs C5 blue (397) with grey on the left and interior; C5 more jagged. C5 WORSE.
- V15: T cell small dim blob left: C4 orange, C5 none. Target: C5 blue has a straight diagonal line in XZ across the mass (375 vs 646). C5 WORSE.
- V16: T cell = green cell above the mass, touching, correct in both. C4 blue = whole mass (1397); C5 blue (1018) with grey inner piece at the bottom of the mass. C5 slightly WORSE (cut).
- V17: identical. T cell large green at left, touching. Blue 553 in both. SAME.
- V18: T cell large green left, touching the mass. Blue = narrow slice at the contact edge in both (672 vs 440 for C5), grey remainder to the right. Ambiguous. SAME.
- V19: T cell tiny/dim at the right edge (C4 orange small, C5 none). Raw: one round magenta cell with a ring-like rim. C4 blue = the whole round cell (1637). C5 blue = right half only (452) with the left half grey = clear over-split of a single round cell. C5 WORSE (worst frame).
- V20: identical. T cell at left, touching bright spot. Blue = irregular right part, grey remainder. SAME (ambiguous).
- V21: identical. T cell has a dim extension toward the mass. Blue is a snaking outline on the right; many grey pieces. SAME (both poor).
- V22: T cell = dumbbell-shaped green (big round + narrow tail) in both, correct. C4 blue = a jagged self-looping scribble with inner loops (163); C5 blue = single clean lobed outline around the right bright cell (310). C5 BETTER.

Wrongly removed by C5: no dead dye-filled target removed (no yellow visible). C5 loses the T cell (tr=0) in V06, V08, V10, V15, V19 where the green object is dim/tiny (C4 tracks a small dim green fragment in V06, V08, V15, V19); ambiguous whether those are real T cell remnants. Non-cell objects: none clearly removed. Merges still present in C5: not demonstrable; the main C5 error here is over-splitting (a straight cut or curved line through what looks like one cell: V01, V02, V07, V12, V19).

Verdict v13
- Tracking: T cell correct in 14-15 visits in both; dim/absent in V04-V06, V08, V10, V15, V19 (C5 correctly shows none or ambiguous). Target identity (the one big magenta mass) is consistent in both, but the C5 target is only a piece of it in many visits.
- C5 vs C4: WORSE in 12 visits (over-splitting a single large cell with straight/curved cut lines), BETTER in 1 (V22), SAME 9.
- Worst frames: V19 (round cell cut in half), V07 (60% of the cell dropped to grey), V01, V06, V12.

# Overall summary table

| Chamber | Visits | Tracking OK (C5, my reading) | C5 better / same / worse | Serious problems |
|---|---|---|---|---|
| v10_MIP_ROI_28 | 21 | 20/21 (V21 ambiguous); C4 jumps to a small pale cell at V19, V21 | 3 / 16 / 2 | V09 target cut by an inner contour (802 -> 449 um^3); V01 jagged partial outline. Otherwise C5 matches or beats C4. |
| v11_MIP_ROI_15 | 22 | about 11/22 clear (V01-V11); V12-V22 ambiguous (fragmenting cluster, 2 green blobs) | 2 / 14 / 6 | Straight-line cuts across cells (V04), round cell cut in two (V12), spurious inner loop in T cell (V18), merges persist V02-V03, V07, V15, V17, V21; unreadable late frames. |
| v12_MIP_ROI_23 | 22 | 5/22 clearly OK in C5 (V02-V04, V06-V07); 12 visits with target missing; 5 visits (V18-V22) no visible target in either | 2 / 8 / 12 | C5 drops the target tumour cell (tr=0, no outline) in V01, V05, V08-V17 even though it is a bright clean magenta cell inside the big green T cell; C5 correct only in not tagging debris as T cell (V05-V07). |
| v12_MIP_ROI_32 | 22 | 22/22 target; T cell correct but orange merges 2 green cells in V15, V19, V21, V22 (C4 and C5 alike) | 5 / 17 / 0 | None caused by C5. C5 fixes wrong T cell (C4 tracks small dim cell or fragment) V01-V04, V20. |
| v13_MIP_ROI_28 | 22 | about 15/22 (T cell dim/absent 7 visits; target mass consistent) | 1 / 9 / 12 | C5 over-splits the large dim tumour cell with straight/curved cuts (V01, V02, V07, V12, V19: 40-60 percent of the cell dropped to grey); volumes unstable 250-1200 um^3. |

Overall: C5 is a clear improvement only for chamber v12_MIP_ROI_32 and mostly neutral/positive on v10. In v11, v12_MIP_ROI_23 and v13 C5 is clearly worse than C4 (target loss in v12_23; over-splitting/straight-line cuts in v11 and v13). No chamber showed a yellow-filled dead target losing its outline.
