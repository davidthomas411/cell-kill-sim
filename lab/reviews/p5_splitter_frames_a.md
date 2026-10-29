# P5 review, group 1 (C4 vs C5), chambers v01_MIP_ROI_2, v01_MIP_ROI_32, v02_MIP_ROI_3, v02_MIP_ROI_32, v03_MIP_ROI_6

Every visit frame (20 per chamber, 100 images) was opened. "XY"/"XZ" = max-projection row; "L"/"R" = left (C4) / right (C5) column. No yellow SYTOX fill was clearly visible in v01_MIP_ROI_2 at any visit (no visibly dead cell).

---------------------------------------------------------------
## v01_MIP_ROI_2  (one large green T cell, one magenta tumour cell, one second T cell parked top-left, grey)
---------------------------------------------------------------
General: C4 and C5 are IDENTICAL in 15/20 visits (same outlines, same volume text). C5 differs in v05, v08, v12 (cleaner, larger target) only. Tracking ID is "tr=1 / tr=1" throughout in both.

Per visit
- v01: T cell (orange) = real large green cell. Blue = the magenta cell above it that it touches; one whole cell, but blue outline is elongated/spiky upward (includes diffuse magenta haze). C5 = C4 (SAME). XZ: blue outline overlaps T cell region in both.
- v02: same, T cell correct; blue is the single magenta cell with holes/lobes at the right edge (XZ small detached blue lobe at far right). SAME. Top-left second T cell correctly grey in both.
- v03: T cell correct. Blue now extends up-left and its upper-left lobe touches/overlaps the grey second T cell (XY, top-left; the blue boundary runs along the grey T cell). Same in L and R. SAME. Blue is a large diffuse polygon wider than the bright magenta body (haze included).
- v04: T cell correct; blue big and irregular (spike pointing up, lobe left), includes diffuse low-intensity magenta. SAME.
- v05: NOTABLE. Cell is now pulled under/inside the T cell. L (C4): many grey fragments inside and below the T cell outline (XY and XZ), plus a separate grey magenta piece at top. R (C5): grey fragments removed, the top magenta blob (debris/bleb, grey in C4) removed; blue outline is one clean contiguous cell wrapping the lower-left of the T cell and the bright magenta spot; target volume 551 -> 788 um^3. C5 BETTER (cleaner, one whole cell, no fragment clutter). Removal of top magenta blob looks correct (faint/diffuse, not a cell). Tracking OK.
- v06: T cell outline in both = large irregular green region with a small enclosed orange hole near the magenta core. Blue = the magenta cell around/behind it. SAME. Outline passes through the T cell (blue covers green T-cell pixels at XY centre) - same in both.
- v07: T cell is now fragmented-looking (orange outline irregular, small separate top lobe, grey sub-outlines inside blue in L). C5 removes the grey interior fragments inside the blue region (L shows several grey slivers around 270,230 and 310,300; R still shows two small ones). Mostly SAME, marginally BETTER. Target is faint magenta haze here (cell is dim/diffuse); blue outline large and soft-edged - ambiguous whether blue over-covers.
- v08: C4 blue is smaller (458 um^3) with a separate GREY lobe at left (top-left lobe of the same magenta cell); C5 includes that left lobe in the blue outline (788 um^3), giving one whole cell. C5 BETTER. T cell outline in both is broken into two pieces joined by narrow neck (large lower body + small upper lobe + inner jagged bit); same in both.
- v09: SAME. T cell in two lobes (top and bottom) joined by a narrow neck; blue is the magenta cell between them. Fine.
- v10: SAME. T cell orange and blue side by side, contact 0; blue = one magenta cell with a notch where it meets T cell. Correct.
- v11: SAME. Blue magenta cell; orange has a small upper-left lobe that is a separate green patch (part of T cell membrane or second small green object) - same L/R.
- v12: C4: grey sliver inside blue at top and XZ (small grey outline), target 435 um^3; C5: grey removed, blue one cell, 582 um^3. C5 slightly BETTER. T cell outline very irregular with enclosed holes in both.
- v13: SAME. Blue = magenta cell left of bright magenta streak; the right part of the bright magenta (right edge) is outside blue in R (slightly) - tiny difference. SAME.
- v14: SAME. Blue has an enclosed hole at left (circle) - same in both. T cell orange has an enclosed bit around bright magenta core.
- v15: C4 has a small grey sliver inside blue (top right); C5 slightly cleaner. SAME / marginally BETTER.
- v16: SAME. Blue cell on the left, T cell overlapping right side. Correct.
- v17: SAME. Blue has an arm extending to the left (XY) wrapping a green blob at left edge; T cell outline sits partly over the magenta core. Ok.
- v18: SAME. T cell large green; blue magenta cell to its upper right. Correct.
- v19: SAME. Orange T cell wraps around blue cell on left (concave); blue has small hole at top. Fine.
- v20: SAME. T cell stretched into a long tail (extends off bottom of the panel); blue magenta cell upper right. Fine.

(a) Removed: nothing wrongly removed in this chamber. Removed in C5: grey fragment clutter (v05, v07, v12) and the faint magenta blob top of v05 (non-cell debris, correct).
(b) Merges still present in C5: none obvious (a single tumour cell). The blue outline in several visits (v03, v04, v06, v07) is larger than the bright magenta body (includes dim haze) in both C4 and C5.

Verdict v01_MIP_ROI_2
1. Tracking correct? YES, 20/20 visits: orange = real big green T cell, blue = the one magenta tumour cell, consistent id.
2. C5 vs C4: mostly identical (15/20 identical). Where different (v05, v08, v12, marginally v07, v15) C5 is better: fewer fragments, left lobe merged into target (v08).
3. Worst frames (both methods): v06-v09 where the T cell wraps over the tumour cell and outlines overlap/irregular; v07 where the target is dim.

---------------------------------------------------------------
## v01_MIP_ROI_32  (crowded: one large green T cell on a cluster of 2-3 magenta tumour cells; main cell on right/centre, a second bright cell lower-left, grey in both)
---------------------------------------------------------------
General: orange = real green T cell in all visits (OK). Blue ("target") sits on the central/right tumour cell but in C4 repeatedly spills into the lower-left tumour cell (merge). C5 target volume is ~25-35% lower than C4 in every visit (e.g. 1145->857, 1775->1119, 963->614 um^3). C5 introduces many extra grey sub-outlines inside the cluster and under the T cell (over-splitting), and several of its cuts are straight lines that run across a cell body. This chamber is hard to judge: the tumour cells are overlapping in projection; where I say "ambiguous" I could not tell by eye.

Per visit
- v01: T cell OK. C4 blue = right-hand big magenta cell; lower-left bright cell grey. C5: a right-hand lobe of the same magenta mass (XY ~ x 700-860, y 310-400; XZ right side) is cut off into a grey outline, blue shrinks to 857. The magenta body looks continuous with blue, so this looks like an unjustified cut. WORSE (ambiguous: the lobe could be a separate cell).
- v02: C4 blue is HUGE (1775 um^3) and covers both the lower-left bright cell and the central cell (a merge, XY and XZ). C5 (1119) turns the bright lower-left blob into a separate grey object and blue is the centre/top-left cell: BETTER (merge broken). The split line is a nearly horizontal straight cut (XY y~290) rather than a gap-following boundary.
- v03: C4 blue = upper-left magenta + diagonal band, 1370. C5 blue is a diagonal band with a straight diagonal edge running through the middle of the T cell footprint, plus many grey slivers inside the T cell outline (XY and XZ). Contact falls 147 -> 17 um^2. WORSE (jagged/straight cut across, more fragmentation). The lower-left cell is grey in both.
- v04: C4 blue = upper-left cell only (963). C5 blue 614, similar shape, but the lower magenta mass (bottom of XY) is now cut into 3 grey pieces vs 1 in C4, and extra grey outlines under the T cell. In XZ C5 blue is a smooth round cell that follows the magenta body well (better than C4 XZ). Mixed: SAME overall, with over-splitting of the lower cell.
- v05: blue = left cell with green spot (both). C5 adds grey straight lines through the lower-left of the XZ magenta body (x 560-650, y 640-700) and splits the lower XY mass into more grey pieces. SAME / slightly WORSE (over-split).
- v06: C4 blue 901 = left cell, irregular. C5 658: blue is cut from the grey interior by a straight diagonal line (XY ~ (690,230)->(640,290); XZ ~ (610,545)->(730,620)), so a big central part of the magenta body becomes grey and not part of target. WORSE (straight cut across a cell).
- v07: C4 blue 1140 (wraps around left/bottom). C5 870: blue similar but left strip (XZ x 610-650) and parts of the interior are separate grey; straight vertical/horizontal edges at the lower-left. SAME to slightly WORSE.
- v08: C4 blue 1311 covers whole large magenta cell. C5 977: in XY a large grey ellipse sits in the middle of the cell, so blue becomes a crescent along the left edge plus a thin strip under the T cell - the tracked target no longer covers the main cell body in XY. XZ blue still covers the body. WORSE (XY).
- v09: C4 blue has a downward tail into the lower-left cell (XY x 180-270, y 340-410): a merge of two tumour cells. C5 drops the tail (grey) - blue 1132: BETTER, merge broken. C5 also adds a few internal grey outlines.
- v10: C4 986 vs C5 700. XY C5 blue is cut by a grey wedge along the T-cell border and bottom edge; XZ C5 blue is a cleaner rounded lower cell. SAME.
- v11: SAME in shape; C5 blue (702) excludes top-left strip and small inner region (grey inside). SAME.
- v12: C4 blue includes a lower-left extension (XY x 190-260, y 360-425) = part of the lower-left tumour cell (merge). C5 removes it (grey). BETTER.
- v13: C4 blue again has the lower-left extension (x 170-250, y 350-425). C5 keeps a small bump (x 680-740, y 340-385) but is mostly separated. BETTER (partial).
- v14: C4 blue (906) has an upward spike at top right; C5 (600) turns this thin strip grey and has a cleaner top edge, but in XZ C5 has a rectangular notch on the left (x 690-720, y 575-610). SAME.
- v15: SAME in XY; C5 XZ has a jagged rectangular bite at upper-left of blue. SAME / marginally WORSE.
- v16: Both blues still include the lower-left tumour extension (merge still present in C5; XY x 660-710, y 365-415). SAME. C5 volume 598 vs 885.
- v17: Both blues still include the bottom lower-left tumour piece (XY ~x 630-720, y 360-425). Merge still present in C5. SAME. Contact area 37.8 -> 4.5 in C5 (blue pulled away from T cell by internal cuts).
- v18: SAME; C5 blue has grey straight strip cut at left/bottom edge (x 710-740, y 300-390). SAME.
- v19: C5 cuts a right-hand strip of the magenta cell off as grey (XY x 780-880, y 260-390; XZ x 810-860, y 545-640), so blue (591) excludes the right-hand bright magenta part which C4 includes (773). WORSE (cuts off part of the cell, diagonal cut).
- v20: C4 and C5 identical (342 um^3, contact 84.8). Large bright T cell (strongly green) overlapping the tumour mass; a ring-shaped whitish object in the centre of the T cell footprint (XY ~ (265,285)) outlined grey; XZ shows white (all-channel) areas. Ambiguous whether this is a dead/dying cell (dye) - no separate outline for it beyond the grey ring. Blue = small piece of right cell. SAME. The panel crops the cluster at the bottom edge in XY.

(a) Removed by C5 that C4 had: no whole cell outline was lost compared with C4. C5 re-labels parts of cells as grey (v01, v06, v08, v19) which loses target area. No yellow dead cell visible. Non-cell removal: none relevant.
(b) Merges still present in C5: v16, v17 (lower-left tumour cell piece inside blue), v13 small. In C4 this merge appears v09, v12, v13, v16, v17 - C5 fixes v09, v12 and partly v13.

Verdict v01_MIP_ROI_32
1. Tracking correct? T cell YES 20/20. Target: blue is the central/right tumour cell the T cell is on, consistent id, but 5/20 visits in C4 merge into the lower-left cell and 2/20 in C5; and 4 C5 visits lose part of the target body. Count target OK ~15/20 (C5), ~13/20 (C4).
2. C5 vs C4: mixed. C5 breaks merges at v02, v09, v12, v13 (BETTER x4) but makes straight, jagged cuts across the cell body at v01, v06, v08, v19 (WORSE x4), adds over-splitting grey pieces elsewhere. Net: roughly even, not an improvement for this crowded cluster.
3. Worst frames: v06, v08 (C5), v19 (C5 cut-off strip), v03 (C5 diagonal cut and fragments in T cell), v16/v17 (merge remains).

---------------------------------------------------------------
## v02_MIP_ROI_3  (single round tumour cell + single T cell at the right panel edge; tumour cell later dies and fills with yellow SYTOX)
---------------------------------------------------------------
General: the scene has one tumour cell and one T cell; both partly cut by the right edge of the panel in the early visits. C4 and C5 are IDENTICAL (same outlines, same volume text) for v01-v15 (v02-v15 shown identical to the pixel as far as I can see). The difference is at the end, where C5 LOSES the dead cell.

Per visit
- v01: T cell OK (green cell on left of the cell pair); blue = the magenta cell to its right/above, cut by the right panel edge; one cell. C4=C5 (SAME). XZ: blue overlaps the T cell pixels (the T cell envelops the tumour cell; outlines overlap).
- v02: blue = one magenta cell with a lobe at top-left (bulge), partly out of frame at right; orange T cell at right edge is mostly out of panel. SAME.
- v03: T cell (green, large, clear) OK; blue = bright magenta cell at right edge touching it. SAME.
- v04: blue = one round magenta cell (lower right), T cell at right edge touching. SAME; outline follows the cell well.
- v05: blue = one round cell, T cell beside it (no contact 0.0). SAME, good. NOTE: the position of the cell jumps between v04 (lower, x~420,y~495) and v05 (upper, x~360,y~290); I cannot tell whether this is stage drift or a different cell - ambiguous, but it is the only tumour cell in the panel.
- v06: round cell, outline follows edge (slight notch at the bottom). SAME, good.
- v07: cell is dim (low magenta); outline still follows the edge with small tail at bottom (XY) / ragged lower-left (XZ). SAME. (Dimness ok for one cell.)
- v08: good outline, small bump at bottom. SAME.
- v09: good outline, tiny bleb at lower right included. SAME.
- v10: good outline, bleb at right included. SAME.
- v11: good. SAME. (XY outline has a notch at the lower right.)
- v12: good outline; yellowish spots appear inside the cell (SYTOX nuclear fragments beginning); C4 has a small grey object at top-right of XZ (near the T cell, magenta fragment), C5 has none (it is gone from the outline set but still visible as magenta in raw). Trivial. SAME.
- v13: two bright YELLOW blobs inside the cell (dye uptake), cell still whole; outline contains both. SAME.
- v14: two yellow blobs stacked vertically, cell intact; outline correct incl. small bleb. SAME.
- v15: two yellow blobs, cell intact, outline correct. SAME.
- v16: SERIOUS DIFFERENCE. The cell has collapsed to two bright YELLOW blobs (dead cell, little magenta left). C4: blue outline around the dead cell, 180 um^3 (also a small grey object lower right). C5: NO BLUE OUTLINE; text shows "Target tr=0, Target vol 0.0 um^3". Only a small grey sliver remains in C5 at the lower right of the yellow blob (XY x 700-740, y 495-560). C5 WORSE: wrongly REMOVED the dead (yellow-filled) target.
- v17: dead yellow cell. C4: blue outline around it (183 um^3). C5: no blue outline, tr=0, 0 um^3; only a grey sliver at left. WORSE (target lost).
- v18: two yellow-green blobs (dead cell fragments / apoptotic bodies). C4: blue outline around both, 112 um^3. C5: no outline of any kind, tr=0. WORSE (target lost; both blobs unlabelled).
- v19: dead yellow blob has moved to the top of the XY panel (x~430,y~60) and is touching the T cell in XZ. C4: blue (tr=3) outline around it (114 um^3) - the id has changed from tr=1 to tr=3 in C4. C5: no outline, tr=0. WORSE.
- v20: both C4 and C5 identical, blue tr=3, 187 um^3: outline around a small magenta/green fragment cluster at the upper right of the panel (in contact with the T cell in XZ). It looks like the remains of the dead cell (magenta fragments with a green spot), not obviously a new cell. Plausible. SAME. In C5 the target is recovered at v20 but after a 4-visit gap (v16-v19).

(a) Removed by C5 that C4 kept: the dead, yellow-filled target at v16, v17, v18, v19 - the key problem. Non-cell objects: small grey objects at v12 (XZ, top right) and v16/v17 (grey sliver by the dead cell) are non-cell/ambiguous; the removal of the v12 object is harmless.
(b) Merges in C5: none (one tumour cell). The outline for the live round cell is very good in both (smooth, follows the edge).

Verdict v02_MIP_ROI_3
1. Tracking correct? C4: 20/20 visits (T cell OK, blue always the tumour cell/its dead remains; id changes tr=1 -> tr=3 at v19, ok). C5: 16/20 (target lost v16-v19, T cell fine 20/20).
2. C5 vs C4: identical for 15 visits, WORSE for 4 (v16-v19, dead cell has no outline), SAME at v20.
3. Worst frames: v16, v17, v18, v19 (C5 loses the dead yellow target; tr=0).

---------------------------------------------------------------
## v02_MIP_ROI_32  (one large green T cell + one magenta tumour cell that becomes filled with yellow/green dye spots and dies; clean 2-object scene)
---------------------------------------------------------------
General: C4 and C5 are IDENTICAL in 18/20 visits (v01-v14, v17-v19, v20 outline). They differ at v15 and v16 only (C5 cuts the T-cell overlap off the target). The tracking is correct throughout. In late visits (v13-v20) the tumour cell contains bright yellow-green SYTOX spots (dying/dead); the whole-cell outline is kept in both C4 and C5 here.

Per visit
- v01: T cell OK (large green). Blue = the magenta cell sitting under the T cell's lower part; one cell. In XZ the blue also contains some green T-cell pixels at the top (outlines overlap). SAME.
- v02: blue = magenta cell below T cell, notch at the bottom (XY), one cell. SAME.
- v03: blue = one round magenta cell; the T cell overlaps its upper-left edge. SAME, good.
- v04: SAME, good (one round cell, T cell beside it).
- v05: SAME, good; blue extends a small tail at lower left (XY). One cell.
- v06: blue = round cell with a small bleb at right (XY ~ (310,365)) and a tail at the bottom; the bleb is a separate small magenta round bud and is included in blue in both. SAME. T cell OK.
- v07: blue = cell below/right of T cell with a thin spiky extension up to the T cell (XY x 230-280, y 270-330) that contains green (T-cell) pixels. Included in both. SAME (slightly over-extended in both).
- v08: blue = round cell on the left, T cell on the right; a thin little loop of blue sticks into the T cell (XY ~ (305,300)). SAME.
- v09: T cell on top, tumour cell below; blue extends a tail at the bottom-left. In XZ T cell and blue overlap heavily (the cell is behind the T cell in z). SAME.
- v10: very clean: T cell left, round tumour cell right, outline tight to the cell, small tail lower left. SAME, good.
- v11: blue = round tumour cell + an extension up/left (XY x 245-360, y 270-340) that contains a green-grey region (the object shown as grey-outlined mesh at top, likely T-cell fragments/extension, outlined grey in both). The blue "ears" at top-right of the extension (x 320-365, y 270-330) include a small green/magenta patch. SAME in C4/C5; extension is questionable (not clearly tumour), ambiguous.
- v12: good. Blue includes a spike at top-left (XY (265,280)) that is a bleb. SAME.
- v13: good; blue includes a patch at the top left (x 225-270, y 270-295, between the T cell and the cell, green-grey) = part of T cell contact zone. SAME. Green dye spots appear inside the cell.
- v14: blue = tumour cell plus a second lobe to the left which overlaps the T cell outline (XY x 140-210, y 205-260) and a small lobe to the right (x 330-355, y 235-260). Both in C4 and C5. The left lobe is inside the green T cell region - probably T-cell pixels or a bleb. SAME; shape is irregular but the cell is whole.
- v15: DIFFERENCE. C4: blue includes an extension that runs into the T cell (XY x 185-230, y 175-230; XZ x 185-250, y 585-660) and a separate grey small round object at right (x 345-385, y 260-300 in XY; x 345-385 in XZ), contact 36.5 um^2, 977 um^3. C5: that T-cell overlap region is now a grey outline (not part of the target), blue 829 um^3, contact 0.1; the small grey object at right is gone (not outlined). C5 BETTER: the blue cell is the round tumour cell alone, boundary follows the visible gap in XY. Removal of the small dim round object at right (a faint green-grey bleb/debris) looks correct.
- v16: C4 blue 1070 um^3: includes the T-cell overlap region and a long spiky tail running down in XZ (x 195-245, y 700-760), contact 76 um^2. C5 750 um^3: the overlap band (XY x 630-700, y 200-300; XZ x 630-700, y 545-700) is cut off as grey, tail is mostly grey. In XZ C5 shows an almost vertical straight line at the cut (x~693, y 570-600) and thin slivers; in XY the cut also removes the magenta crescent on the cell's left side that C4 contained (x 670-690, y 215-245) - so some real cell is lost. Net: C5 BETTER for removing the T-cell overlap/tail, but the cut is jagged and not exactly on the cell boundary (mixed). I score BETTER.
- v17: SAME. Cell with green-yellow spots, T cell on the left, contact 50 um^2 identical. Blue has a small tail at the lower end in both.
- v18: SAME. Yellow spots in cell, T cell left; blue includes lower-right tail (x 300-330, y 250-310). SAME.
- v19: SAME. Tumour cell lower-left with yellow dye (dead), T cell upper right; blue includes a sliver on top (XY x 240-300, y 190-240) overlapping the T cell. SAME.
- v20: Both show blue around the yellow-filled cell at the right panel edge (cut by the panel edge) and the T cell above it. The blue outline of the dead cell is retained in C5 (not removed). Same outline; ID label is tr=2 in C4 but tr=1 in C5 (C5 id is consistent with the earlier tr=1, C4's id switched). C5 BETTER on identity only; outline SAME.

(a) Wrongly removed in C5: none. The dead yellow-filled cell keeps its outline at v18-v20. Non-cell removal: the small grey object (faint bleb) at v15 removed, fine.
(b) Merges in C5: none of two tumour cells. In both C4 and C5 the blue outline repeatedly includes T-cell/contact-zone green material at the cell edge (v07, v11, v13, v14, v19), i.e. blue is slightly too large where the T cell touches. C5 fixes this only at v15-v16.

Verdict v02_MIP_ROI_32
1. Tracking correct? YES 20/20 visits for the T cell (orange); blue is the one tumour cell the T cell is touching in 20/20 (id tr=1 in C5 throughout; C4 switches to tr=2 at v20).
2. C5 vs C4: identical for 18 visits; BETTER at v15 and v16 (T-cell overlap and tail removed, one cell); v20 better only in track id.
3. Worst frames (both methods): v14 (blue has extra left lobe inside T cell), v11 (extension), v16 C4 (tail + overlap), v16 C5 (straight/jagged cut, loses magenta crescent).

---------------------------------------------------------------
## v03_MIP_ROI_6  (large green T cell on a multi-body magenta tumour mass with small round magenta bodies nearby; target dies late, yellow)
---------------------------------------------------------------
General: C5 target volume is only 30-45% of C4 for v01-v12 (e.g. 1064->371, 980->321, 922->288) because the main bright magenta cell is relabelled grey and the tracked blue shrinks to a thin diagonal ribbon or the piece lying under/against the T cell. From v13 to v20 C4 and C5 are identical (same volumes). Small round magenta bodies (blebs/apoptotic bodies or tiny cells, ~1/5 the diameter of the main cell) are outlined grey in C4 and not outlined at all in C5.

Per visit
- v01: In XY the orange T-cell outline covers mostly the MAGENTA body on the left; the green crescent is on the right side of it, so the T cell outline is arguably too big (both C4 and C5; ambiguous because they overlap in projection). C4 blue = magenta body beneath/around the T cell (1064). C5 blue (371) = a thin curved band running diagonally across the middle (XY (640,265)->(745,205)), with a large grey outline around the left-hand magenta body. WORSE: the cell is cut by a curved line through its body, and target is a sliver.
- v02: T cell OK (green). C4 blue 980 = the big bright magenta cell (right) plus the small lobe at lower left, touched by the T cell; small round bodies at top and left are grey. C5 blue 321 = only the small lower-left lobe and a thin strip; the whole big bright magenta cell (XY x 780-885, y 255-385; XZ right) is grey (separate object). WORSE: the cell the T cell touches is not the target; one cell cut in two by a curved line. The small round objects at top/left (round faint rings ~35 px) have no outline in C5: looks like correct removal of debris/vesicles (ambiguous).
- v03: C4 blue 1127 = big magenta mass + right bud (ring body at (355,300) included). C5 510: blue = upper right part with a squiggly boundary inside the T cell (XY (720,285-300) and XZ (715,680-690) show small zigzag blue loops), big grey diagonal polygon at lower right, T cell outline filled with grey fragments. WORSE (fragmentation, jagged boundary).
- v04: C4 994 = whole magenta mass (incl. T-cell overlap, with small blue holes in T cell). C5 331 = lobe under the T cell and thin strips; bright upper-right magenta region (XY x 790-885, y 200-310) is grey. WORSE. Two small round bodies top-left are grey in C4, no outline in C5.
- v05: C4 781 vs C5 319. Same pattern: bright right-hand cell (XY x 760-880, y 190-320) is grey in C5, blue = thin strip along T cell edge and lower-left lobes. WORSE. Small rings at top-left removed in C5 (ambiguous: could be small cells).
- v06: C4 948 is a MERGE: blue joins the upper-left twin round bodies (XY (240,160)), the bright cell, and lower bodies in one outline. C5 479: upper twin bodies and the bright cell are grey, blue is a ribbon (XY (730,225)->(850,250)) crossing the bright cell plus the mass lying under the T cell. C5 separates the upper cells (good) but the bright cell the T cell is on is still not the target and the ribbon is a straight cut. WORSE (mixed).
- v07: C4 = C5 in blue (731): blue = the main magenta mass (3 lobes) under the T cell; contact 45 um^2. C5 drops the two small grey objects inside the T cell and the grey ring cell at left (round magenta cell with green centre, (180,220); a small cell, ambiguous). SAME / marginally BETTER (cleaner). Target good here in both.
- v08: C4 763 = main bright cell + lower round lobe with green spot. C5 372: bright cell (XY x 760-890, y 190-290) grey; blue is a diagonal ribbon through the T-cell overlap plus the lower bud. WORSE. Small left ring cell with green centre removed in C5 (ambiguous). A small bright green blob (XY (660,275)) is unoutlined in both - fine (T-cell fragment).
- v09: C4 922: one blue outline over the bright cell, a bridge, and the lower round cell with a green dot (merge of 3 things). C5 288: the bright cell is grey; blue = bridge region + lower round cell + overlap inside T cell. C5 separates the lower round cell from the bright cell (merge broken) but the bright cell the T cell touches is not the target; XZ blue is a tiny region in the T cell. WORSE (mixed).
- v10: C4 852: blue = bright cell + upper arm + lower round cell (merge). C5 390: blue = lower round cell + bridge + piece in T cell; bright cell and arms grey, with a horizontal straight cut line (XY (755-860,272)). WORSE (mixed).
- v11: C4 816: blue = bright cell + T-cell overlap; C5 336: bright cell grey, with straight diagonal blue edge (XY (770,250)->(850,285)); blue loops inside the T cell. The small lower-left round cell with a bright yellow-green spot ((215,420) in C4, grey) has no outline in C5; it is a small separate object, not the target. WORSE.
- v12: C4 778: blue = the large oval magenta cell (upper) and its extension in the T-cell overlap. C5 323: the oval cell (XY x 655-780, y 215-310) is grey, blue is a curved band across the lower half only. WORSE (a single oval cell is cut in two by a curved line, target gets the lesser part). C4 had a grey circle around the yellow-green bright spot inside the T cell; C5 has none.
- v13: C4 = C5 (754). Blue = magenta cell with a bright yellow-green spot (left), T cell on the right with a blue extension inside it (XY x 330-380, y 300-410). SAME. C5 removes a small grey magenta blob at the right (XY (455,270)) = debris, fine.
- v14: C4 = C5 (718). Blue = the cell cluster of ~3-4 round magenta bodies, each with a green dot - one outline for what may be several small cell bodies (possible merge, ambiguous). A bright YELLOW blob at the left (XY (255,370)) is outside the blue in both (a dead/dye-filled object with no outline in C5 and only a partial grey bit in C4). SAME.
- v15: C4 = C5 (578). Blue = rounded cell, T cell right; yellow blob at the left edge of blue (XY (265,375)), not outlined in either. SAME.
- v16: C4 = C5 (742). Blue over the cell with a big yellow spot at its left edge and another yellow blob at the top (XY (340,240)) outside the blue in both, unoutlined (dead fragments, ambiguous). SAME.
- v17: C4 = C5 (670). Cell with yellow blob inside at left; blue has a thin tail down across the T cell (XY (310-340,395-445)). SAME.
- v18: C4 = C5 (692). Target is now a dim, ghost-like cell (pale) above the T cell; the yellow blob is lying INSIDE the orange T-cell outline (XY (275,360)), i.e. a dye-filled object not outlined as target. SAME. Ambiguous whether that yellow blob is the dead tumour cell proper - the blue cell above is dim/white.
- v19: C4 = C5 (429). Blue around a yellow-filled dead cell (XY top, two bright yellow lobes), good outline. SAME. C5 drops the small grey object at left.
- v20: C4 = C5 (274). Blue outline around the dead yellow cell, T cell below, contact 0. SAME. Good in both.

(a) Wrongly removed by C5 vs C4: no dead target outline lost. But C5 relabels the main bright magenta cell as grey (not target) in v01-v12 (every one of those visits): functionally a loss of the real target. Small round objects (v02-v08, ring-like magenta bodies, some with a green centre) and small yellow/grey bits (v11, v12, v14-v19) lose their grey outline in C5 - probably correct (debris/blebs) but the ring-with-green-centre bodies at v07/v08 could be small cells (ambiguous). A yellow-filled blob at v14-v16 is unoutlined in C5 (and mostly in C4).
(b) Merges still present in C5: no multi-cell merge within blue in v01-v12 (blue too small instead); the cluster of 3-4 round bodies in v14-v17 remains one outline, as in C4. C4 merges: v06, v09, v10 (bright cell + lower round cell + upper bodies).

Verdict v03_MIP_ROI_6
1. Tracking correct? T cell: YES 19-20/20 (v01 XY outline ambiguous). Target: C4 mostly the right cell but merged with neighbours in v03-v06, v09, v10 (about 14/20 clean); C5 only 9/20 (v07, v13-v20) - for v01-v06 and v08-v12 the blue is a minor/diagonal piece and the big bright cell the T cell attacks is grey.
2. C5 vs C4: WORSE in 11 visits (v01-v06, v08-v12; two mixed - v06, v09, v10 break merges but lose the real target), SAME in 9 (v07, v13-v20 identical), BETTER 0. Straight/curved cuts through the middle of a single cell body are the main failure.
3. Worst frames: v02, v05, v08, v11, v12 (large bright cell grey, blue a sliver, volume 30-40% of C4), v01 (diagonal band).

===============================================================
## OVERALL SUMMARY (C5 judged against C4 and the raw colours; 20 visits per chamber, all opened)
===============================================================

| Chamber | Tracking OK visits (C5) / 20 | C5 BETTER / SAME / WORSE (visits) | Serious problems |
|---|---|---|---|
| v01_MIP_ROI_2  | 20/20 (C4 20/20) | 5 / 15 / 0 (better at v05, v07, v08, v12, v15; 15 identical) | None. Both methods leave a large soft blue outline around a dim target in v03-v07. |
| v01_MIP_ROI_32 | T cell 20/20; target about 15/20 (C4 about 13/20) | 4 / 11 / 5 (better v02, v09, v12, v13; worse v01, v03, v06, v08, v19) | Crowded cluster of 2-3 tumour cells: C5 target volume about 25-35% lower than C4 throughout; straight/diagonal cuts across the cell body (v06, v08, v19); lower-left cell still merged into blue at v16-v17; many extra grey fragments. |
| v02_MIP_ROI_3  | 16/20 (C4 20/20) | 0 / 16 / 4 (worse v16-v19) | C5 REMOVES the dead yellow-filled target at v16, v17, v18, v19 (tr=0, 0.0 um^3, no outline); C4 keeps it (180, 183, 112, 114 um^3). Target reappears (tr=3) only at v20. |
| v02_MIP_ROI_32 | 20/20 (C4 20/20; C4 id flips to tr=2 at v20) | 2 / 18 / 0 (better v15, v16) | None serious. v16 C5 cut is straight/jagged and trims part of the magenta crescent; both methods include T-cell contact material in blue (v07, v11, v13, v14, v19). Dead yellow cell keeps its outline at v18-v20. |
| v03_MIP_ROI_6  | T cell 19-20/20; target 9/20 (C4 about 14/20) | 0 / 9 / 11 (worse v01-v06, v08-v12; v13-v20 identical) | In v01-v12 C5 turns the large bright magenta cell (the one the T cell attacks) into a grey non-target and the blue becomes a thin diagonal/curved ribbon (target volume 30-45% of C4); straight/curved cuts through a single cell body; C4 merges (v06, v09, v10) are broken but at the cost of the real target. |

Totals over 100 visits: C5 better 11, same 69, worse 20.
Overall picture: C5 is identical to C4 where there is one clean round tumour cell. It helps in a few frames by dropping fragment clutter (v01_MIP_ROI_2 v05/v08/v12), T-cell overlap (v02_MIP_ROI_32 v15/v16) and by separating a neighbouring tumour cell (v01_MIP_ROI_32 v02/v09/v12). It hurts where the target is large, bright or crowded: it cuts one tumour cell along a straight or curved line and tracks the lesser piece (v03_MIP_ROI_6 v01-v12, v01_MIP_ROI_32 v01/v06/v08/v19), and it loses a dead yellow-filled target entirely (v02_MIP_ROI_3 v16-v19).
Caveat: several judgements are for crowded overlapping scenes (v01_MIP_ROI_32, v03_MIP_ROI_6) where MIP overlap of T cell and tumour makes "which cell is which" ambiguous; the per-visit notes flag those. No yellow SYTOX signal was seen in v01_MIP_ROI_2 or v01_MIP_ROI_32 (except the ring/white region at v01_MIP_ROI_32 v20, ambiguous).
