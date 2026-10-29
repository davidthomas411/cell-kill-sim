# P6 frame review D: v09_MIP_ROI_38, v10_MIP_ROI_28, v12_MIP_ROI_32

Reviewer: independent frame reviewer (Claude). Every frame of the three chambers was opened and inspected (22 + 21 + 22 = 65 frames). Left = C4, right = C6.
Verdicts judge the blue TARGET outline of C6 against C4. "Same" means identical outline and identical volume on the panel.

General observations
- No yellow SYTOX-positive dead cell appears in any frame of these three chambers, so failure (c) (dead dye-filled target losing its outline) was not testable here.
- Where C6 and C4 give the same volume to 0.1 um^3, the outlines are identical. Most "same" frames are exactly that.
- Target identity is the main issue. In v09 and v10 C4 and C6 follow different tumour cells for long stretches.

## v09_MIP_ROI_38 (22 visits)

Context: three tumour cells and one big T cell. C4 follows the top cell (ragged, holes, strips) for most of the series and flips targets. C6 follows the right/lower cell that eventually docks onto the T cell. C6 volumes are smooth (616-889) while C4 swings 148-813.

- 01: BETTER. C4 blue is a fragmented top cell (208 um^3, strip plus grey remainder, case a). C6 blue is the whole right cell (646). Different cell, but a clean one.
- 02: BETTER. C4 558 includes a blue lobe on a cell that overlaps the T cell. C6 616 is clean on the right cell. Target identity differs from C4 (C6 cell is not yet touching the T cell, contact 0).
- 03: BETTER. C6 627 clean; C4 603 touches the T cell with an irregular XZ outline.
- 04: SAME. Both clean and ~660 um^3, but different cells (C4 top, C6 right).
- 05: BETTER. C4 328 has a hole and a blue/grey cut in the top cell. C6 679 full clean cell.
- 06: BETTER. C4 201 ragged and cut in two. C6 708 clean.
- 07: BETTER. C4 229 is a ragged strip on the top cell and the contacting cell is grey. C6 738 outlines the contacting cell. T cell contact 12.5 um^2 is captured.
- 08: SAME. Identical (813, contact 28.8).
- 09: BETTER. C4 299 is the non-contacting upper cell and the contacting cell is grey. C6 708 outlines the contacting cell with a thin grey strip at the interface. T cell is cut into orange plus grey pieces in both.
- 10: BETTER. C4 380 is the upper cell with holes. C6 608 is the lower cell in contact (120 um^2). Orange T cell has a ragged protrusion in both.
- 11: BETTER. C4 308 (upper cell, holes). C6 662 contacting cell, clean.
- 12: BETTER. C4 352 (upper cell, ragged). C6 889 lower cell below the T cell, outline sensible in XZ. Blue overlaps the T footprint in XY only because the cell sits under the T cell.
- 13: BETTER. C4 652 upper cell; C6 593 contacting cell. Both clean, C6 target is the continuous track.
- 14: BETTER. C4 287 (upper cell, internal blue lines). C6 668 lower cell, clean.
- 15: BETTER. C4 213 is a fragmented ring on the far cell. C6 804 whole left cell.
- 16: BETTER. C4 149 fragmented. C6 829 whole cell.
- 17: BETTER. C4 397 with an internal hole. C6 675 whole cell.
- 18: BETTER. C4 318 with trailing blue fragments in XZ. C6 715 clean.
- 19: SAME. Identical (756).
- 20: MIXED / slightly worse. C6 613 vs C4 918. In C6 the left magenta cell has an extra grey sliver cut out of it and the volume dropped 756 to 613. C4 keeps one piece. A blue loop inside the green T cell appears in both.
- 21: SAME. Identical (369). Volume drop from v20 is in both, shape is ragged magenta cluster, unclear.
- 22: SAME. Identical (708). Blue is now a round cell that is not the v21 outline in either method. Both methods have the same target jump 369 to 708.

T cell (orange): correct and the same in both through v01-v18. In v19-v22 the orange outline is big and merges a bright green debris blob and a tail, in both methods.

Tally v09: better 16, same 5, mixed 1, worse 0.

## v10_MIP_ROI_28 (21 visits)

Context: three tumour cells touch one T cell (upper-left cell A, lower cell B, then a faint third cell C). C4 follows A (big cell, 561-981 um^3) from v01 to about v18. C6 matches C4 on v04-v11 but is unstable on v01-v03 and abandons cell A from v12 to v18 for a smaller cell, giving a target volume collapse (897 to 187).

- 01: WORSE. Case (a)/(b) failure. C6 blue merges the lower cell with the lower half of the upper cell and leaves a grey sliver of the upper cell (checked on an enlarged crop). C4 separates the two cells properly (blue top cell, grey lower cell).
- 02: WORSE. C6 blue jumps to the lower cell B (458) while C4 stays on A (982). Volume 650 to 458 and target identity flipped.
- 03: WORSE. C6 blue is still lower cell B, now only 220 um^3 with an odd XZ outline inside the T cell footprint. C4 861 clean on A.
- 04: SAME. Both 872 on A. C6 drops the grey fragments on the dim lower-right cell so the cell has no outline (see note).
- 05: SAME. Identical (814).
- 06: SAME. Identical (794). C6 has no grey pieces inside the T cell; C4 has tiny grey fragments. Slightly cleaner, target unchanged.
- 07: SAME. Identical (840). Dim cell C has no outline in C6 (C4 has a grey outline).
- 08: SAME. Identical (873). C6 cleaner (no grey blob in the T cell).
- 09: SAME. Identical (802).
- 10: SAME. Identical (938).
- 11: SAME. Identical (897).
- 12: WORSE. C6 target switches to the small dim cell at lower right, 187 um^3, ragged hooked outline. Original cell A (725 in C4) becomes grey. Case (e)/(f).
- 13: WORSE. C6 321 on the lower cell; A is grey. C4 715 on A.
- 14: WORSE. C6 462 on a lower-left cell, A grey. C4 585 on A.
- 15: WORSE. C6 402, lower-left cell with a grey piece inside the T footprint; big round cell A (913 in C4) is grey. 
- 16: WORSE. C6 407; A (849 in C4) grey.
- 17: WORSE. C6 369; A (800 in C4) grey.
- 18: WORSE. C6 422; A (709 in C4) grey. In both methods the lower-left and right cells have ragged outlines.
- 19: MIXED. C4 follows A as it fades (blue on a dim upper-right ghost, 113). C6 outlines the large left cell well (618) but it is the cell C6 switched to at v12. C6 volume is plausible, C4 volume collapse is a cell fading.
- 20: SAME. Identical (525), left cell. In C4 this is a new id (tr=2), so C4 also switches here.
- 21: SAME (both doubtful). C4 target is a small lower-right piece (154). C6 target is the upper cell (321) with a blue loop reaching into the T cell in XZ. Different cells, neither convincing.

T cell (orange): correct, same in both. C4 v01-v03 orange is also fine here (single T cell).

Notes: cell C (dim faint cell at the right of the T cell, v04-v11) loses its grey outline in C6 on v04, v07-v09 (C4 had a fragmented grey outline). It is very faint, so this is a minor deletion (case d), but it is the cell C6 later picks as target at v12.

Tally v10: better 0, same 10, mixed 1, worse 10.

## v12_MIP_ROI_32 (22 visits)

Context: one big tumour cell with a single T cell. The blue target is identical in C4 and C6 on every visit (same volumes in every frame). The only differences are the orange T cell and grey objects.

- 01: BETTER. Target identical (709.5). C4 orange is a small round dim object at top right, the real large green T cell is grey. C6 orange is the large T cell. (case e fix)
- 02: BETTER. Target identical (692). C4 orange still on the small round dim cell. C6 orange on the large T cell, with a small neighbouring green blob attached by a neck (minor).
- 03: BETTER. Identical target (673). C4 orange on a small upper-left object and the big T cell is grey. C6 orange on the big T cell. Contact is correctly 23.8 um^2 in C6 (0 in C4).
- 04: BETTER. Identical target (691). C4 orange on a small fragment, T cell grey. C6 orange on the big T cell (contact 29.9). A grey sliver cuts the T/tumour interface in both.
- 05: SAME. Identical (685), T cell orange in both.
- 06: SAME. Identical (675).
- 07: SAME. Identical (750). Orange T includes a jagged strip through the tumour in both.
- 08: SAME. Identical (627).
- 09: SAME. Identical (644). Grey piece inside the T cell in both.
- 10: SAME. Identical (647).
- 11: SAME. Identical (646).
- 12: SAME. Identical (629).
- 13: SAME. Identical (634).
- 14: SAME. Identical (630).
- 15: SAME. Identical (631). Orange merges a small separate green cell into the T outline in both.
- 16: SAME. Identical (587).
- 17: SAME. Identical (601).
- 18: SAME. Identical (659).
- 19: SAME. Identical (554). Orange T outline includes a bright separate blob on the left in both.
- 20: SAME. Identical (554). C4 orange T cell is messy and includes tumour-coloured area at the interface; C6 orange T cell is tidier but the T cell is split orange plus a large grey part. Target unchanged.
- 21: SAME. Identical (574). Orange T cell merges a bright blob and tail in both.
- 22: SAME. Identical (484). Same merge.

T cell (orange): C6 is correct on v01-v04 where C4 tracks the wrong object (a small dim round cell then a small fragment). From v05 on both agree. Several late frames (v15, v19, v21, v22) merge a separate bright green cell or blob into the T outline in both methods.

Tally v12: better 4, same 18, mixed 0, worse 0. All "better" frames are T-cell corrections, none change the blue target.

## Overall tally

| Chamber | better | same | mixed | worse |
|---|---|---|---|---|
| v09_MIP_ROI_38 | 16 | 5 | 1 | 0 |
| v10_MIP_ROI_28 | 0 | 10 | 1 | 10 |
| v12_MIP_ROI_32 | 4 | 18 | 0 | 0 |

## Worst frames

1. v10_MIP_ROI_28 / 01: C6 blue merges the two touching tumour cells (lower cell plus lower half of the upper cell) and cuts a grey sliver off the upper cell. C4 separates them correctly.
2. v10_MIP_ROI_28 / 03: C6 target is 220 um^3 on the lower cell (C4 861 on A); XZ outline sits inside the T cell. Target volume collapse.
3. v10_MIP_ROI_28 / 12: C6 switches target to the small dim cell at the lower right (187 um^3, ragged hooked outline), original cell A becomes grey. Volume collapse 897 to 187 between neighbouring visits.
4. v10_MIP_ROI_28 / 13-18: C6 stays on the smaller cell (321-462 um^3) while original cell A (700-913 in C4) is grey. Six frames of the wrong, or at least switched, target.
5. v10_MIP_ROI_28 / 02: target flips from the merged blob to lower cell B (650 to 458) one visit after v01, then 220 on v03, then 872 on v04.
6. v09_MIP_ROI_38 / 20: C6 cuts an extra grey sliver off the left magenta cell, volume 756 to 613 (C4 918). The only non-better frame in v09.

Caveats
- v09 C6 and C4 follow different tumour cells for much of the series; I judged C6 better because its target outline is clean and continuous (volume 593-889) and ends on the cell that docks onto the T cell, whereas C4's target is a ragged or fragmented cell with holes. If the intended target is the top cell, the verdict for v09 v01-v18 would need re-checking against the target-choice decision.
- v10 target choice is ambiguous in v12-v21 (three tumour cells around one T cell). The verdict "worse" rests on C6 abandoning the cell it and C4 both tracked from v04 to v11 and on a volume collapse of ~75 percent in one visit.
