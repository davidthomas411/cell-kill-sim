# P6 frame review B (C4 left vs C6 right)

Chambers: v03_MIP_ROI_6, v03_MIP_ROI_30, v04_MIP_ROI_24, v07_MIP_ROI_2. All 82 frames opened and read
(20 + 20 + 20 + 22). Frames: figures/pilot/p6/frames/<chamber>/<visit>.jpg.

Verdict key (target = blue outline): better / same / mixed / worse = C6 relative to C4.
"ID" notes refer to the "Target tr= | T cell tr=" caption line; C4 sometimes shows tr=2/3, C6 is always tr=1.
The caption volume is a single number for both columns in three chambers because the outlines are identical there;
only ROI_2 shows different volumes (C4 / C6).

## Headline

- ROI_6, ROI_30, ROI_24: the blue (target) and orange (T cell) outlines are pixel-identical between C4 and C6 in every visit.
  C6 changes nothing on the target there. The only differences are (i) ROI_6: C6 drops some small grey outlines on real
  magenta tumour fragments, (ii) ROI_24: C6 keeps constant track IDs where C4 flips IDs.
- ROI_2 is the only chamber where C6 changes the target, and it is mostly worse: straight-line and horizontal cuts through
  touching tumour cells at visits 1-2, one correct separation at visit 3, and a likely wrong-cell target at visits 17-19.
- No dead (yellow) target lost its outline in any of the four chambers (c: none). Yellow-dye targets at ROI_6 v16-20,
  ROI_30 v20, ROI_24 v19-20 are outlined in both columns.
- Target-volume collapses (f) are shared by C4 and C6 in ROI_6/30/24 (not C6-specific); see per-chamber notes.

## v03_MIP_ROI_6 (20 visits)

Target and T cell outlines identical in all 20 visits. Differences are in grey (non-target) objects only.

| Visit | Verdict | Note |
|---|---|---|
| 01 | same | identical; target (1064) includes the whole magenta mass, T cell orange correct |
| 02 | same | C6 deletes the small round magenta cell at left (grey in C4) -> real tumour object lost (d, non-target) |
| 03 | same | C6 deletes the small round cell at top-left (grey in C4) (d, non-target) |
| 04 | same | C6 deletes the top-most small round cell (d, non-target); other grey cell kept |
| 05 | same | C6 deletes the small left cell (d, non-target) |
| 06 | same | identical |
| 07 | same | C6 drops two tiny grey outlines on magenta bits inside the T cell; trivial |
| 08 | same | identical |
| 09 | same | C6 deletes the magenta fragment at top-left (d, non-target). Target blue still carries the round lower cell as a tail (both) |
| 10 | same | identical |
| 11 | same | C6 deletes two grey tumour fragments (top and far left) (d, non-target) |
| 12 | same | identical |
| 13 | same | C6 deletes the small magenta cell at right edge (d, non-target) |
| 14 | same | C6 drops a tiny grey piece next to the yellow blob (yellow blob unoutlined in both; it is outside the target) |
| 15 | same | C6 drops a small grey piece below target |
| 16 | same | C6 drops small grey piece; yellow dead part inside blue in both |
| 17 | same | C6 drops small grey piece; identical target |
| 18 | same | C6 drops small grey piece inside T cell; identical target |
| 19 | same | C6 drops grey debris to left of target; identical target |
| 20 | same | dead yellow target outlined in both; 274 um^3 |

Tally ROI_6 (target): better 0 / same 20 / mixed 0 / worse 0.
Non-target real-cell deletions by C6: 7 visits (02, 03, 04, 05, 09, 11, 13). Each is a small round magenta tumour
object that C4 outlined in grey and C6 leaves without any outline. Minor for the target but a (d) violation.
T cell (orange) correct and identical in both. Shared issue: volume falls 692 -> 429 -> 274 over v18-20 (dye-filled, shrinking
target; plausible, not C6-specific).

## v03_MIP_ROI_30 (20 visits)

Every frame identical between C4 and C6 (blue, orange, no grey objects either side). Target is the magenta cell, T cell is the
big green cell, both correct throughout; dead yellow target at v20 outlined in both.

| Visit | Verdict | Note |
|---|---|---|
| 01-20 | same (all 20) | identical outlines, identical volumes (843, 766, 838, 832, 834, 877, 824, 799, 824, 812, 744, 765, 875, 916, 929, 685, 592, 432, 582, 767) |

Tally ROI_30: better 0 / same 20 / mixed 0 / worse 0.
Shared (not C6-specific) issues: v07 blue wraps two magenta lobes around the T cell (left and lower), not separated in either;
v17-v18 volume drops 685 -> 592 -> 432 then 582 -> 767 because the target is partly hidden under the T cell and its outline is
cut by the T-cell outline (outline of the target stops at the T-cell boundary in XY).

## v04_MIP_ROI_24 (20 visits)

Blue and orange outlines identical in all 20 visits. The only difference is track-ID stability: C4 shows ID flips, C6 does not.

| Visit | Verdict | Note |
|---|---|---|
| 01-05 | same | identical; clean single target and single T cell |
| 06 | same (ID better) | C4 T cell tr=2, C6 tr=1; same outline |
| 07 | same (ID better) | C4 T cell tr=2, C6 tr=1 |
| 08 | same (ID better) | C4 T cell tr=2, C6 tr=1 |
| 09-16 | same | identical |
| 17 | same (ID better) | C4 target tr=2, C6 tr=1 |
| 18 | same (ID better) | C4 T cell tr=3, C6 tr=1 |
| 19 | same (ID better) | C4 T cell tr=3, C6 tr=1 |
| 20 | same | identical |

Tally ROI_24 (outline): better 0 / same 20 / mixed 0 / worse 0. ID continuity: C6 better at 6 visits (06, 07, 08, 17, 18, 19);
the one T cell and one tumour cell are the same cells throughout, so C6's constant IDs look right.
Shared issues: target volume 1396 (v09) -> 985 (v10) -> 653 (v11) -> 580 (v12) -> 1086 (v13): in v10-v12 the blue outline covers only
the dim, partly overlapped part of the tumour cell next to the T cell (XY), so the dip is a segmentation artefact in both,
not C6. v10-v11 and v17-v18 also show blue wrapping into the T cell region.

## v07_MIP_ROI_2 (22 visits)  <- the only chamber where C6 differs on the target

Volumes shown as C4 / C6 when different.

| Visit | Verdict | Note |
|---|---|---|
| 01 | worse | C4 merges the two touching tumour cells into one blue (1261). C6 cuts with a dead-straight vertical line through the middle of the right cell (927): blue = left cell + half of right cell, grey = right-hand sliver. Single cell wrongly cut (a); the true cell boundary is further left. Volume is closer to a single cell but the cut is in the wrong place |
| 02 | worse | C4 merged blue (1332). C6 slices horizontally: a long grey strip crosses both cells and the lower parts of both cells go grey, blue is the upper part (955). Cell wrongly cut into blue strip plus grey remainder (a) |
| 03 | better | C4 merged blue (1309). C6 separates the two touching cells correctly: blue = right cell (690), left cell grey (b). Boundary is slightly bitten into the left cell but acceptable |
| 04 | same | identical messy blue with grey interior fragments (647); target is a single ring-like cell cut into pieces in both |
| 05 | same | identical thin, ragged blue (554) |
| 06 | same | identical (580) |
| 07 | same | identical (476) |
| 08 | same | identical (501); clean target / T cell split in both |
| 09 | same | identical (595) |
| 10 | same | identical (898), volume jump from 595 is in both |
| 11 | same | identical (966) |
| 12 | same (ID better) | identical outline, volume collapse 966 -> 305 in both; C4 target tr=2, C6 tr=1 |
| 13 | same | identical (948) |
| 14 | same | identical (1215) |
| 15 | mixed | targets differ. C4 (923) picks the lower-left tumour cell, C6 (768) picks the upper cell and leaves the lower-left cell grey. Contact area is 77 um^2 (C4) vs 235 (C6) vs 177 at v14 and 238 at v16, so C6 is plausibly the better continuity, but I cannot confirm from the image which cell is the v14 target; flag for manual check (e) |
| 16 | same (ID better) | identical (443); C4 target tr=2, C6 tr=1 |
| 17 | worse | C4 (245) = strip along the lower right edge plus an inner fragment, contact 101. C6 (222) = a strip at the top (away from the T cell) plus a fragment inside the T cell, contact 2.3. Both are poor, but C6 has dropped the cell in contact with the T cell; the remainder of the contacting cell is grey (a, e) |
| 18 | worse | C4 (501) = cell at the right end of the T cell with fragments, contact 432. C6 (611) = the left-hand cell with fragments, contact 0.2: target switched to the non-contacting tumour cell (e) |
| 19 | worse | C4 (538) = clean right-hand cell touching the T cell (contact 110). C6 (800) = the left cell, contact 20; right cell left grey. At v20 both return to the lower cell (271), so C6 jumps cell and back (e, f); C4 is continuous 18 -> 19 -> 20 |
| 20 | same | identical (271) |
| 21 | same | identical (501); in both, blue merges the large left cell with a lower fragment under the T cell (two objects, one outline) |
| 22 | same (ID better) | tumour gone, target tr=0 in both; C4 T cell tr=4, C6 tr=1 |

Tally ROI_2 (22): better 1 (v03) / same 15 / mixed 1 (v15) / worse 5 (v01, v02, v17, v18, v19).
ID continuity better in C6 at v12, v16, v22 (outline unchanged there). T cell (orange) correct in both everywhere.
C6-specific volume instability: 927, 955, 690 at v01-03 (vs C4 1261, 1332, 1309) and 222, 611, 800 at v17-19 (vs C4 245, 501, 538).
Shared failures (not C6-specific): v04-v06 ragged, fragmented blue on a faint cell; v12 volume collapse 966 -> 305.

## Overall tally

| Chamber | Visits | better | same | mixed | worse |
|---|---|---|---|---|---|
| v03_MIP_ROI_6 | 20 | 0 | 20 (7 of them with minor non-target deletions) | 0 | 0 |
| v03_MIP_ROI_30 | 20 | 0 | 20 | 0 | 0 |
| v04_MIP_ROI_24 | 20 | 0 (6 with better ID continuity only) | 20 | 0 | 0 |
| v07_MIP_ROI_2 | 22 | 1 | 15 | 1 | 5 |
| Total | 82 | 1 | 75 | 1 | 5 |

## Worst frames

1. v07_MIP_ROI_2 / 01: C6 cuts the right tumour cell with a straight vertical plane; blue gets half a cell, grey the other half.
2. v07_MIP_ROI_2 / 02: C6 slices both touching cells horizontally; a grey strip runs through the middle; blue is only the upper part.
3. v07_MIP_ROI_2 / 19: C6 targets the left, non-contacting tumour cell (800) while C4 follows the contacting right cell (538); C6 then jumps back at v20.
4. v07_MIP_ROI_2 / 18: same wrong-cell target (611, contact 0.2 vs 432).
5. v07_MIP_ROI_2 / 17: C6 target is a top strip plus an inner fragment, contact 2.3 um^2; the contacting cell is grey.
6. (Minor, non-target) v03_MIP_ROI_6 / 02, 03, 04, 05, 09, 11, 13: real small round magenta tumour cells that C4 outlined are unoutlined in C6.
7. (Uncertain) v07_MIP_ROI_2 / 15: target differs between C4 and C6; needs a manual check of which cell is the target.

## Bottom line

On these four chambers C6 is not an improvement. Where outlines are already clean (ROI_30, ROI_24, ROI_6) C6 is identical to C4
on the target, and its only effects are better track-ID continuity (ROI_24, three frames in ROI_2) and a small number of
deleted non-target tumour cells (ROI_6). Where there are two touching tumour cells (ROI_2 v01-03), the splitter produces one correct
separation (v03) and two bad cuts (v01, v02). The whole-series tracker appears to pick the wrong tumour cell at ROI_2 v17-19.
