# P6 frame review A: C4 (left) vs C6 (right)

Chambers: v01_MIP_ROI_2, v01_MIP_ROI_32, v02_MIP_ROI_3, v02_MIP_ROI_32. All 80 frames opened and inspected.
Frames: figures/pilot/p6/frames/<chamber>/<visit>.jpg. Volumes are um^3 (the panel prints one number when both columns agree).
Verdict key: better / same / mixed / worse = C6 target outline relative to C4.

## Headline

C6 is identical to C4 on 72 of 80 frames (same outline, same volume, same T cell). It differs on 8 frames:
3 better, 4 mixed (all in the crowded tumour cluster v01_MIP_ROI_32), 1 clearly worse (v02_MIP_ROI_3 visit 20: target lost).
No chamber gives evidence that C6 is a broad improvement. No dead (yellow) target lost its outline except the last frame of v02_MIP_ROI_3
(and there the cell is not yellow; it is a dim magenta/green cell). The orange T cell was the same in both columns on every frame.
Several shared failures (identical in C4 and C6) remain and are listed at the end.

## v01_MIP_ROI_2 (one tumour cell, T cell on top; no SYTOX signal at any visit)

| Visit | Verdict | Note |
|---|---|---|
| 01 | same | identical, 503 |
| 02 | same | identical, 567 |
| 03 | same | identical, 566 |
| 04 | same | identical, 740 |
| 05 | better | C4 had grey fragments cut out of the inside of the blue target (551) plus a separate grey magenta blob above; C6 is one clean blue outline (788). The faint blob above the cell is dropped by C6 (debris, acceptable). |
| 06 | same | identical, 620 |
| 07 | same | identical, 657 (grey slivers inside the target are present in both) |
| 08 | better | C4 cut a grey piece off the upper-left of the target (458, looks like a (a) wrong cut); C6 keeps one blue outline (788). |
| 09 | same | identical, 589 |
| 10 | same | identical, 442 |
| 11 | same | identical, 566 |
| 12 | better | C4 had a grey region inside the target (435); C6 one blue outline (582). |
| 13 | same | identical, 634 |
| 14 | same | identical, 504 |
| 15 | same | identical, 598 (small grey sliver inside the target in both) |
| 16 | same | identical, 818 |
| 17 | same | identical, 475 |
| 18 | same | identical, 585 |
| 19 | same | identical, 549 |
| 20 | same | identical, 460 |

Tally: better 3, same 17, mixed 0, worse 0.
Target/T cell: correct in both columns on every visit. In visits 13-20 the single orange outline is large and elongated and may be two T cells merged (same in C4 and C6; not a C6 change). Target volume across C6: 503, 567, 566, 740, 788, 620, 657, 788, 589, 442, 566, 582, 634, 504, 598, 818, 475, 585, 549, 460. Swings of 30 percent between visits are present in both methods (mostly T-cell contact leaking into the target outline), so (f) is a shared problem here, not a C6 one.

## v01_MIP_ROI_32 (crowded cluster: target cell plus a second tumour cell lower-left, T cell upper-right)

| Visit | Verdict | Note |
|---|---|---|
| 01 | same | identical, 1145. Lower-left tumour cell is grey in both (correct). |
| 02 | better | C4 blue (1775) merges the target with the lower-left tumour cell into one object; C6 (1080) separates the lower-left cell as grey and keeps the target. Matches the 1145 -> 1370 trend. This is case (b). C6 also adds extra grey subdivisions in the XZ panel (mild fragmentation). |
| 03 | same | identical, 1371 |
| 04 | same | identical, 963. Blue covers only the upper-left part of the cluster and big grey remainder (a) in both. |
| 05 | same | identical, 1199 (same grey remainder) |
| 06 | same | identical, 901 |
| 07 | same | identical, 1140 |
| 08 | same | identical, 1311 |
| 09 | mixed | C4 (1485) has a thin blue tail running into the lower-left tumour cell (wrong); C6 (1050) removes the tail but, in the XZ panel, now shows several grey pieces cut out of the target body and a blue strip. Right direction on the tail, new fragmentation inside the main cell. |
| 10 | same | identical, 986 |
| 11 | same | identical, 889 |
| 12 | mixed | C4 (911) has a blue tail into the lower-left cell; C6 (602) drops it but also has grey pieces between the blue target and the orange T cell. 889 -> 602 -> 760 is a dip not seen in C4 (889 -> 911 -> 1063). |
| 13 | mixed | same pattern: C4 1063 with tail, C6 760 with grey fragments in the XZ panel. Volume of C6 is probably an under-estimate; C4 is probably an over-estimate. |
| 14 | same | identical, 906 |
| 15 | same | identical, 817 |
| 16 | same | identical, 885 (blue tail into lower-left cell remains in both) |
| 17 | same | identical, 808 (tail remains in both) |
| 18 | same | identical, 872 |
| 19 | same | identical, 773 |
| 20 | same | identical, 342. Target collapses from 773 to 342; T cell now covers much of the tumour cell (strong green/white overlap) and the blue outline is a strip with grey remainders. Same in both. |

Tally: better 1, same 16, mixed 3, worse 0.
Target/T cell: target is the right cell throughout in both. Orange T cell correct and identical in both.
Honest read: the only visit where C6 is clearly right is 02. On 09, 12, 13 C6 trades one error (tail into the neighbour cell) for another (fragmentation of the target plus a volume dip). I would not call those better.

## v02_MIP_ROI_3 (round tumour cell at the field edge, T cell at right edge; dies late)

| Visit | Verdict | Note |
|---|---|---|
| 01 | same | identical, 970 (cell cropped by the image edge) |
| 02 | same | identical, 663 |
| 03 | same | identical, 197 (cell mostly outside the field; shared volume collapse) |
| 04 | same | identical, 765 |
| 05 | same | identical, 636 |
| 06 | same | identical, 631 |
| 07 | same | identical, 821 |
| 08 | same | identical, 628 |
| 09 | same | identical, 643 |
| 10 | same | identical, 676 |
| 11 | same | identical, 743 |
| 12 | same | identical, 788. C4 had a tiny grey magenta object at the edge in XZ, C6 drops it (negligible). |
| 13 | same | identical, 798; two yellow SYTOX blobs inside, outline kept in both |
| 14 | same | identical, 875 |
| 15 | same | identical, 900 |
| 16 | same | identical, 180. Cell is dye-filled (yellow) and shrunken; blue outline kept in both (c fine). Grey sliver at lower edge in both. |
| 17 | same | identical, 183, yellow cell outlined in both |
| 18 | same | identical, 112, yellow cell outlined in both |
| 19 | same | identical, 114. C4 label tr=3, C6 tr=1, outlines identical. |
| 20 | worse | C4 still has a blue target (tr=3, 187). C6 has no target (Target tr=0, vol 0.0): the cell is outlined grey, so it was segmented but the tracker dropped the target. Failure type (e)/(f). Not a dead-dye case (cell is dim magenta with a green spot), but it is a lost target on the final visit. |

Tally: better 0, same 19, mixed 0, worse 1.
T cell (orange, at the right edge): correct and identical on all visits.
Shared issues (both C4 and C6): target volume 970 -> 663 -> 197 -> 765 over visits 1-4 is mostly the image edge cropping the cell. Volume 900 (v15) -> 180 (v16) -> 112 (v18) is the cell dying and losing its magenta membrane signal; the outline does track the yellow cell, so this looks real, but "volume" after death is a different quantity from before and should not be compared directly.

## v02_MIP_ROI_32 (round tumour cell with green inclusions, T cell; becomes dye-filled late)

| Visit | Verdict | Note |
|---|---|---|
| 01 | same | identical, 425 |
| 02 | same | identical, 542 |
| 03 | same | identical, 698 |
| 04 | same | identical, 854 |
| 05 | same | identical, 834 |
| 06 | same | identical, 900 (small blebs outlined in both) |
| 07 | same | identical, 935 |
| 08 | same | identical, 930 |
| 09 | same | identical, 883 |
| 10 | same | identical, 850 |
| 11 | same | identical, 896. Grey object and blue lobe sit over T-cell-green material between the cells in both (possible T-cell/target mixing, unchanged by C6). |
| 12 | same | identical, 943 |
| 13 | same | identical, 834 |
| 14 | same | identical, 1055 (blue reaches over the T-cell edge in both) |
| 15 | same | identical, 977. C4 has a separate small grey blob right of the cell (faint green debris); C6 drops it. Not a real tumour cell, harmless. |
| 16 | same | identical, 1070 |
| 17 | same | identical, 945 |
| 18 | same | identical, 1088; yellow dye appearing, outline kept |
| 19 | same | identical, 1173; yellow dead cell kept as target in both |
| 20 | same | identical, 1228; yellow dead cell kept as target in both (C4 id tr=2, C6 tr=1, same outline) |

Tally: better 0, same 20, mixed 0, worse 0.
Target and T cell correct on all visits. No (a) or (c) failures. Target volume rises smoothly 425 -> 1228 with no collapse.

## Overall tally

| Chamber | better | same | mixed | worse |
|---|---|---|---|---|
| v01_MIP_ROI_2 | 3 | 17 | 0 | 0 |
| v01_MIP_ROI_32 | 1 | 16 | 3 | 0 |
| v02_MIP_ROI_3 | 0 | 19 | 0 | 1 |
| v02_MIP_ROI_32 | 0 | 20 | 0 | 0 |
| Total | 4 | 72 | 3 | 1 |

(Totals add to 80: 4 + 72 + 3 + 1.)

## Worst frames (C6 vs C4)

1. v02_MIP_ROI_3 visit 20: C6 loses the target entirely (tr=0, vol 0.0) while C4 keeps it (187). Only outright regression found.
2. v01_MIP_ROI_32 visit 12: C6 target 602 (C4 911); tail into the neighbour tumour cell removed, but grey fragments next to the target and a volume dip between 889 and 760.
3. v01_MIP_ROI_32 visit 13: C6 target 760 (C4 1063); same fragmentation pattern in the XZ panel.
4. v01_MIP_ROI_32 visit 09: C6 target 1050 (C4 1485); tail removed but target body now cut by grey pieces, XZ shows several sub-regions.

## Failures that C6 does NOT fix (identical in C4 and C6)

- v01_MIP_ROI_32 visit 04 and 05: blue covers only part of the cluster, big grey remainder (case a).
- v01_MIP_ROI_32 visits 16-17: blue tail into the neighbouring lower-left tumour cell persists.
- v01_MIP_ROI_32 visit 20: target volume 773 -> 342, T cell over the tumour cell.
- v01_MIP_ROI_2 visits 13-20: orange T cell outline is long and probably merges two T cells.
- v02_MIP_ROI_32 visit 11: blue lobe and grey object over T-cell-green material between the cells.
- v02_MIP_ROI_3 visit 03: volume 197 because the cell is cut by the field edge.

## Where C6 helped

- v01_MIP_ROI_32 visit 02: neighbouring tumour cell correctly separated from the target (case b).
- v01_MIP_ROI_2 visits 05, 08, 12: no grey cut-outs inside the target; volume closer to neighbours.
