# Verification plan: a quality dataset, stage by stage

Written 2026-10-28 after the visual review (`status/visual_review_2026-10-28.md`).
Priority: a verified dataset. No stage feeds the next until it passes its gate.

## How every gate works

1. Render a real-data gallery for the stage: random examples (fixed seed,
   stratified by condition and donor) plus the extremes of each quantity.
   Raw membrane/actin images under every panel (volumes fetched with
   `src/download_raw.py --file-list`).
2. One colour scheme across all panels: target blue, T cell orange, other
   cells grey (labelled), model shapes dashed in fixed colours. A full key on
   every panel.
3. Claude reviews every panel and writes findings by panel ID. The user
   reviews the same gallery.
4. Fix what the review finds, re-render the same panels, review again.
5. Commit the gallery code, the findings and the pass/fail note. Only then
   does the next stage use the outputs.

## Gates

| Gate | Stage | What must hold | Status |
|---|---|---|---|
| G1 | Cell identity (masks to tracked T cell + contacted target) | The tracked T cell and its contacted target are each one whole, separate cell. Flag: fragmented, merged (two cells in one object), hollow, cut by the image edge, tiny fragments. For every flagged case, find the failing layer by showing side by side the raw membrane image, the lab's U-Net labels as deposited (classes 0-4), our extracted object, and the measurement: (a) U-Net labels wrong while the raw cell is whole means a lab segmentation error (repair, exclude, or re-segment from raw); (b) labels fine but our object wrong means our extraction (fix the code); (c) the raw cell itself breaking up means real biology (keep, treat as a possible death signal). Per-experiment rates of (a), (b) and (c) reported (hd10 first). Masks are not assumed correct: matching the paper's numbers does not validate them, since the lab used the same masks. | Open: QC job running |
| G2 | Death labels | Overlap is with the contacted target only, in the code and in the panels. Every killed call shows dye inside the target that persists; neighbour-overlap false kills removed (3D overlap or contacted-target projection). Threshold sensitivity (403-410 ADU, area 20-50 px) reported. | Open: gallery/label mismatch to resolve |
| G3 | Contact and cup descriptors | Recomputed on G1-clean cells only. Gallery of random and extreme cups shows real dents, no hull bridging across gaps. Z-spacing caveat kept. | Waits on G1 |
| G4 | Associations (5b, 7) | Rerun on G1-G3 outputs. Reported as re-analysis of committed plans, with the earlier results kept for comparison. | Waits on G2, G3 |
| G5 | Mechanical model | Only if G4 still shows something to explain. Pin rigid-body motion, register the reference shape, check re-solve reproducibility and curve inversion, real-data fit gallery. | Parked |

## Done so far

- G0 (reproduce the lab's numbers): passed on external checks (395/395 row
  counts; contact area within 0.5% for 369/419). Not yet checked visually;
  G1 covers the same masks.
