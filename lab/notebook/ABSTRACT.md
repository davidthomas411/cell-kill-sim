## Abstract (updated 29 October 2026, after pilot round 3)

**Aim.** Build 3D-over-time "digital twins" of CAR-T cells attacking leukaemia
cells, from the Fei lab's public light-sheet movies, and ask whether the way a T
cell pushes on its target predicts killing.

**What has been done.**
- We reproduced the lab's published contact areas exactly.
- Early results suggested that deeper dents go with faster killing. Visual review
  then showed that the lab's cell outlines (masks) are often wrong: hollow, two
  cells merged, or with lobes where there is no cell. These errors inflate the
  dents, so all dent and killing results are on hold and the mechanical model is
  parked.
- Since then we have been redrawing the cells from the raw images, comparing
  methods on movies of 8 chambers and reviewing every frame. Three rounds so far:
  - A general AI outliner (Cellpose-SAM) failed.
  - Repairing the lab's masks cut whole cells into pieces.
  - A membrane-based method (PlantSeg), improved by us into "C4", is the best so far.

**Where we are now.**
- C4 outlines each cell tightly and keeps dying cells, which the lab's masks drop.
- It still sometimes merges touching tumour cells.
- Its tracking of which cell is which over time slips.
- The review found that two chambers labelled "survived" contain a target that
  dies after the T cell leaves. Our kill labels depend on how a kill is defined,
  so we are asking the Fei lab.

**What's next.**
1. Test C4, unchanged, on 12 chambers it was not tuned on (running now).
2. Fix tracking and the merging of touching cells.
3. Agree on what counts as a kill.
4. Redraw all 502 chambers by processing each movie as it downloads.
5. Redo the dent and killing analysis on the new outlines.
