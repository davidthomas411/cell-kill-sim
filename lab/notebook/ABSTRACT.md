## Abstract (updated 29 October 2026, after P6)

**Aim.** Build 3D-over-time "digital twins" of CAR-T cells attacking leukaemia
cells, from the Fei lab's public light-sheet movies, and ask whether the way a T
cell pushes on its target predicts killing.

**What has been done.**
- We reproduced the lab's published contact areas exactly.
- Visual review then showed that the lab's cell outlines (masks) are often wrong:
  hollow, two cells merged, or with lobes where there is no cell. These errors
  inflate the dents, so all dent and killing results are on hold and the mechanical
  model is parked.
- We are redrawing the cells from the raw images and reviewing every frame. An
  improved membrane-based method ("C4") was best on 8 chambers and held up on 12
  it had never seen.
- Round 5 tried a splitter for touching tumour cells; it cut single cells and deleted
  dead cells, so we rejected it. Round 6 rewrote it to be more careful.

**Where we are now.**
- The careful splitter changed little: over 311 frames it matched the earlier outlines
  in 3 in 4, was better in 1 in 8 and worse in 1 in 16. It still cuts some touching
  cells along lines with no visible gap.
- In about half the chambers it is unclear even by eye which tumour cell is "the
  target" (two touch the T cell, or neither does). **We decided to leave out anything
  too ambiguous for now:** 10 of the 20 test chambers are set aside, 2 are limited by
  merged cells, and 8 are clean. On the clean 8, the new outlines plus a whole-movie
  tracker are as good as or better than before.
- Open for the Fei lab: how a kill is defined, and what the magenta signal in one
  chamber is.

**What's next.**
1. Test on more chambers how often a target is unambiguous, before redrawing all 502.
2. Agree on what counts as a kill, then redo the death labels on the clean chambers.
3. Redraw all 502 chambers, processing each movie as it downloads.
4. Redo the dent and killing analysis on the clean outlines.
