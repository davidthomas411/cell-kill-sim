## Abstract (updated 29 October 2026, after P5)

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
- Round 5 tried two fixes on all 20 chambers. Four reviewers looked at every frame.

**Where we are now.**
- A splitter for touching tumour cells helped in about 1 frame in 6, changed nothing
  in 3 in 5, and made things worse in 1 in 5. It cut single cells in two and deleted
  some dead cells, which are the kill evidence. We rejected it.
- A tracker that picks the T cell and target once from the whole movie fixed a few
  wrong choices and caused a few; it is not yet better overall.
- Where tumour cells stay merged, it is unclear even by eye which cell is the target.
  Two crowded chambers and two faint ones may need to be left out.
- Open for the Fei lab: how a kill is defined (two "survived" chambers contain a
  target that dies after the T cell leaves), and what the magenta signal in one
  chamber is.

**What's next.**
1. Rewrite the splitter to cut only at a visible gap and never delete a dead cell;
   review every frame again.
2. Agree on what counts as a kill, and on which chambers are usable.
3. Redraw all 502 chambers, processing each movie as it downloads.
4. Redo the dent and killing analysis on the new outlines.
