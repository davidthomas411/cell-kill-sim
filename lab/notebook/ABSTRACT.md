## Abstract (updated 29 October 2026, after the held-out test)

**Aim.** Build 3D-over-time "digital twins" of CAR-T cells attacking leukaemia
cells, from the Fei lab's public light-sheet movies, and ask whether the way a T
cell pushes on its target predicts killing.

**What has been done.**
- We reproduced the lab's published contact areas exactly.
- Visual review then showed that the lab's cell outlines (masks) are often wrong:
  hollow, two cells merged, or with lobes where there is no cell. These errors
  inflate the dents, so all dent and killing results are on hold and the mechanical
  model is parked.
- We are redrawing the cells from the raw images, comparing methods on movies and
  reviewing every frame. After three rounds on 8 chambers, an improved membrane-based
  method ("C4") was best.
- We then tested C4, unchanged, on 12 chambers it had never seen.

**Where we are now.**
- C4's outlines held up on the new chambers: solid, tight, and without the lab
  masks' phantom lobes, including in the donor whose lab masks are hollow.
- Two problems now dominate:
  - touching tumour cells merge into one (5 of 12 chambers);
  - the wrong cell is often picked as "the target" or "the T cell".
- One chamber may contain no tumour cell at all.
- Separately, two chambers labelled "survived" contain a target that dies after the
  T cell leaves. How a kill is defined is a question for the Fei lab.

**What's next.**
1. Fix cell choice and tracking, and the separation of touching tumour cells.
   Retest on all 20 chambers.
2. Agree on what counts as a kill.
3. Redraw all 502 chambers by processing each movie as it downloads.
4. Redo the dent and killing analysis on the new outlines.
