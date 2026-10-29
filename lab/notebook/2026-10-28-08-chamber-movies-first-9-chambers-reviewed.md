---
id: 2026-10-28-08-chamber-movies-first-9-chambers-reviewed
date: '2026-10-28'
title: 'Chamber movies: first 9 chambers reviewed'
gate: G2
status: reviewed
author: claude
reviewers: [claude]
signoff: null
code_commit: c5e88f1
datasets: [raw_volumes, lab_masks, step4_death_labels_v2, step5_descriptors]
protocols: [gallery-review@v1]
galleries: []
supersedes: null
superseded_by: null
---

## Question

Do the chamber movies (src/review/movies.py: raw top/side views of all visits,
mask outlines, and a strip chart of cup depth, contact area, target volume and
the death call) let us check the per-visit numbers against what the cells do?
What do they show for the 9 fully downloaded chambers?

## What was run

`src/review/movies.py`, run by agy (Argon High). It produced 18 clips (visits and
burst) of 157-949 KB each, listed in figures/movies/index.csv.

## What was looked at

The first, middle and last frame of every visits movie (9 chambers), reviewed by Claude.

## Observations

- The format works. Death-dye yellow, the T cell (green) and the target (magenta)
  are clearly separable, and the strip chart makes numeric jumps easy to match to images.
- **Possible missed death, v03 ROI 30** (labelled survived): at visit 20 a magenta
  cell is bright yellow with dye. Contact had ended (contact area 0), so either this
  is the contacted target dying after the T cell left, or it is a neighbour. Check
  identity across visits.
- **Death calls flicker, v03 ROI 6** (killed at visit 8): the per-visit call
  alternates dead and alive after visit 8, and visit 20 is "alive" while yellow dye
  is visible. Is a dead call lost when the dying target's mask fragments or the
  contact ends? Same pattern at v01 ROI 32, where only visits 9-11 are called
  dead although the target stays dye-positive.
- **Volume jumps from merges, v05 ROI 8:** target volume spikes to 2,277 um3 at
  visit 10 (two cells merged into one mask), with a matching cup-depth spike.
- **v06 ROI 28** (low-killing donor hd8): fragmented, scattered outlines. Target
  volume is 17 um3 at visit 1, and the chamber is crowded with dying cells. Low-quality chamber.
- **Tracking switches target, v02 ROI 3:** contact ends at visit 2, and the outlined
  "target" later is a different cell.

## Decision

Movies become the standard review view. G2 (death labels) needs a check of the
whole death timeline per chamber, not single visits, including dye on the contacted
target after contact ends.

## Open issues

- Death labels: explain the dead/alive flicker in v03 ROI 6, and the missed late
  death in v03 ROI 30.
- Raw volumes exist for only 9 chambers. Movies for more chambers need one sweep
  per visit downloaded (about 1.5 GB per chamber).
