---
id: 2026-10-28-07-clean-subset-gallery-review-mask-only-rule-fails
date: '2026-10-28'
title: 'Clean-subset gallery review: mask-only rule fails against raw images'
gate: G1
status: reviewed
author: claude
reviewers: [claude]
signoff: null
code_commit: b1fde32
datasets: [clean_subset_v0.1, qc_flags, lab_masks, raw_volumes]
protocols: [clean-cell-rule@v1, gallery-review@v1]
galleries: [clean-2026-10-28]
supersedes: null
superseded_by: null
---

## Question

Does clean-cell rule v1 (status/clean_subset_criteria.md: one piece, lab raw
label fraction >= 0.9, not at the edge, volume 200-1500 um^3, volume within
0.7-1.4x of the chamber median) keep only whole, correctly segmented targets?
And does it reject only bad ones?

## What was run

src/review/make_clean_gallery.py by agy (Argon High). 110 panels, each with raw
channel 2 and the U-Net labels as deposited; report .agy_clean_gallery_report.md.

## What was looked at

All 110 panels, reviewed by Claude. Verdicts are in lab/reviews/clean-2026-10-28.csv.

## Observations

**Clean visits (rule says keep)**
- Random clean, 52 panels: 21 ok (40%), 17 problem, 14 unsure.
- Extremes, 18 panels: 6 ok, 9 problem, 3 unsure.
- Failure types the rule cannot see, because it uses mask properties only:
  - a hollow mask inside a solid raw cell: all 3 hd10 dasatinib randoms
    (clean_rand_46-48), plus 10, 36 and ext_12;
  - mask lobes where the raw image shows no cell: clean_rand_01, 13, 17, 19,
    23, 30, 44;
  - two raw cells merged into one mask: clean_rand_43, ext_02, 03, 04, 08, 11.
- Every one of the 5 deepest clean cups (5.3-5.8 um) is a merge or fragment.
  Deep cups that are real exist: clean_rand_39 (3.7 um), with the raw target
  visibly wrapping the T cell.
- Good matches are common in v01-v05, v07 and v08.

**Rejected visits (rule says drop)**
- C1 (extracted pieces): 7 of 8 correctly rejected.
- C3 (edge): 8 of 8 correct.
- C4 (volume): 7 of 8 correct. Borderline-small cat4_07 is a real small cell.
- C2 (lab labels in pieces): only 3 of 8 correct. In 3 panels (cat2_01, 06, 08)
  a good target failed because of detached U-Net specks elsewhere.
- C5 (volume vs chamber median): only 2 of 8 correct. In 4 panels the median
  is pulled by bad visits, so good masks are rejected.

## Decision

Rule v1 is rejected. It keeps too many bad masks and drops good ones (C2, C5).
The clean subset v0.1 is not usable.

Next, rule v2 must use the raw image. Proposed: agreement between each target
mask and raw channel 2. For example, the fraction of mask voxels above a local
raw threshold, and the fraction of the raw target signal inside the mask. This
would replace C2 and C5, and keep C1, C3 and C4 (bounds to be re-tuned).
Calibrate the thresholds on these 110 verdicts, then validate on a fresh
gallery. This needs raw volumes for every visit used, which is about 75 MB per
sweep: tens of GB for a sweep-per-visit subset, so the size must be decided.

## Open issues

- Raw channel identity: channel 2 is brighter in the target core than at the
  rim, so it is likely a whole-cell label, not membrane. Gallery captions say
  "561 nm membrane". Confirm the channel meaning with the Fei lab.
- Per-sweep QC needed (see the cup-depth validation entry).
