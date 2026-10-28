# Clean-cell subset: criteria fixed before any outcome is looked at

Written 2026-10-28, from mask properties only (`data/qc/per_visit_qc.csv`,
`src/mask_qc.py`). No death labels or associations were used to choose these.

## Rule (per contact visit, trace_num == 1)

A visit is clean when all of these hold:

1. `target_n_components == 1` and `tcell_n_components == 1`: our extracted
   target and T cell are each one connected object.
2. `target_raw_largest_frac >= 0.9`: in the lab's U-Net labels as deposited,
   the largest target piece holds at least 90% of the target voxels. This
   rejects targets that the network itself labelled in pieces.
3. `touches_fov_edge == False`: neither cell is cut by the image border.
4. `200 <= target_volume_um3 <= 1500`: a plausible single Nalm6 cell. This
   rejects tiny fragments and most merged pairs.
5. `0.7 <= target_vol_ratio_to_own_median <= 1.4`: target volume is stable
   relative to its own chamber, so no sudden merge or split at that visit.

Deliberately **not** used: target solidity or sphericity. A real cup lowers
solidity (Spearman rho = -0.26 with cup depth), so filtering on it would
remove deep cups and bias exactly the quantity under study.

## Size (counted from masks only)

- 1,213 of 4,523 contact visits (27%); 243 chambers with at least 1 clean
  visit, 111 with at least 4.
- By experiment, clean fraction: v01 0.28, v02 0.34, v03 0.34, v04 0.40,
  v05 0.29, v06 0.19, v07 0.16, v08 0.38, v09 (hd10) 0.05, v10 0.31,
  v11 (hd2 dasatinib) 0.04, v12 (hd10 dasatinib) 0.16, v13 (hd5 dasatinib)
  0.14.
- Consequence: dasatinib (60 visits) and donor hd10 (18 untreated visits) are
  thin. The clean subset supports within-experiment analyses in v01-v05, v08
  and v10. It cannot support the 5-vs-5 donor comparison or the dasatinib
  control on its own.

## Before use

The subset passes only after a real-image gallery of it (random visits plus
the extremes of cup depth, volume and raw-label fraction, raw membrane under
every panel) is reviewed panel by panel and shows whole, separate, solid
cells. Associations with death also wait for gate G2 (death labels).
