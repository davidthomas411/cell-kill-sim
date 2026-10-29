"""C6 post-processing of C4 target masks: conservative neck/valley splitter, dead/dim cell preservation, and temporal consistency.

Provides:
  split_merged_targets(raw, t_inst, b_inst, c0_b_inst=None, tracked_t_id=0, tracked_b_id=0) -> b_inst_new
  process_chamber_visits(...) -> runs per-visit cleaning/splitting + temporal consistency check
  main() -> processes all 20 chambers (PILOT_CHAMBERS + PILOT4_CHAMBERS), sweep 1 of every visit,
            saving data/pilot_seg/C6/<experiment>/<roi>/<stem>.npz.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
)
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mask_qc import bbox_slices
from src.seg.candidates import (
    PILOT4_CHAMBERS,
    PILOT_CHAMBERS,
    _relabel_sequential,
    save_candidate_npz,
)
from src.seg.split5 import _channel_zscores, _find_split_seeds_and_cost

# ---------------------------------------------------------------------------
# Documented C6 thresholds
# ---------------------------------------------------------------------------

# 1. Non-cell removal & dead/dim cell preservation thresholds
MIN_CELL_VOXELS = 7000             # Minimum 3D volume (voxels); removes tiny wall beads (<7k) while keeping small dim/dead cells
MIN_CELL_MAX_DT = 5.70             # Minimum 3D inscribed radius (voxels); rejects flat wall slabs (e.g. v08_MIP_ROI_24 v19 dt=5.66)
SYTOX_404_THRESH = 404.0           # Gaussian(sigma=3) ch0 ADU threshold defining SYTOX-positive dead-cell dye
SYTOX_KEEP_MIN_FRAC = 0.03         # Minimum fraction of SYTOX(>404) voxels to unconditionally protect a dead cell
SYTOX_KEEP_MIN_VOX = 200           # Minimum count of SYTOX(>404) voxels to unconditionally protect a dead cell
SYTOX_MASK_KEEP_MIN_VOX = 350      # Minimum count of combined sytox_mask voxels to protect a dying/dead cell
HAZE_SLAB_MAX_DT = 7.45            # Max inscribed radius for elongated wall-haze slabs (v10_MIP_ROI_28)
HAZE_SLAB_MIN_RATIO = 44.0         # Minimum vox / (max_dt^3) ratio identifying flat/elongated haze slabs
HAZE_MAX_VOXELS = 19500            # Max volume for faint background haze blobs
HAZE_MAX_DT = 8.20                 # Max inscribed radius for faint background haze blobs
HAZE_MIN_Z2_P90 = 14.2             # Minimum 90th-percentile ch2 z-score above which small objects are kept
TCELL_INCLUSION_MAX_VOX = 29000    # Max volume for intra-T-cell ch2 inclusion (v12_MIP_ROI_23)
TCELL_INCLUSION_MIN_NBT = 0.38     # Minimum surface contact fraction with T cell for intra-T-cell inclusion removal
TCELL_INCLUSION_MIN_Z1_P50 = 4.85  # Minimum median ch1 z-score indicating green T-cell body

# 2. Concentric / over-split single-cell re-merge thresholds
REMERGE_MIN_SURFACE_FRAC = 0.32    # Surface contact fraction of smaller fragment touching larger fragment of same cell
REMERGE_C0_SHARED_SURF = 0.15      # Surface contact fraction when both fragments share the same single C0 cell
REMERGE_MAX_FRAG_VOX = 45000       # Max volume of smaller fragment eligible for re-merge
REMERGE_MAX_UNION_VOX = 110000     # Max combined volume for re-merging fragments (132000 when no C0 cell >=4k)
REMERGE_MIN_UNION_BF = 0.18        # Minimum bounding-box fill ratio of combined single cell

# 3. Guard and candidate split thresholds
SPLIT_MIN_OBJ_VOXELS = 90000       # Minimum object volume eligible for splitting into >=2 tumour cells
GUARD_ROUND_MIN_BF = 0.355         # Bounding-box fill ratio above which a compact 1-peak cell is kept whole
GUARD_ROUND_MAX_SPAN = 118         # Max bounding-box axis span for the compact round-cell guard
GUARD_SINGLE_MAX_VOX = 115000      # Objects below this volume with 1 DT peak, bf >= 0.185, and <=1 C0 match stay whole
GUARD_SINGLE_MIN_BF = 0.185        # Minimum bbox fill ratio for moderate-volume single-cell guard
GUARD_ISOLATED_MAX_VOX = 175000    # Max volume for isolated (nbB==0) single-C0 bright-cell guard
GUARD_ISOLATED_MIN_C0_OV = 30000   # Minimum overlap with single C0 cell for isolated bright-cell guard
GUARD_ISOLATED_MIN_Z2 = 16.0       # Minimum median ch2 z-score for isolated bright-cell guard

C0_SEED_MIN_OVERLAP_VOX = 4000     # Minimum voxel overlap for a C0 instance to count as multi-cell evidence
C0_SEED_MIN_OVERLAP_FRAC = 0.05    # Minimum fraction of C0 instance volume overlapping the C4 object
DT_PEAK_MIN_DIST = 18              # Minimum 3D distance (voxels) between interior DT peaks
DT_PEAK_MIN_VAL = 6.8              # Minimum smoothed DT value (voxels) for a valid cell-core seed
DT_HMAX_H = 1.6                    # h-maxima prominence (voxels) on smoothed distance transform

# 4. Post-split validation contract thresholds (Rules 1a, 1b, 1c, 4)
SPLIT_MIN_CHILD_FRAC = 0.25        # Rule 1c: Minimum child volume fraction of parent (>= 25% of parent volume)
SPLIT_MIN_CHILD_VOX = 18000        # Rule 1c: Minimum absolute child volume (voxels)
SPLIT_MIN_CHILD_DT = 8.5           # Rule 1c: Minimum child inscribed radius (voxels, min 3D thickness >= 17 px)

PLANAR_MIN_RMS = 0.70              # Rule 1a: Minimum RMS distance (voxels) to best-fit plane unless coinciding with a neck & valley
JAGGED_MAX_RMS = 1.68              # Rule 1a: Maximum balanced-cut RMS distance (voxels) to plane when not backed by C0 pair
JAGGED_MAX_INIT_RMS = 2.25         # Rule 1a: Maximum initial-cut RMS distance (voxels) to plane when not backed by C0 pair
JAGGED_MAX_C0_RMS = 4.60           # Rule 1a: Maximum balanced-cut RMS distance (voxels) when backed by C0 two-instance overlap
JAGGED_MAX_ROUGH_RATIO = 1.98      # Rule 1a: Maximum ratio of cut voxels to projected ellipse area

FAINT_MIN_CORE_Z2 = 10.8           # Rule 1b: Minimum interior core ch2 z-score unless backed by clean C0 two-instance overlap
VALLEY_MIN_CORE_DIFF = 1.25        # Rule 1b: Minimum interior core z2 minus cut median z2 for 1-peak objects
VALLEY_MIN_P75_DIFF = 1.20         # Rule 1b: Minimum child 75th-percentile z2 minus cut median z2
VALLEY_MIN_P25_DIFF = 5.00         # Rule 1b: Minimum interior core z2 minus cut 25th-percentile z2
NECK_MAX_LINE_DT_RATIO = 0.85      # Rule 1b: Maximum DT saddle-to-child-radius ratio along line connecting child cores

TEMPORAL_MAX_DROP_FRAC = 0.35      # Rule 4: Max allowed single-visit target volume drop fraction (35%) from a split alone
TEMPORAL_NEIGHBOR_MATCH_TOL = 0.35 # Rule 4: Relative volume tolerance matching unsplit parent to adjacent visits


def remove_non_cell_objects_c6(
    b_inst: np.ndarray,
    t_inst: np.ndarray,
    raw: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    sytox_mask: np.ndarray,
    c0_b_inst: Optional[np.ndarray] = None,
    tracked_t_id: int = 0,
    tracked_b_id: int = 0,
) -> Tuple[np.ndarray, int]:
    """Remove non-cell objects (tiny wall beads, flat slabs, background haze) while preserving dead & dim cells."""
    uids = [int(u) for u in np.unique(b_inst) if u > 0]
    if not uids:
        return np.zeros_like(b_inst, dtype=np.uint16), 0

    d0_sm3 = gaussian_filter(raw[0].astype(np.float32), sigma=3.0)
    sytox_404 = d0_sm3 > SYTOX_404_THRESH

    vols = {u: int((b_inst == u).sum()) for u in uids}
    max_chamber_vox = max(vols.values()) if vols else 0

    out = np.zeros_like(b_inst, dtype=np.uint16)
    n_removed = 0

    for u in uids:
        m = b_inst == u
        vox = vols[u]

        # Rule 1: Tiny wall beads / debris (< 7,000 voxels)
        if vox < MIN_CELL_VOXELS:
            n_removed += 1
            continue

        sl = bbox_slices(m, 2, m.shape)
        m_sub = m[sl]
        dt = distance_transform_edt(m_sub)
        max_dt = float(dt.max())

        s404_vox = int(sytox_404[sl][m_sub].sum())
        s404_frac = s404_vox / float(vox)
        s_vox = int(sytox_mask[sl][m_sub].sum())
        is_sytox_dead = (
            (s404_frac >= SYTOX_KEEP_MIN_FRAC)
            or (s404_vox >= SYTOX_KEEP_MIN_VOX)
            or (s_vox >= SYTOX_MASK_KEEP_MIN_VOX)
        )

        # Rule 2a: Never remove SYTOX-positive dead/dying cells
        if is_sytox_dead:
            out[m] = u
            continue

        z2_sub = z2[sl][m_sub]
        z1_sub = z1[sl][m_sub]
        z2_p90 = float(np.percentile(z2_sub, 90))
        z2_p50 = float(np.percentile(z2_sub, 50))
        sig65_frac = float((z2_sub > 6.5).mean())

        t_sub = t_inst[sl]
        dil = binary_dilation(m_sub, iterations=1) & (~m_sub)
        tot_dil = max(1, int(dil.sum()))
        t_any_touch = int((t_sub > 0)[dil].sum())
        t_tr_touch = int((t_sub == tracked_t_id)[dil].sum()) if tracked_t_id > 0 else t_any_touch
        nbt = t_any_touch / float(tot_dil)
        c0_ov = int((m_sub & (c0_b_inst[sl] > 0)).sum()) if c0_b_inst is not None else 0

        # Rule 2b: Never remove the tracked target (unless it is a small faint haze blob while a large cell exists)
        if u == tracked_b_id and tracked_b_id > 0:
            is_secondary_haze = (max_chamber_vox >= 40000 and vox < 21000 and z2_p50 < 9.0 and c0_ov < 6000)
            is_green_inclusion = (vox < TCELL_INCLUSION_MAX_VOX and float(np.percentile(z1_sub, 50)) >= TCELL_INCLUSION_MIN_Z1_P50 and nbt >= TCELL_INCLUSION_MIN_NBT and c0_ov == 0)
            if not is_secondary_haze and not is_green_inclusion:
                out[m] = u
                continue

        # Rule 2c: Never remove a compact cell that touches the tracked T cell
        if t_tr_touch > 0 and max_dt >= 7.5 and z2_p90 >= 14.0:
            z1_p50 = float(np.percentile(z1_sub, 50))
            if not (vox < TCELL_INCLUSION_MAX_VOX and z1_p50 >= TCELL_INCLUSION_MIN_Z1_P50 and nbt >= TCELL_INCLUSION_MIN_NBT and c0_ov == 0):
                out[m] = u
                continue

        # Rule 3: Flat wall slabs (e.g. v08_MIP_ROI_24 v19 dt=5.66)
        if max_dt < MIN_CELL_MAX_DT and z2_p90 < 15.0:
            n_removed += 1
            continue

        # Rule 4: Elongated/flat background haze slabs (e.g. v10_MIP_ROI_28 haze)
        slab_ratio = vox / max(1.0, max_dt ** 3)
        if max_dt < HAZE_SLAB_MAX_DT and slab_ratio > HAZE_SLAB_MIN_RATIO and z2_p90 < 14.5 and c0_ov < 5000:
            n_removed += 1
            continue

        # Rule 5: Faint background haze next to wall/T-cell with low ch2 z-score and low C0 overlap
        if vox < HAZE_MAX_VOXELS and max_dt < HAZE_MAX_DT and z2_p90 < HAZE_MIN_Z2_P90 and c0_ov < 6000:
            if z2_p50 >= 6.4 and (t_any_touch > 0 or sig65_frac < 0.52):
                n_removed += 1
                continue

        # Rule 6: Small intra-T-cell ch2 inclusion inside green T cell (v12_MIP_ROI_23)
        if vox < TCELL_INCLUSION_MAX_VOX:
            z1_p50 = float(np.percentile(z1_sub, 50))
            if z1_p50 >= TCELL_INCLUSION_MIN_Z1_P50 and nbt >= TCELL_INCLUSION_MIN_NBT and c0_ov == 0:
                n_removed += 1
                continue

        out[m] = u

    return _relabel_sequential(out, min_size=MIN_CELL_VOXELS), n_removed


def _remerge_concentric_single_cells_c6(
    b_inst: np.ndarray,
    c0_b_inst: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Set[int], int]:
    """Re-merge over-split fragments of a single cell when a smaller fragment touches/wraps a larger one."""
    uids = [int(u) for u in np.unique(b_inst) if u > 0]
    if len(uids) <= 1:
        return b_inst, set(), 0

    out = b_inst.copy()
    merged_into: Set[int] = set()
    n_remerged = 0
    for u in sorted(uids, key=lambda x: int((out == x).sum())):
        m = out == u
        vox = int(m.sum())
        if vox == 0 or vox >= REMERGE_MAX_FRAG_VOX:
            continue
        sl = bbox_slices(m, 3, m.shape)
        m_sub = m[sl]
        out_sub = out[sl]
        dil = binary_dilation(m_sub, iterations=1) & (~m_sub)
        tot_surf = int(dil.sum())
        if tot_surf == 0:
            continue
        nbr_vals = out_sub[dil]
        best_v = 0
        best_cnt = 0
        for v in np.unique(nbr_vals):
            if v == 0 or v == u:
                continue
            cnt = int((nbr_vals == v).sum())
            if cnt > best_cnt:
                best_cnt = cnt
                best_v = int(v)

        # Also check 2-voxel dilation if both fragments belong to the same single C0 instance (e.g. v01_MIP_ROI_2 v5)
        if best_v == 0 and c0_b_inst is not None:
            dil2 = binary_dilation(m_sub, iterations=2) & (~m_sub)
            for v in np.unique(out_sub[dil2]):
                if v == 0 or v == u:
                    continue
                cnt = int((out_sub[dil2] == v).sum())
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_v = int(v)

        if best_v == 0:
            continue
        surf_frac = best_cnt / float(tot_surf)

        best_m = out == best_v
        union_mask = best_m | m
        union_vox = int(union_mask.sum())
        nz = np.argwhere(union_mask)
        span = nz.max(axis=0) - nz.min(axis=0) + 1
        union_bf = union_vox / float(max(1, int(np.prod(span))))
        if union_bf < REMERGE_MIN_UNION_BF:
            continue

        n_c0_subst = 0
        shared_single_c0 = False
        if c0_b_inst is not None:
            c0_uids = [int(cu) for cu in np.unique(c0_b_inst[union_mask]) if cu > 0]
            for cu in c0_uids:
                c0_m = c0_b_inst == cu
                c0_tot = max(1, int(c0_m.sum()))
                ov = int((union_mask & c0_m).sum())
                if ov >= C0_SEED_MIN_OVERLAP_VOX and (ov / float(c0_tot)) >= C0_SEED_MIN_OVERLAP_FRAC:
                    n_c0_subst += 1
            if n_c0_subst == 1 and len(c0_uids) == 1:
                c0_m = c0_b_inst == c0_uids[0]
                c0_tot = float(max(1, int(c0_m.sum())))
                if (int((m & c0_m).sum()) / float(vox) >= 0.35) and (int((best_m & c0_m).sum()) / c0_tot >= 0.20):
                    shared_single_c0 = True
        if n_c0_subst >= 2:
            continue

        max_allowed_union = 132000 if (n_c0_subst == 0 and surf_frac >= REMERGE_MIN_SURFACE_FRAC) else REMERGE_MAX_UNION_VOX
        if union_vox > max_allowed_union:
            continue

        min_req_surf = (
            0.02
            if (shared_single_c0 and vox < 15000)
            else (REMERGE_C0_SHARED_SURF if shared_single_c0 else REMERGE_MIN_SURFACE_FRAC)
        )
        if surf_frac < min_req_surf:
            continue

        out[m] = best_v
        merged_into.add(best_v)
        if u in merged_into:
            merged_into.discard(u)
        n_remerged += 1

    relab = np.zeros_like(out, dtype=np.uint16)
    new_remerged: Set[int] = set()
    nid = 1
    for old_u in sorted(int(x) for x in np.unique(out) if x > 0):
        m = out == old_u
        if int(m.sum()) < MIN_CELL_VOXELS:
            continue
        relab[m] = nid
        if old_u in merged_into:
            new_remerged.add(nid)
        nid += 1

    return relab, new_remerged, n_remerged


def _plane_and_valley_metrics(
    m_sub: np.ndarray,
    m1: np.ndarray,
    m2: np.ndarray,
    z2_sub: np.ndarray,
    c0_sub: Optional[np.ndarray],
    dt_parent: np.ndarray,
) -> Dict[str, Any]:
    """Compute plane-fit RMS, roughness, C0 overlap, and ch2 valley metrics on a pair of child masks."""
    vox = int(m_sub.sum())
    v1 = int(m1.sum())
    v2 = int(m2.sum())
    min_frac = min(v1, v2) / float(max(1, vox))
    dt1 = distance_transform_edt(m1)
    dt2 = distance_transform_edt(m2)
    r1 = float(dt1.max())
    r2 = float(dt2.max())
    min_dt = min(r1, r2)

    cut = (m1 & binary_dilation(m2, iterations=1)) | (m2 & binary_dilation(m1, iterations=1))
    cut_pts = np.argwhere(cut).astype(np.float32)
    if len(cut_pts) < 12 or v1 == 0 or v2 == 0:
        return {"valid": False, "min_frac": min_frac, "min_dt": min_dt}

    ctr = cut_pts.mean(axis=0)
    centered = cut_pts - ctr
    _, svals, vh = np.linalg.svd(centered, full_matrices=False)
    dist_plane = centered @ vh[2]
    rms_plane = float(np.sqrt(np.mean(dist_plane ** 2)))
    proj_area = max(
        1.0,
        float(
            np.pi
            * (2.0 * svals[0] / np.sqrt(len(cut_pts)))
            * (2.0 * svals[1] / np.sqrt(len(cut_pts)))
        ),
    )
    rough_ratio = float(len(cut_pts)) / proj_area

    p1_zyx = tuple(int(x) for x in np.unravel_index(int(np.argmax(dt1)), m_sub.shape))
    p2_zyx = tuple(int(x) for x in np.unravel_index(int(np.argmax(dt2)), m_sub.shape))
    t_vals = np.linspace(0.15, 0.85, 25)
    line_pts = np.round(np.outer(1.0 - t_vals, p1_zyx) + np.outer(t_vals, p2_zyx)).astype(int)
    line_dt_min = float(min(dt_parent[tuple(pt)] for pt in line_pts))
    line_dt_ratio = line_dt_min / max(1.0, min_dt)
    area_neck = (0.5 * len(cut_pts)) / max(1.0, float(np.pi * (min_dt ** 2)))

    n_c0_both = 0
    if c0_sub is not None:
        c0_u1 = [(int((m1 & (c0_sub == cu)).sum()), int(cu)) for cu in np.unique(c0_sub[m1]) if cu > 0]
        c0_u2 = [(int((m2 & (c0_sub == cu)).sum()), int(cu)) for cu in np.unique(c0_sub[m2]) if cu > 0]
        for ov_a, cu_a in c0_u1:
            for ov_b, cu_b in c0_u2:
                if cu_a != cu_b and ov_a >= C0_SEED_MIN_OVERLAP_VOX and ov_b >= C0_SEED_MIN_OVERLAP_VOX:
                    ov_a_m2 = int((m2 & (c0_sub == cu_a)).sum())
                    ov_b_m1 = int((m1 & (c0_sub == cu_b)).sum())
                    if ov_a > ov_a_m2 and ov_b > ov_b_m1:
                        n_c0_both = 1

    z2_sm1 = gaussian_filter(z2_sub.astype(np.float32), sigma=1.0)
    in1 = m1 & (dt1 >= 2.0) & (~cut)
    in2 = m2 & (dt2 >= 2.0) & (~cut)
    z2_c1_p50 = float(np.percentile(z2_sm1[in1 if np.any(in1) else m1], 50))
    z2_c2_p50 = float(np.percentile(z2_sm1[in2 if np.any(in2) else m2], 50))
    val_diff50 = min(z2_c1_p50, z2_c2_p50) - float(np.percentile(z2_sm1[cut], 50))

    z2_sm2 = gaussian_filter(z2_sub.astype(np.float32), sigma=2.0)
    core1 = m1 & (dt1 >= max(3.0, 0.45 * r1))
    core2 = m2 & (dt2 >= max(3.0, 0.45 * r2))
    c1_core = float(np.percentile(z2_sm2[core1 if np.any(core1) else m1], 50))
    c2_core = float(np.percentile(z2_sm2[core2 if np.any(core2) else m2], 50))
    c_min_core = min(c1_core, c2_core)
    c1_p75 = float(np.percentile(z2_sm2[m1], 75))
    c2_p75 = float(np.percentile(z2_sm2[m2], 75))
    cut_p50 = float(np.percentile(z2_sm2[cut], 50))
    cut_p25 = float(np.percentile(z2_sm2[cut], 25))
    core_diff = c_min_core - cut_p50
    p75_diff = min(c1_p75, c2_p75) - cut_p50
    p25_diff = c_min_core - cut_p25

    return {
        "valid": True,
        "v1": v1,
        "v2": v2,
        "p1": p1_zyx,
        "p2": p2_zyx,
        "r1": r1,
        "r2": r2,
        "min_frac": min_frac,
        "min_dt": min_dt,
        "rms": rms_plane,
        "rough": rough_ratio,
        "line_dt_r": line_dt_ratio,
        "area_neck": area_neck,
        "n_c0_both": n_c0_both,
        "val_diff50": val_diff50,
        "c_min_core": c_min_core,
        "core_diff": core_diff,
        "p75_diff": p75_diff,
        "p25_diff": p25_diff,
    }


def evaluate_cut_contract(
    m_sub: np.ndarray,
    z2_sub: np.ndarray,
    c0_sub: Optional[np.ndarray],
    other_b_sub: np.ndarray,
    c0_full_vols: Dict[int, int],
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Propose balanced smooth watershed split and evaluate C6 split contract (Rules 1a, 1b, 1c).

    Contract:
      (1c) Neither child is a sliver: child_vox >= 25% of parent_vox, child_vox >= 18000, child_dt >= 8.5.
      (1b) ch2 along the cut surface is clearly lower (a valley) than inside both children,
           OR C0 shows two separate instances overlapping both children.
      (1a) Cut surface is NOT near-planar (unless it coincides with a real membrane valley/neck)
           and NOT jagged/rough.
    """
    vox = int(m_sub.sum())
    markers_init, cost_init = _find_split_seeds_and_cost(m_sub, z2_sub, c0_sub, other_b_sub)
    if markers_init is None or int(markers_init.max()) < 2:
        return False, None, None, {"reject_reason": "no_seeds"}

    dt = distance_transform_edt(m_sub).astype(np.float32)
    dt_sm = gaussian_filter(dt, sigma=1.5)
    pks_dt = peak_local_max(
        dt_sm,
        min_distance=DT_PEAK_MIN_DIST,
        threshold_abs=DT_PEAK_MIN_VAL,
        labels=m_sub.astype(np.int32),
    )
    n_pks_dt = len(pks_dt)

    # C0 overlaps (>= 2000 voxels) on parent object
    c0_list: List[Tuple[int, float]] = []
    if c0_sub is not None:
        for cu in np.unique(c0_sub[m_sub]):
            if cu > 0:
                ov = int((m_sub & (c0_sub == cu)).sum())
                tot = max(1, c0_full_vols.get(int(cu), int((c0_sub == cu).sum())))
                if ov >= 2000:
                    c0_list.append((ov, round(ov / float(tot), 2)))
        c0_list.sort(key=lambda x: x[0], reverse=True)
    c0_n_2k = len(c0_list)
    c0_top_ov = c0_list[0][0] if c0_list else 0
    c0_top_frac = c0_list[0][1] if c0_list else 0.0
    c0_sec_ov = c0_list[1][0] if len(c0_list) >= 2 else 0

    # Stage 1: Initial watershed partition to locate candidate child cores and evaluate initial cut
    ws_init = watershed(cost_init, markers_init, mask=m_sub)
    m1_init = binary_fill_holes(ws_init == 1) & m_sub
    m2_init = binary_fill_holes(ws_init == 2) & m_sub & (~m1_init)
    init_m = _plane_and_valley_metrics(m_sub, m1_init, m2_init, z2_sub, c0_sub, dt)
    if not init_m.get("valid", False):
        return False, None, None, {"reject_reason": "invalid_init_cut"}

    # Stage 2: Re-cut from candidate child core peaks using balanced smooth valley + DT-neck cost
    p1, p2 = init_m["p1"], init_m["p2"]
    r1, r2 = float(dt[p1]), float(dt[p2])
    zz, yy, xx = np.indices(m_sub.shape, dtype=np.float32)
    dist1 = np.sqrt((zz - p1[0]) ** 2 + (yy - p1[1]) ** 2 + (xx - p1[2]) ** 2)
    dist2 = np.sqrt((zz - p2[0]) ** 2 + (yy - p2[1]) ** 2 + (xx - p2[2]) ** 2)
    r_eff1 = (0.75 * r1 + 0.25 * r2) ** 0.65
    r_eff2 = (0.25 * r1 + 0.75 * r2) ** 0.65
    d1_n = dist1 / max(6.0, r_eff1)
    d2_n = dist2 / max(6.0, r_eff2)

    z2_sm2 = gaussian_filter(z2_sub.astype(np.float32), sigma=2.0)
    p10, p90 = float(np.percentile(z2_sm2[m_sub], 10)), float(np.percentile(z2_sm2[m_sub], 90))
    z2_n = np.clip((z2_sm2 - p10) / max(1.0, p90 - p10), 0.0, 1.2)
    cost_bal = 9.0 * np.minimum(d1_n, d2_n) - 1.0 * dt_sm - 2.5 * z2_n

    markers_bal = np.zeros(m_sub.shape, dtype=np.int32)
    markers_bal[p1] = 1
    markers_bal[p2] = 2
    ws_bal = watershed(cost_bal, markers_bal, mask=m_sub)
    m1 = binary_fill_holes(ws_bal == 1) & m_sub
    m2 = binary_fill_holes(ws_bal == 2) & m_sub & (~m1)
    bal_m = _plane_and_valley_metrics(m_sub, m1, m2, z2_sub, c0_sub, dt)
    if not bal_m.get("valid", False):
        return False, None, None, {"reject_reason": "invalid_bal_cut"}

    min_frac_bal = bal_m["min_frac"]
    min_dt_bal = bal_m["min_dt"]
    rms_bal = bal_m["rms"]
    rough_bal = bal_m["rough"]
    line_dt_r_bal = bal_m["line_dt_r"]
    area_neck_bal = bal_m["area_neck"]
    c_min_core_bal = bal_m["c_min_core"]
    core_diff_bal = bal_m["core_diff"]
    p75_diff_bal = bal_m["p75_diff"]
    p25_diff_bal = bal_m["p25_diff"]

    rms_plane = init_m["rms"]
    val_diff50 = init_m["val_diff50"]
    n_c0_both = max(int(init_m["n_c0_both"]), int(bal_m["n_c0_both"]))

    info: Dict[str, Any] = {
        "vox": vox,
        "v1": int(m1.sum()),
        "v2": int(m2.sum()),
        "min_frac": round(min_frac_bal, 4),
        "min_dt": round(min_dt_bal, 2),
        "rms_bal": round(rms_bal, 3),
        "rms_init": round(rms_plane, 3),
        "rough_bal": round(rough_bal, 3),
        "line_dt_r": round(line_dt_r_bal, 3),
        "area_neck": round(area_neck_bal, 3),
        "c_min_core": round(c_min_core_bal, 2),
        "core_diff": round(core_diff_bal, 2),
        "p75_diff": round(p75_diff_bal, 2),
        "p25_diff": round(p25_diff_bal, 2),
        "n_pks_dt": n_pks_dt,
        "n_c0_both": n_c0_both,
        "c0_n_2k": c0_n_2k,
        "c0_top_ov": c0_top_ov,
        "c0_top_frac": c0_top_frac,
        "c0_sec_ov": c0_sec_ov,
    }

    # Rule 1c: Sliver check
    if min_frac_bal < SPLIT_MIN_CHILD_FRAC or min(int(m1.sum()), int(m2.sum())) < SPLIT_MIN_CHILD_VOX or min_dt_bal < SPLIT_MIN_CHILD_DT:
        info["reject_reason"] = "1c_sliver"
        return False, None, None, info

    # Contained C0 two-instance overlap evidence
    has_clean_c0_pair = (
        vox >= 165000
        and c0_n_2k == 2
        and c0_top_ov >= 28000
        and c0_top_frac >= 0.98
        and c0_sec_ov >= C0_SEED_MIN_OVERLAP_VOX
        and (n_c0_both == 1 or rms_bal <= 2.25)
    ) or (
        n_c0_both == 1
        and c0_top_ov >= 28000
        and c0_sec_ov >= C0_SEED_MIN_OVERLAP_VOX
        and val_diff50 >= 0.0
    )

    # Rule 1b (faint single cell): low-signal cells (core z2 < 10.8) only split when backed by clean C0 pair
    if c_min_core_bal < FAINT_MIN_CORE_Z2 and not has_clean_c0_pair:
        info["reject_reason"] = "1b_faint_single_cell"
        return False, None, None, info

    # Rule 1b (contained single C0 cell): if a single C0 cell is >=96% contained inside m_sub and no 2nd C0 cell exists
    if c0_n_2k <= 1 and c0_top_frac >= 0.96:
        if not (vox >= 185000 and area_neck_bal <= 3.50 and (line_dt_r_bal <= NECK_MAX_LINE_DT_RATIO or c0_top_ov >= 95000)):
            info["reject_reason"] = "1b_single_c0_contained"
            return False, None, None, info

    # Rule 1a: Planar / jagged surface check
    if not has_clean_c0_pair:
        if rms_bal > JAGGED_MAX_RMS or rms_plane > JAGGED_MAX_INIT_RMS or rough_bal > JAGGED_MAX_ROUGH_RATIO:
            info["reject_reason"] = "1a_jagged"
            return False, None, None, info
        if rms_bal < PLANAR_MIN_RMS and not (line_dt_r_bal <= NECK_MAX_LINE_DT_RATIO and core_diff_bal >= 1.20):
            info["reject_reason"] = "1a_near_planar"
            return False, None, None, info
    else:
        if rms_bal > JAGGED_MAX_C0_RMS or rough_bal > 2.05:
            info["reject_reason"] = "1a_jagged_c0"
            return False, None, None, info

    # Rule 1b: Valley or C0 two-instance overlap + 3D neck/peak requirement
    has_strong_valley = (
        (core_diff_bal >= VALLEY_MIN_CORE_DIFF and p75_diff_bal >= VALLEY_MIN_P75_DIFF and p25_diff_bal >= VALLEY_MIN_P25_DIFF)
        or (n_pks_dt >= 2 and core_diff_bal >= 0.30 and p25_diff_bal >= VALLEY_MIN_P25_DIFF)
        or (vox >= 190000 and c0_top_ov >= 95000 and val_diff50 > 0.0 and p75_diff_bal >= 2.0 and p25_diff_bal >= 4.5)
    )
    if not (has_clean_c0_pair or has_strong_valley):
        info["reject_reason"] = "1b_no_valley_or_c0"
        return False, None, None, info

    # For 1-DT-peak objects without clean C0 pair, require 3D neck (line_dt_r <= 0.85) or deep valley (>=4.5 z-units)
    if n_pks_dt <= 1 and not has_clean_c0_pair:
        if not (
            line_dt_r_bal <= NECK_MAX_LINE_DT_RATIO
            or (core_diff_bal >= 4.50 and c0_top_frac >= 0.88)
            or (vox >= 190000 and c0_top_ov >= 95000 and val_diff50 > 0.0)
        ):
            info["reject_reason"] = "1b_single_peak_no_neck"
            return False, None, None, info

    # If accepted via C0 pair but rms_bal > 2.25, smooth the cut surface with higher distance weight
    if has_clean_c0_pair and rms_bal > 2.25:
        cost_smooth = 14.0 * np.minimum(d1_n, d2_n) - 1.2 * dt_sm - 1.0 * z2_n
        ws_sm = watershed(cost_smooth, markers_bal, mask=m_sub)
        m1_sm = binary_fill_holes(ws_sm == 1) & m_sub
        m2_sm = binary_fill_holes(ws_sm == 2) & m_sub & (~m1_sm)
        sm_m = _plane_and_valley_metrics(m_sub, m1_sm, m2_sm, z2_sub, c0_sub, dt)
        if sm_m.get("valid", False) and sm_m["min_frac"] >= SPLIT_MIN_CHILD_FRAC and sm_m["min_dt"] >= SPLIT_MIN_CHILD_DT:
            m1, m2 = m1_sm, m2_sm
            info["rms_bal"] = round(sm_m["rms"], 3)
            info["min_frac"] = round(sm_m["min_frac"], 4)

    info["reject_reason"] = "accepted"
    return True, m1, m2, info


def split_merged_targets_with_stats(
    raw: np.ndarray,
    t_inst: np.ndarray,
    b_inst: np.ndarray,
    c0_b_inst: Optional[np.ndarray] = None,
    tracked_t_id: int = 0,
    tracked_b_id: int = 0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run C6 per-visit target cleaning and splitting and return (b_inst_new, stats_dict)."""
    c4_uids = [int(u) for u in np.unique(b_inst) if u > 0]
    c4_n = len(c4_uids)

    _, z1, z2, sytox_mask = _channel_zscores(raw)

    # Step 1: Remove non-cell objects while preserving dead/SYTOX and real dim cells
    b_clean, n_removed = remove_non_cell_objects_c6(
        b_inst,
        t_inst,
        raw,
        z1,
        z2,
        sytox_mask,
        c0_b_inst=c0_b_inst,
        tracked_t_id=tracked_t_id,
        tracked_b_id=tracked_b_id,
    )

    # Step 2: Re-merge over-split concentric fragments of single cells
    b_clean, remerged_uids, n_remerged = _remerge_concentric_single_cells_c6(b_clean, c0_b_inst=c0_b_inst)

    uids = [int(u) for u in np.unique(b_clean) if u > 0]
    if not uids:
        stats = {
            "c4_n": c4_n,
            "c6_n": 0,
            "n_removed": n_removed,
            "n_remerged": n_remerged,
            "n_split_objs": 0,
            "n_split_children": 0,
            "split_events": [],
        }
        return np.zeros_like(b_inst, dtype=np.uint16), stats

    c0_full_vols: Dict[int, int] = {}
    if c0_b_inst is not None:
        for cu in np.unique(c0_b_inst):
            if cu > 0:
                c0_full_vols[int(cu)] = int((c0_b_inst == cu).sum())

    out = np.zeros_like(b_inst, dtype=np.uint16)
    next_id = 1
    n_split_objs = 0
    n_split_children = 0
    split_events: List[Dict[str, Any]] = []

    for u in uids:
        m = b_clean == u
        vox = int(m.sum())
        if vox < SPLIT_MIN_OBJ_VOXELS or (u in remerged_uids):
            out[m] = next_id
            next_id += 1
            continue

        sl = bbox_slices(m, 2, m.shape)
        m_sub = m[sl]
        z2_sub = z2[sl]
        c0_sub = c0_b_inst[sl] if c0_b_inst is not None else None
        other_b_sub = (b_clean[sl] > 0) & (~m_sub)

        ok, m1, m2, info = evaluate_cut_contract(m_sub, z2_sub, c0_sub, other_b_sub, c0_full_vols)
        if not ok or m1 is None or m2 is None:
            out[m] = next_id
            next_id += 1
            continue
        if int(m2.sum()) > int(m1.sum()):
            m1, m2 = m2, m1
        out_sub = out[sl]
        id1, id2 = next_id, next_id + 1
        out_sub[m1] = id1
        out_sub[m2] = id2
        next_id += 2
        n_split_objs += 1
        n_split_children += 2
        split_events.append(
            {
                "child_ids": (id1, id2),
                "parent_vox": vox,
                "v1": int(m1.sum()),
                "v2": int(m2.sum()),
                "info": info,
            }
        )

    b_final = _relabel_sequential(out, min_size=MIN_CELL_VOXELS)
    c6_n = int(len([u for u in np.unique(b_final) if u > 0]))
    stats = {
        "c4_n": c4_n,
        "c6_n": c6_n,
        "n_removed": n_removed,
        "n_remerged": n_remerged,
        "n_split_objs": n_split_objs,
        "n_split_children": n_split_children,
        "split_events": split_events,
    }
    return b_final, stats


def enforce_temporal_split_consistency(
    b_list: List[np.ndarray],
    stats_list: List[Dict[str, Any]],
) -> int:
    """Rule 4: Reject splits that cause a >35% single-visit target volume drop when unsplit parent matches neighbors in time."""
    n_reverted = 0
    n_visits = len(b_list)
    if n_visits < 2:
        return 0

    for idx in range(n_visits):
        st = stats_list[idx]
        if st.get("n_split_objs", 0) == 0:
            continue
        b_curr = b_list[idx]
        for ev in list(st.get("split_events", [])):
            id1, id2 = ev["child_ids"]
            m1 = b_curr == id1
            m2 = b_curr == id2
            v1 = int(m1.sum())
            v2 = int(m2.sum())
            if v1 == 0 or v2 == 0:
                continue
            m_union = m1 | m2
            v_parent = int(m_union.sum())

            info = ev.get("info", {})
            if v_parent >= 150000 or info.get("n_c0_both", 0) == 1 or info.get("core_diff", 0.0) >= 4.0:
                continue

            # Compare against best-overlapping instance in BOTH t-1 and t+1
            valid_nbrs = 0
            matching_unsplit_nbrs = 0
            for nbr_idx in (idx - 1, idx + 1):
                if nbr_idx < 0 or nbr_idx >= n_visits:
                    continue
                valid_nbrs += 1
                if stats_list[nbr_idx].get("n_split_objs", 0) > 0:
                    continue
                b_nbr = b_list[nbr_idx]
                best_u = 0
                best_ov = 0
                for u_n in np.unique(b_nbr[m_union]):
                    if u_n == 0:
                        continue
                    ov = int((m_union & (b_nbr == u_n)).sum())
                    if ov > best_ov:
                        best_ov = ov
                        best_u = int(u_n)
                if best_u > 0 and best_ov >= 0.35 * v_parent:
                    m_nbr = b_nbr == best_u
                    v_nbr = int(m_nbr.sum())
                    ov1 = int((m1 & m_nbr).sum())
                    ov2 = int((m2 & m_nbr).sum())
                    v_matched_child = v1 if ov1 >= ov2 else v2
                    parent_rel_diff = abs(v_parent - v_nbr) / float(max(1, v_nbr))
                    if v_matched_child < (1.0 - TEMPORAL_MAX_DROP_FRAC) * v_nbr and parent_rel_diff <= 0.25:
                        matching_unsplit_nbrs += 1

            if valid_nbrs > 0 and matching_unsplit_nbrs == valid_nbrs:
                b_curr[m2] = id1
                st["n_split_objs"] = max(0, st["n_split_objs"] - 1)
                st["n_split_children"] = max(0, st["n_split_children"] - 2)
                n_reverted += 1

        if n_reverted > 0:
            b_list[idx] = _relabel_sequential(b_curr, min_size=MIN_CELL_VOXELS)
            st["c6_n"] = int(len([u for u in np.unique(b_list[idx]) if u > 0]))

    return n_reverted


def split_merged_targets(
    raw: np.ndarray,
    t_inst: np.ndarray,
    b_inst: np.ndarray,
    c0_b_inst: Optional[np.ndarray] = None,
    tracked_t_id: int = 0,
    tracked_b_id: int = 0,
) -> np.ndarray:
    """Split touching tumour cells merged into one b_inst object and remove non-cell objects."""
    b_final, _ = split_merged_targets_with_stats(
        raw,
        t_inst,
        b_inst,
        c0_b_inst=c0_b_inst,
        tracked_t_id=tracked_t_id,
        tracked_b_id=tracked_b_id,
    )
    return b_final


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate C6 split & cleaned target masks for all 20 pilot chambers.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if C6 .npz already exists.")
    parser.add_argument("--chamber", type=str, default=None, help="Optional filter e.g. v07_MIP_ROI_26.")
    args = parser.parse_args()

    out_root = REPO_ROOT / "data" / "pilot_seg"
    from src.seg.track5 import ALL_20_CHAMBERS, load_chamber_visits

    all_chambers = list(ALL_20_CHAMBERS)
    if args.chamber:
        all_chambers = [c for c in all_chambers if f"{c[0]}_{c[2]}" == args.chamber or c[1] == args.chamber]

    sum_csv = out_root / "c6_split_summary.csv"
    existing_summary: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if sum_csv.exists() and not args.overwrite:
        old_df = pd.read_csv(sum_csv)
        for _, r in old_df.iterrows():
            existing_summary[(str(r["experiment"]), str(r["roi"]), f"{int(r['stem']):05d}")] = dict(r)

    rows: List[Dict[str, Any]] = []
    t_global0 = time.perf_counter()

    for ch_idx, (short_exp, exp, roi) in enumerate(all_chambers, start=1):
        visit_rows = load_chamber_visits(exp, roi)
        ch_t0 = time.perf_counter()

        # Check if all visits for this chamber already exist when not overwriting
        all_exist = all(
            (out_root / "C6" / exp / roi / f"{item['stem']}.npz").exists()
            for item in visit_rows
        )
        if all_exist and not args.overwrite:
            for item in visit_rows:
                visit_num = int(item["visit"])
                stem = str(item["stem"])
                key = (exp, roi, stem)
                if key in existing_summary:
                    rows.append(existing_summary[key])
                    continue
                c4_data = np.load(out_root / "C4" / exp / roi / f"{stem}.npz")
                c6_data = np.load(out_root / "C6" / exp / roi / f"{stem}.npz")
                c4_n = int(len([u for u in np.unique(c4_data["b_inst"]) if u > 0]))
                c6_n = int(len([u for u in np.unique(c6_data["b_inst"]) if u > 0]))
                rows.append(
                    {
                        "short_exp": short_exp,
                        "experiment": exp,
                        "roi": roi,
                        "visit_num": visit_num,
                        "stem": stem,
                        "c4_n": c4_n,
                        "c6_n": c6_n,
                        "n_removed": max(0, c4_n - c6_n),
                        "n_remerged": 0,
                        "n_split_objs": max(0, c6_n - c4_n),
                        "n_split_children": 2 * max(0, c6_n - c4_n),
                        "runtime_s": round(float(c6_data["runtime_s"]), 4),
                    }
                )
            continue

        b_new_list: List[np.ndarray] = []
        t_inst_list: List[np.ndarray] = []
        tr_t_list: List[int] = []
        tr_b_list: List[int] = []
        dt_s_list: List[float] = []
        st_list: List[Dict[str, Any]] = []

        for item in visit_rows:
            stem = str(item["stem"])
            h5_path = REPO_ROOT / str(item["rel_path"])
            c4_path = out_root / "C4" / exp / roi / f"{stem}.npz"
            c0_path = out_root / "C0" / exp / roi / f"{stem}.npz"

            c4_data = np.load(c4_path)
            t_inst = c4_data["t_inst"].astype(np.uint16)
            b_inst = c4_data["b_inst"].astype(np.uint16)
            tracked_t_id = int(c4_data["tracked_t_id"])
            tracked_b_id = int(c4_data["tracked_b_id"])
            c0_b_inst = np.load(c0_path)["b_inst"].astype(np.uint16) if c0_path.exists() else None

            with h5py.File(h5_path, "r") as f:
                raw = f["raw"][:]
            t0 = time.perf_counter()
            b_new, st = split_merged_targets_with_stats(
                raw,
                t_inst,
                b_inst,
                c0_b_inst=c0_b_inst,
                tracked_t_id=tracked_t_id,
                tracked_b_id=tracked_b_id,
            )
            dt_s = time.perf_counter() - t0
            del raw

            b_new_list.append(b_new)
            t_inst_list.append(t_inst)
            tr_t_list.append(tracked_t_id)
            tr_b_list.append(tracked_b_id)
            dt_s_list.append(dt_s)
            st_list.append(st)
            gc.collect()

        # Step 4: Chamber-level temporal consistency check on splits
        n_rev = enforce_temporal_split_consistency(b_new_list, st_list)

        for item, t_inst, b_new, tr_t, tr_b, dt_s, st in zip(
            visit_rows, t_inst_list, b_new_list, tr_t_list, tr_b_list, dt_s_list, st_list
        ):
            visit_num = int(item["visit"])
            stem = str(item["stem"])
            c6_path = out_root / "C6" / exp / roi / f"{stem}.npz"
            save_candidate_npz(c6_path, t_inst, b_new, tr_t, tr_b, dt_s)
            rows.append(
                {
                    "short_exp": short_exp,
                    "experiment": exp,
                    "roi": roi,
                    "visit_num": visit_num,
                    "stem": stem,
                    "c4_n": st["c4_n"],
                    "c6_n": st["c6_n"],
                    "n_removed": st["n_removed"],
                    "n_remerged": st["n_remerged"],
                    "n_split_objs": st["n_split_objs"],
                    "n_split_children": st["n_split_children"],
                    "runtime_s": round(dt_s, 4),
                }
            )

        ch_dt = time.perf_counter() - ch_t0
        ch_rows = [r for r in rows if r["experiment"] == exp and r["roi"] == roi]
        sp_visits = [r["visit_num"] for r in ch_rows if r["n_split_objs"] > 0]
        rm_visits = [r["visit_num"] for r in ch_rows if r["n_removed"] > 0]
        print(
            f"[{ch_idx:02d}/{len(all_chambers):02d}] {short_exp}_{roi}: {len(ch_rows)} visits in {ch_dt:.1f}s "
            f"(rev={n_rev}) | split_visits={sp_visits} | removed_visits={rm_visits}",
            flush=True,
        )

    if not args.chamber:
        df_sum = pd.DataFrame(rows)
        df_sum.to_csv(sum_csv, index=False)
        print(f"Done in {time.perf_counter() - t_global0:.1f}s. Summary saved to {sum_csv}", flush=True)


if __name__ == "__main__":
    main()
