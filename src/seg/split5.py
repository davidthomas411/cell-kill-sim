"""C5 post-processing of C4 target masks: split merged tumour cells and remove non-cell objects.

Provides:
  split_merged_targets(raw, t_inst, b_inst, c0_b_inst=None) -> b_inst_new
  main() -> processes all 20 chambers (PILOT_CHAMBERS + PILOT4_CHAMBERS), sweep 1 of every visit,
            saving data/pilot_seg/C5/<experiment>/<roi>/<stem>.npz.
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
    binary_erosion,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    label,
)
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mask_qc import bbox_slices
from src.reproduce_contact_area import STRUCT26
from src.seg.candidates import (
    PILOT4_CHAMBERS,
    PILOT_CHAMBERS,
    _relabel_sequential,
    save_candidate_npz,
)

# ---------------------------------------------------------------------------
# Documented thresholds for non-cell removal and merged-target splitting
# ---------------------------------------------------------------------------

# 1. Non-cell removal thresholds
MIN_CELL_VOXELS = 14000          # Minimum 3D volume (voxels, ~106 um^3) for a valid tumour cell
MIN_CELL_MAX_DT = 7.5            # Minimum 3D inscribed radius (voxels, ~1.47 um); rejects flat wall slabs/beads
HAZE_MAX_VOXELS = 26000          # Upper volume bound for faint background haze blobs (e.g. v10_MIP_ROI_28)
HAZE_MAX_DT = 9.5                # Max inscribed radius for faint haze blobs
HAZE_MIN_Z2_P90 = 14.5           # Minimum 90th-percentile ch2 z-score required for small marginal blobs
HAZE_MIN_SIG65_FRAC = 0.65       # Minimum fraction of voxels above z2 > 6.5 required for small marginal blobs
TCELL_INCLUSION_MAX_VOX = 29000  # Max volume for intra-T-cell ch2 inclusion / dividing green cell blob (v12_MIP_ROI_23)
TCELL_INCLUSION_MIN_NBT = 0.38   # Surface contact fraction with T cell above which a small ch1-bright blob is rejected
TCELL_INCLUSION_MIN_Z1_P50 = 4.2 # Median ch1 z-score inside object indicating green T-cell body rather than tumour cell

# 2. Concentric / over-split single-cell re-merge thresholds
REMERGE_MIN_SURFACE_FRAC = 0.45  # Fraction of smaller fragment's outer surface touching larger fragment
REMERGE_C0_SHARED_SURF = 0.25    # Surface contact fraction when both fragments share the same single C0 cell
REMERGE_MAX_FRAG_VOX = 45000     # Max volume of smaller fragment eligible for concentric re-merge
REMERGE_MAX_UNION_VOX = 115000   # Max combined volume for re-merging fragments of a single cell
REMERGE_MIN_UNION_BF = 0.18      # Minimum bounding-box fill ratio of combined single cell

# 3. Guard and splitting thresholds
SPLIT_MIN_OBJ_VOXELS = 75000     # Minimum object volume eligible for splitting into >=2 tumour cells
GUARD_ROUND_MIN_BF = 0.365       # Bounding-box fill ratio above which a compact 1-peak cell is kept whole
GUARD_ROUND_MAX_SPAN = 115       # Max bounding-box axis span for the compact round-cell guard
GUARD_SINGLE_MAX_VOX = 100500    # Objects below this volume with 1 DT peak, bf >= 0.185, and <=1 C0 match stay whole
GUARD_SINGLE_MIN_BF = 0.185      # Minimum bbox fill ratio for the moderate-volume single-cell guard
GUARD_ISOLATED_MAX_VOX = 165000  # Max volume for isolated (nbB==0) single-C0 bright-cell guard (e.g. v04_MIP_ROI_24)
GUARD_ISOLATED_MIN_C0_OV = 32000 # Minimum overlap with single C0 cell for isolated bright-cell guard
GUARD_ISOLATED_MIN_Z2 = 18.0     # Minimum median ch2 z-score for isolated bright-cell guard
GUARD_ISOLATED_MAX_PK_RATIO = 0.80  # Max secondary/primary DT peak ratio for isolated single-cell guard

C0_SEED_MIN_OVERLAP_VOX = 4000   # Minimum voxel overlap for a C0 instance to count as multi-cell evidence
C0_SEED_MIN_OVERLAP_FRAC = 0.05  # Minimum fraction of C0 instance volume overlapping the C4 object
DT_PEAK_MIN_DIST = 18            # Minimum 3D distance (voxels) between interior seeds
DT_PEAK_MIN_VAL = 6.5            # Minimum smoothed DT value (voxels) for a valid cell-core seed
DT_HMAX_H = 1.4                  # h-maxima prominence (voxels) on smoothed distance transform
SEED_FAR_RADIUS_MULT = 1.25      # Secondary seed must lie >= 1.25 * r1 outside primary core sphere
SPLIT_MIN_CHILD_VOX = 16000      # Minimum volume (voxels) for each child instance after watershed split
SPLIT_MIN_CHILD_DT = 7.5         # Minimum max-DT (voxels) for each child instance after watershed split


def _channel_zscores(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Gaussian-smoothed (sigma=1.0) robust z-score volumes for ch0, ch1, ch2 and SYTOX mask."""
    z_vols: List[np.ndarray] = []
    for ch in range(3):
        s = gaussian_filter(raw[ch].astype(np.float32), sigma=1.0)
        sub = s[::2, ::2, ::2]
        med = float(np.median(sub))
        mad = float(np.median(np.abs(sub - med)))
        noise = max(1.0, 1.4826 * mad)
        z_vols.append((s - med) / noise)
    d0_sm3 = gaussian_filter(raw[0].astype(np.float32), sigma=3.0)
    sytox_mask = (d0_sm3 > 400.0) | (z_vols[0] > 5.0)
    return z_vols[0], z_vols[1], z_vols[2], sytox_mask


def remove_non_cell_objects(
    b_inst: np.ndarray,
    t_inst: np.ndarray,
    z1: np.ndarray,
    z2: np.ndarray,
    sytox_mask: np.ndarray,
    c0_b_inst: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """Remove non-cell objects (small wall beads, faint background haze, intra-T-cell inclusions).

    Returns (cleaned_b_inst, n_removed).
    """
    uids = [int(u) for u in np.unique(b_inst) if u > 0]
    if not uids:
        return np.zeros_like(b_inst, dtype=np.uint16), 0

    out = np.zeros_like(b_inst, dtype=np.uint16)
    n_removed = 0
    for u in uids:
        m = b_inst == u
        vox = int(m.sum())
        if vox < MIN_CELL_VOXELS:
            n_removed += 1
            continue

        sl = bbox_slices(m, 2, m.shape)
        m_sub = m[sl]
        dt = distance_transform_edt(m_sub)
        max_dt = float(dt.max())

        # Rule 1: Thin slabs / wall beads with small inscribed radius (e.g. v12_MIP_ROI_32, v02_MIP_ROI_3 v12)
        if max_dt < MIN_CELL_MAX_DT:
            n_removed += 1
            continue

        z2_sub = z2[sl][m_sub]
        z1_sub = z1[sl][m_sub]
        sytox_vox = int(sytox_mask[sl][m_sub].sum())
        is_sytox_dead = (sytox_vox >= 1500) or (sytox_vox / float(vox) >= 0.15)

        z2_p90 = float(np.percentile(z2_sub, 90))
        sig65_frac = float((z2_sub > 6.5).mean())

        # Rule 2: Faint background haze (e.g. v10_MIP_ROI_28 lower-left)
        if (not is_sytox_dead) and vox < HAZE_MAX_VOXELS and max_dt < HAZE_MAX_DT:
            if z2_p90 < HAZE_MIN_Z2_P90 or sig65_frac < HAZE_MIN_SIG65_FRAC:
                n_removed += 1
                continue

        # Rule 3: Small ch2 inclusion inside/touching a green T cell (e.g. v12_MIP_ROI_23)
        if vox < TCELL_INCLUSION_MAX_VOX:
            z1_p50 = float(np.percentile(z1_sub, 50))
            t_sub = t_inst[sl] > 0
            dil = binary_dilation(m_sub, iterations=1) & (~m_sub)
            nbt = int(t_sub[dil].sum()) / float(max(1, int(dil.sum())))
            c0_ov = int((m_sub & (c0_b_inst[sl] > 0)).sum()) if c0_b_inst is not None else 0
            if z1_p50 >= TCELL_INCLUSION_MIN_Z1_P50 and (nbt >= TCELL_INCLUSION_MIN_NBT or c0_ov == 0):
                n_removed += 1
                continue

        out[m] = u

    return _relabel_sequential(out, min_size=MIN_CELL_VOXELS), n_removed


def _remerge_concentric_single_cells(
    b_inst: np.ndarray,
    c0_b_inst: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Set[int], int]:
    """Re-merge over-split fragments of a single cell when a smaller fragment is wrapped by a larger one.

    Returns (b_merged, remerged_uids, n_remerged).
    """
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
        sl = bbox_slices(m, 2, m.shape)
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
        if best_v == 0:
            continue
        surf_frac = best_cnt / float(tot_surf)

        best_m = out == best_v
        union_mask = best_m | m
        union_vox = int(union_mask.sum())
        if union_vox > REMERGE_MAX_UNION_VOX:
            continue
        nz = np.argwhere(union_mask)
        span = nz.max(axis=0) - nz.min(axis=0) + 1
        union_bf = union_vox / float(max(1, int(np.prod(span))))
        if union_bf < REMERGE_MIN_UNION_BF:
            continue

        # Check C0 overlap structure on the union
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
                if (int((m & c0_m).sum()) / c0_tot >= 0.20) and (int((best_m & c0_m).sum()) / c0_tot >= 0.20):
                    shared_single_c0 = True
        if n_c0_subst >= 2:
            continue

        min_req_surf = REMERGE_C0_SHARED_SURF if shared_single_c0 else REMERGE_MIN_SURFACE_FRAC
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


def _find_split_seeds_and_cost(
    m_sub: np.ndarray,
    z2_sub: np.ndarray,
    c0_sub: Optional[np.ndarray],
    other_b_sub: np.ndarray,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """Find watershed seeds and cost volume inside a single candidate merged target mask m_sub.

    Evidence order:
      1. ch2 membrane signal between cells (internal membrane ridge or dim valley inside m_sub)
         combined with 3D distance transform h-maxima / local maxima seeds outside the primary sphere.
      2. Lab C0 instances when >=2 C0 target instances each overlap m_sub substantially.
    """
    vox = int(m_sub.sum())
    nz = np.argwhere(m_sub)
    span = nz.max(axis=0) - nz.min(axis=0) + 1
    max_span = int(span.max())
    bbox_fill = vox / float(max(1, int(np.prod(span))))

    dt = distance_transform_edt(m_sub).astype(np.float32)
    dt_sm = gaussian_filter(dt, sigma=1.5)

    # Primary DT peaks (Evidence 1a)
    pks_dt = peak_local_max(
        dt_sm,
        min_distance=DT_PEAK_MIN_DIST,
        threshold_abs=DT_PEAK_MIN_VAL,
        labels=m_sub.astype(np.int32),
    )

    # Guard 1: compact, round single-peak cells (bf >= 0.365, max_span <= 115) never split
    if bbox_fill >= GUARD_ROUND_MIN_BF and max_span <= GUARD_ROUND_MAX_SPAN and len(pks_dt) <= 1:
        return None, dt_sm

    # Count C0 instances with substantial overlap on m_sub (Evidence 2)
    c0_seed_masks: List[Tuple[int, np.ndarray]] = []
    c0_max_ov = 0
    if c0_sub is not None:
        for cu in np.unique(c0_sub):
            if cu == 0:
                continue
            cm = c0_sub == cu
            inter = m_sub & cm
            ov = int(inter.sum())
            if ov > c0_max_ov:
                c0_max_ov = ov
            c0_full_vox = int(cm.sum())
            if ov >= C0_SEED_MIN_OVERLAP_VOX and (ov / float(max(1, c0_full_vox))) >= C0_SEED_MIN_OVERLAP_FRAC:
                c0_seed_masks.append((ov, inter))
    c0_seed_masks.sort(key=lambda x: x[0], reverse=True)
    has_c0_multi = len(c0_seed_masks) >= 2

    # Guard 2: moderate-volume single-peak cell without multi-C0 overlap stays whole
    if not has_c0_multi:
        if vox < GUARD_SINGLE_MAX_VOX and bbox_fill >= GUARD_SINGLE_MIN_BF and len(pks_dt) <= 1:
            return None, dt_sm

    # Guard 3: isolated (no other b_inst in chamber) bright single-C0 cell (e.g. v04_MIP_ROI_24 v10/v18)
    z2_p50 = float(np.percentile(z2_sub[m_sub], 50))
    if (not np.any(other_b_sub)) and (not has_c0_multi) and c0_max_ov >= GUARD_ISOLATED_MIN_C0_OV and z2_p50 >= GUARD_ISOLATED_MIN_Z2:
        if vox < GUARD_ISOLATED_MAX_VOX:
            if len(pks_dt) <= 1:
                return None, dt_sm
            pks_vals = sorted([float(dt_sm[tuple(p)]) for p in pks_dt], reverse=True)
            if pks_vals[1] / max(1.0, pks_vals[0]) < GUARD_ISOLATED_MAX_PK_RATIO:
                return None, dt_sm

    # Build internal ch2 membrane ridge & valley evidence inside m_sub
    z2_sm = gaussian_filter(z2_sub.astype(np.float32), sigma=1.2)
    inner = m_sub & (dt > 3.5)
    if np.any(inner):
        p50_in = float(np.percentile(z2_sm[inner], 50))
        p95_in = float(np.percentile(z2_sm[inner], 95))
        denom = max(1.0, p95_in - p50_in)
        mem_ridge = np.clip((z2_sm - p50_in) / denom, 0.0, 1.5) * np.clip((dt - 2.5) / 2.5, 0.0, 1.0)
    else:
        mem_ridge = np.zeros_like(dt_sm)

    body_sm = gaussian_filter(z2_sub.astype(np.float32), sigma=4.0) * np.clip(dt / 6.0, 0.0, 1.0)
    if np.any(m_sub):
        b_max = max(1.0, float(np.percentile(body_sm[m_sub], 95)))
        body_norm = np.clip(body_sm / b_max, 0.0, 1.2)
    else:
        body_norm = np.zeros_like(dt_sm)

    # Primary seed p1 at deepest interior core of m_sub
    p1 = tuple(int(x) for x in np.unravel_index(int(np.argmax(dt_sm * m_sub)), m_sub.shape))
    r1 = float(dt[p1])
    if r1 < DT_PEAK_MIN_VAL:
        return None, dt_sm

    zz, yy, xx = np.indices(m_sub.shape, dtype=np.float32)
    dist1 = np.sqrt((zz - p1[0]) ** 2 + (yy - p1[1]) ** 2 + (xx - p1[2]) ** 2)

    # Exterior region outside Cell 1's core sphere where a second cell core can reside
    far_mask = m_sub & (dist1 >= max(22.0, SEED_FAR_RADIUS_MULT * r1))
    if int(far_mask.sum()) < SPLIT_MIN_CHILD_VOX:
        far_mask = m_sub & (dist1 >= max(18.0, 1.15 * r1))
    if int(far_mask.sum()) < SPLIT_MIN_CHILD_VOX or float((dt * far_mask).max()) < DT_PEAK_MIN_VAL:
        return None, dt_sm

    # Evidence 1 & 2: Gather candidate secondary seeds inside far_mask from DT peaks,
    # membrane-carved DT peaks, h-maxima, and C0/interior core maxima, then pick the deepest interior core
    cand_p2_list: List[Tuple[int, int, int]] = []
    core_score = dt_sm + 2.0 * body_norm

    # 1a. Primary DT peaks inside far_mask
    for p in pks_dt:
        pt = tuple(int(x) for x in p)
        if far_mask[pt] and dt[pt] >= DT_PEAK_MIN_VAL:
            cand_p2_list.append(pt)

    # 1b. Membrane-ridge-carved DT peaks inside far_mask
    if np.any(inner):
        ridge_thr = float(np.percentile(z2_sm[inner], 75))
        carved = m_sub & (~(inner & (z2_sm >= ridge_thr)))
        dt_carved = gaussian_filter(distance_transform_edt(carved).astype(np.float32), sigma=1.5)
        pks_c = peak_local_max(
            dt_carved,
            min_distance=DT_PEAK_MIN_DIST,
            threshold_abs=5.5,
            labels=m_sub.astype(np.int32),
        )
        for p in pks_c:
            pt = tuple(int(x) for x in p)
            if far_mask[pt] and dt[pt] >= DT_PEAK_MIN_VAL:
                cand_p2_list.append(pt)

    # 1c. h-maxima of combined DT + cell-body field inside far_mask
    if has_c0_multi or max_span >= 82 or bbox_fill <= 0.335 or vox >= 130000:
        combo = dt_sm + 3.0 * body_norm
        hmax = h_maxima(combo * m_sub, h=DT_HMAX_H) & far_mask & (dt >= DT_PEAK_MIN_VAL)
        if np.any(hmax):
            pts = np.argwhere(hmax)
            best_pt = pts[int(np.argmax([float(combo[tuple(p)]) for p in pts]))]
            cand_p2_list.append(tuple(int(x) for x in best_pt))

    # 1d / 2. Exterior core maximum when C0 confirms >=2 cells or object is large and non-compact
    if (
        has_c0_multi
        or (vox >= 100000 and (max_span >= 82 or bbox_fill <= 0.335))
        or (vox >= 85000 and bbox_fill <= 0.25)
    ):
        score_far = core_score * far_mask
        if has_c0_multi:
            c0_sec = c0_seed_masks[1][1] & far_mask
            if int(c0_sec.sum()) >= 1000 and float((dt * c0_sec).max()) >= DT_PEAK_MIN_VAL:
                score_far = score_far * (1.0 + 0.5 * c0_sec.astype(np.float32))
        if float(score_far.max()) >= DT_PEAK_MIN_VAL:
            pt_far = tuple(int(x) for x in np.unravel_index(int(np.argmax(score_far)), m_sub.shape))
            if float(dt[pt_far]) >= DT_PEAK_MIN_VAL:
                cand_p2_list.append(pt_far)

    if not cand_p2_list:
        return None, dt_sm

    # Select the candidate with the strongest interior core score (dt_sm + 2.0 * body_norm)
    p2 = max(cand_p2_list, key=lambda pt: float(core_score[pt]))
    r2 = float(dt[p2])
    if r2 < DT_PEAK_MIN_VAL:
        return None, dt_sm

    dist2 = np.sqrt((zz - p2[0]) ** 2 + (yy - p2[1]) ** 2 + (xx - p2[2]) ** 2)
    d1_norm = dist1 / max(6.0, r1 ** 0.65)
    d2_norm = dist2 / max(6.0, r2 ** 0.65)

    # Scale-normalized seed distance + internal ch2 membrane ridge & valley watershed cost
    cost = 12.0 * np.minimum(d1_norm, d2_norm) + 8.0 * mem_ridge - 3.0 * body_norm - 0.35 * dt_sm

    markers = np.zeros(m_sub.shape, dtype=np.int32)
    markers[p1] = 1
    markers[p2] = 2
    for idx in (1, 2):
        dil_s = binary_dilation(markers == idx, iterations=2) & m_sub
        markers[dil_s] = idx

    return markers, cost


def split_merged_targets_with_stats(
    raw: np.ndarray,
    t_inst: np.ndarray,
    b_inst: np.ndarray,
    c0_b_inst: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run C5 target cleaning and splitting and return (b_inst_new, stats_dict)."""
    c4_uids = [int(u) for u in np.unique(b_inst) if u > 0]
    c4_n = len(c4_uids)

    _, z1, z2, sytox_mask = _channel_zscores(raw)

    # Step 1: Remove non-cell objects (wall beads, faint background haze, intra-T-cell inclusions)
    b_clean, n_removed = remove_non_cell_objects(b_inst, t_inst, z1, z2, sytox_mask, c0_b_inst=c0_b_inst)

    # Step 2: Re-merge over-split concentric fragments of single cells (e.g. v01_MIP_ROI_2 v5, v08_MIP_ROI_24 v2/v18)
    b_clean, remerged_uids, n_remerged = _remerge_concentric_single_cells(b_clean, c0_b_inst=c0_b_inst)

    uids = [int(u) for u in np.unique(b_clean) if u > 0]
    if not uids:
        stats = {
            "c4_n": c4_n,
            "c5_n": 0,
            "n_removed": n_removed,
            "n_remerged": n_remerged,
            "n_split_objs": 0,
            "n_split_children": 0,
        }
        return np.zeros_like(b_inst, dtype=np.uint16), stats

    out = np.zeros_like(b_inst, dtype=np.uint16)
    next_id = 1
    n_split_objs = 0
    n_split_children = 0

    for u in uids:
        m = b_clean == u
        vox = int(m.sum())
        if vox < SPLIT_MIN_OBJ_VOXELS or (u in remerged_uids):
            out[m] = next_id
            next_id += 1
            continue

        sl = bbox_slices(m, 3, m.shape)
        m_sub = m[sl]
        z2_sub = z2[sl]
        c0_sub = c0_b_inst[sl] if c0_b_inst is not None else None
        other_b_sub = (b_clean[sl] > 0) & (~m_sub)

        markers, cost = _find_split_seeds_and_cost(m_sub, z2_sub, c0_sub, other_b_sub)
        if markers is None or int(markers.max()) < 2:
            out[m] = next_id
            next_id += 1
            continue

        ws = watershed(cost, markers, mask=m_sub)
        child_ids = [int(cid) for cid in np.unique(ws) if cid > 0]
        valid_children = []
        for cid in child_ids:
            cm = ws == cid
            cvox = int(cm.sum())
            if cvox < SPLIT_MIN_CHILD_VOX:
                continue
            cdt = float(distance_transform_edt(cm).max())
            if cdt < SPLIT_MIN_CHILD_DT:
                continue
            valid_children.append((cvox, cid))

        if len(valid_children) < 2:
            out[m] = next_id
            next_id += 1
            continue

        valid_cids = {cid for _, cid in valid_children}
        if len(valid_cids) < len(child_ids):
            keep_markers = np.where(np.isin(ws, list(valid_cids)), ws, 0)
            ws = watershed(cost, keep_markers, mask=m_sub)

        out_sub = out[sl]
        for _, cid in sorted(valid_children, reverse=True):
            cm = binary_fill_holes(ws == cid) & m_sub
            out_sub[cm] = next_id
            next_id += 1
        n_split_objs += 1
        n_split_children += len(valid_children)

    b_final = _relabel_sequential(out, min_size=MIN_CELL_VOXELS)
    c5_n = int(len([u for u in np.unique(b_final) if u > 0]))
    stats = {
        "c4_n": c4_n,
        "c5_n": c5_n,
        "n_removed": n_removed,
        "n_remerged": n_remerged,
        "n_split_objs": n_split_objs,
        "n_split_children": n_split_children,
    }
    return b_final, stats


def split_merged_targets(
    raw: np.ndarray,
    t_inst: np.ndarray,
    b_inst: np.ndarray,
    c0_b_inst: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Split touching tumour cells merged into one b_inst object and remove non-cell objects."""
    b_final, _ = split_merged_targets_with_stats(raw, t_inst, b_inst, c0_b_inst=c0_b_inst)
    return b_final


def main() -> None:
    """Process cached C4 npz of all 20 chambers (sweep 1 of every visit) and save C5 npz + summary CSV."""
    parser = argparse.ArgumentParser(description="Run C5 target splitting and cleaning on all 20 chambers.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing C5 npz files.")
    args = parser.parse_args()

    manifest = pd.read_csv(REPO_ROOT / "data" / "scidb_19235_raw" / "manifest.csv")
    all_chambers = list(PILOT_CHAMBERS) + list(PILOT4_CHAMBERS)
    out_root = REPO_ROOT / "data" / "pilot_seg"
    sum_csv = out_root / "c5_split_summary.csv"
    existing_summary: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    if sum_csv.exists() and not args.overwrite:
        df_old = pd.read_csv(sum_csv)
        for _, r in df_old.iterrows():
            existing_summary[(str(r["experiment"]), str(r["roi"]), f"{int(r['stem']):05d}")] = dict(r)

    rows: List[Dict[str, Any]] = []
    t_global0 = time.perf_counter()

    for ch_idx, (short_exp, exp, roi) in enumerate(all_chambers, start=1):
        sub = manifest[(manifest["experiment"] == exp) & (manifest["roi"] == roi)].copy()
        visit_rows = []
        for visit_idx, vdf in sub.groupby("visit_index"):
            vdf_s1 = vdf[vdf["sweep_index"] == 1]
            if len(vdf_s1) > 0:
                chosen = vdf_s1.iloc[0]
            else:
                vdf_s2 = vdf[vdf["sweep_index"] == 2]
                chosen = vdf_s2.iloc[0] if len(vdf_s2) > 0 else vdf.sort_values("sweep_index").iloc[0]
            visit_rows.append(
                {
                    "visit": int(visit_idx),
                    "sweep": int(chosen["sweep_index"]),
                    "stem": f"{int(chosen['stem']):05d}",
                }
            )
        visit_rows.sort(key=lambda d: d["visit"])

        ch_t0 = time.perf_counter()

        for item in visit_rows:
            visit_num = int(item["visit"])
            stem = str(item["stem"])
            h5_path = REPO_ROOT / "data" / "scidb_19235_raw" / exp / roi / f"{stem}.h5"
            c4_path = out_root / "C4" / exp / roi / f"{stem}.npz"
            c0_path = out_root / "C0" / exp / roi / f"{stem}.npz"
            c5_path = out_root / "C5" / exp / roi / f"{stem}.npz"

            if c5_path.exists() and not args.overwrite:
                key = (exp, roi, stem)
                if key in existing_summary:
                    rows.append(existing_summary[key])
                    continue
                c4_data = np.load(c4_path)
                c5_data = np.load(c5_path)
                c4_n = int(len([u for u in np.unique(c4_data["b_inst"]) if u > 0]))
                c5_n = int(len([u for u in np.unique(c5_data["b_inst"]) if u > 0]))
                rows.append(
                    {
                        "short_exp": short_exp,
                        "experiment": exp,
                        "roi": roi,
                        "visit_num": visit_num,
                        "stem": stem,
                        "c4_n": c4_n,
                        "c5_n": c5_n,
                        "n_removed": max(0, c4_n - c5_n),
                        "n_remerged": 0,
                        "n_split_objs": max(0, c5_n - c4_n),
                        "n_split_children": 2 * max(0, c5_n - c4_n),
                        "runtime_s": round(float(c5_data["runtime_s"]), 4),
                    }
                )
                continue

            c4_data = np.load(c4_path)
            t_inst = c4_data["t_inst"].astype(np.uint16)
            b_inst = c4_data["b_inst"].astype(np.uint16)
            tracked_t_id = int(c4_data["tracked_t_id"])
            tracked_b_id = int(c4_data["tracked_b_id"])
            c0_b_inst = np.load(c0_path)["b_inst"].astype(np.uint16) if c0_path.exists() else None

            with h5py.File(h5_path, "r") as f:
                raw = f["raw"][:]
            t0 = time.perf_counter()
            b_new, st = split_merged_targets_with_stats(raw, t_inst, b_inst, c0_b_inst=c0_b_inst)
            dt_s = time.perf_counter() - t0

            save_candidate_npz(
                c5_path,
                t_inst,
                b_new,
                tracked_t_id,
                tracked_b_id,
                dt_s,
            )
            del raw

            rows.append(
                {
                    "short_exp": short_exp,
                    "experiment": exp,
                    "roi": roi,
                    "visit_num": visit_num,
                    "stem": stem,
                    "c4_n": st["c4_n"],
                    "c5_n": st["c5_n"],
                    "n_removed": st["n_removed"],
                    "n_remerged": st["n_remerged"],
                    "n_split_objs": st["n_split_objs"],
                    "n_split_children": st["n_split_children"],
                    "runtime_s": round(dt_s, 4),
                }
            )
            gc.collect()

        ch_dt = time.perf_counter() - ch_t0
        ch_rows = [r for r in rows if r["experiment"] == exp and r["roi"] == roi]
        sp_visits = [r["visit_num"] for r in ch_rows if r["n_split_objs"] > 0]
        rm_visits = [r["visit_num"] for r in ch_rows if r["n_removed"] > 0]
        print(
            f"[{ch_idx:02d}/20] {short_exp}_{roi}: {len(ch_rows)} visits in {ch_dt:.1f}s "
            f"| split_visits={sp_visits} | removed_visits={rm_visits}",
            flush=True,
        )

    df_sum = pd.DataFrame(rows)
    df_sum.to_csv(sum_csv, index=False)
    print(f"Done in {time.perf_counter() - t_global0:.1f}s. Summary saved to {sum_csv}", flush=True)


if __name__ == "__main__":
    main()
