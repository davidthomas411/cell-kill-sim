"""Whole-series cell selection and Hungarian tracking for CAR-T + target chambers (Phase 5).

Rules implemented:
1. T cell = the ch1-bright cell with the largest volume that persists across visits
   (evaluated over the whole series, not one visit), followed visit-to-visit by
   3D overlap and centroid distance (Hungarian assignment), tolerating missing visits.
2. Target = the tumour cell with the largest contact area with the tracked T cell
   at FIRST contact (using count_interface_faces from src/seg/candidates.py), then
   followed forward (and backward to visit 1 if first contact > visit 1) solely by
   overlap/centroid distance (Hungarian) without ever re-picking by contact on later
   visits. Preserves target identity through cell death (SYTOX ch0 sigma=3 > 404 ADU)
   and after contact ends; once a target is dead, prevents switching to a non-overlapping
   live cell after the dead corpse leaves. Reports 'no target' (target_track_id=0,
   tracked_b_id=0) if the T cell never touches a tumour cell.
3. Accepts instance source parameter ('C4' now, 'C5' later) without depending on split5.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, gaussian_filter, generate_binary_structure
from scipy.optimize import linear_sum_assignment

from src.seg.candidates import (
    FACE_AREA_UM2,
    PILOT4_CHAMBERS,
    PILOT_CHAMBERS,
    REPO_ROOT,
    VOXEL_UM,
    VOXEL_VOL_UM3,
    count_interface_faces,
)

ALL_20_CHAMBERS: List[Tuple[str, str, str]] = PILOT_CHAMBERS + PILOT4_CHAMBERS
STRUCT6 = generate_binary_structure(3, 1)

# Threshold constants
MIN_TCELL_VOX_ABS = 8500          # ~64 um^3; rejects wall beads (2.5k-7.8k vox in v12)
MIN_TCELL_VOX_REL = 0.08          # >=8% of chamber's 75th-percentile largest T-cell volume
MIN_TARGET_VOX_ABS = 2500         # ~19 um^3; minimum tumour instance size in C4
MIN_FIRST_CONTACT_FACES = 250     # >=250 shared voxel faces (~9.6 um^2) for genuine synapse contact (fallback >0)

TCELL_MAX_DIST_PX = 185.0         # ~36.3 um max centroid jump per visit for motile T cell / stage shift
TCELL_MAX_GAP_VISITS = 6          # tolerate up to 5 consecutive missing visits (e.g. v12_MIP_ROI_23 v5-7)

TARGET_MAX_DIST_LIVE_PX = 175.0   # ~34.3 um max centroid jump for live target (handles chamber shifts in v05_ROI_8)
TARGET_MAX_DIST_DEAD_PX = 125.0   # ~24.5 um max centroid jump for floating dead target corpse (v02_MIP_ROI_3 v19)
TARGET_MAX_GAP_VISITS = 8         # tolerate missing visits without dropping the target track
MERGE_VOL_RATIO_THRESH = 1.38     # target volume > 1.38x unmerged core volume indicates merged instance
DEAD_DYE_VOX_THRESH = 2000        # >=2000 SYTOX voxels (>404 ADU)
DEAD_DYE_FRAC_THRESH = 0.20       # >=20% of target volume filled with SYTOX dye marks target as dead

UNCERTAIN_MIN_IOU = 0.10          # flag visit match as uncertain if 3D IoU < 0.10
UNCERTAIN_MAX_JUMP_PX = 45.0      # flag visit match as uncertain if centroid jump > 45 px (~8.8 um)


@dataclass
class InstanceProps:
    uid: int
    vox: int
    centroid: np.ndarray
    mask: np.ndarray
    ch1_excess: float = 0.0
    ch2_excess: float = 0.0
    dye_vox: int = 0
    dye_frac: float = 0.0


def load_chamber_visits(exp: str, roi: str) -> List[Dict[str, object]]:
    """Return sorted sweep-1 visit metadata rows for a chamber from manifest.csv."""
    manifest_df = pd.read_csv(REPO_ROOT / "data" / "scidb_19235_raw" / "manifest.csv")
    sub = manifest_df[
        (manifest_df["experiment"] == exp)
        & (manifest_df["roi"] == roi)
        & (manifest_df["ok"] == True)  # noqa: E712
        & (manifest_df["sweep_index"] != 6)
    ].copy()

    visit_rows: List[Dict[str, object]] = []
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
                "rel_path": str(chosen["path"]),
            }
        )
    visit_rows.sort(key=lambda d: int(d["visit"]))
    return visit_rows


def _compute_sytox_mask_on_b(raw0: np.ndarray, b_inst: np.ndarray) -> np.ndarray:
    """Compute exact (sigma=3.0 gaussian > 404.0) SYTOX mask over the bounding box of b_inst > 0."""
    sytox_dye = np.zeros(b_inst.shape, dtype=bool)
    b_any = b_inst > 0
    if not np.any(b_any):
        return sytox_dye
    nz = np.argwhere(b_any)
    pad = 13
    lo = np.maximum(0, nz.min(axis=0) - pad)
    hi = np.minimum(b_inst.shape, nz.max(axis=0) + pad + 1)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    sm0_sub = gaussian_filter(raw0[sl].astype(np.float32), sigma=3.0)
    sytox_dye[sl] = sm0_sub > 404.0
    return sytox_dye


def extract_visit_props(
    t_inst: np.ndarray,
    b_inst: np.ndarray,
    raw: np.ndarray,
) -> Tuple[List[InstanceProps], List[InstanceProps], np.ndarray]:
    """Compute per-instance geometric and channel properties for T and tumour instances."""
    sytox_dye = _compute_sytox_mask_on_b(raw[0], b_inst)

    ch1 = raw[1]
    ch2 = raw[2]
    bg1 = float(np.median(ch1[::4, ::4, ::4]))
    bg2 = float(np.median(ch2[::4, ::4, ::4]))

    t_props: List[InstanceProps] = []
    for u in np.unique(t_inst):
        if u == 0:
            continue
        m = t_inst == int(u)
        vox = int(m.sum())
        cent = np.argwhere(m).mean(axis=0).astype(np.float64)
        ch1_ex = max(0.0, float(ch1[m].mean()) - bg1)
        t_props.append(
            InstanceProps(
                uid=int(u),
                vox=vox,
                centroid=cent,
                mask=m,
                ch1_excess=ch1_ex,
            )
        )

    b_props: List[InstanceProps] = []
    for u in np.unique(b_inst):
        if u == 0:
            continue
        m = b_inst == int(u)
        vox = int(m.sum())
        cent = np.argwhere(m).mean(axis=0).astype(np.float64)
        ch2_ex = max(0.0, float(ch2[m].mean()) - bg2)
        dvox = int((m & sytox_dye).sum())
        dfrac = float(dvox) / float(vox) if vox > 0 else 0.0
        b_props.append(
            InstanceProps(
                uid=int(u),
                vox=vox,
                centroid=cent,
                mask=m,
                ch2_excess=ch2_ex,
                dye_vox=dvox,
                dye_frac=dfrac,
            )
        )

    return t_props, b_props, sytox_dye


def track_tcell_series(
    all_t_props: List[List[InstanceProps]],
) -> List[Tuple[int, int, float, float, int, str]]:
    """Select and track the persistent large ch1-bright T cell across all visits.

    Returns per visit: (tracked_t_id, tcell_track_id, iou_prev, jump_px, uncertain_flag, reason).
    """
    n_visits = len(all_t_props)
    max_vols = [max((p.vox for p in vprops), default=0) for vprops in all_t_props]
    nonzero_max = [v for v in max_vols if v > 0]
    ref_t_vol = float(np.percentile(nonzero_max, 75)) if nonzero_max else 0.0
    min_t_vox = max(MIN_TCELL_VOX_ABS, int(MIN_TCELL_VOX_REL * ref_t_vol))

    # Filter out small wall beads / debris fragments
    filtered_props: List[List[InstanceProps]] = []
    for vprops in all_t_props:
        kept = [p for p in vprops if p.vox >= min_t_vox]
        if not kept and vprops and ref_t_vol < MIN_TCELL_VOX_ABS:
            kept = [max(vprops, key=lambda p: p.vox)]
        filtered_props.append(kept)

    tracks: Dict[int, Dict[str, object]] = {}
    next_tid = 1

    for v in range(n_visits):
        cur_list = filtered_props[v]
        if not cur_list:
            continue
        active_tids = [
            tid for tid, tr in tracks.items() if (v - int(tr["last_v"])) <= TCELL_MAX_GAP_VISITS
        ]
        assigned_j = set()

        if active_tids:
            n_act = len(active_tids)
            n_cur = len(cur_list)
            cost = np.zeros((n_act, n_cur + n_act), dtype=np.float64)
            match_meta: Dict[Tuple[int, int], Tuple[float, float]] = {}

            for i, tid in enumerate(active_tids):
                pm: np.ndarray = tracks[tid]["mask"]  # type: ignore[assignment]
                pc: np.ndarray = tracks[tid]["centroid"]  # type: ignore[assignment]
                pv = int(tracks[tid]["vox"])
                hist_len = len(tracks[tid]["history"])  # type: ignore[arg-type]
                gap = v - int(tracks[tid]["last_v"])
                eff_max_dist = TCELL_MAX_DIST_PX * (1.0 + 0.25 * (gap - 1))

                for j, cp in enumerate(cur_list):
                    inter = int(np.sum(pm & cp.mask))
                    union = int(np.sum(pm | cp.mask))
                    iou = float(inter) / float(union) if union > 0 else 0.0
                    ov_min = float(inter) / float(min(pv, cp.vox)) if min(pv, cp.vox) > 0 else 0.0
                    dist = float(np.linalg.norm(pc - cp.centroid))
                    vol_ratio = float(min(pv, cp.vox)) / float(max(pv, cp.vox)) if max(pv, cp.vox) > 0 else 0.0
                    vol_norm = min(1.0, float(cp.vox) / max(1.0, ref_t_vol))
                    maturity_bonus = 0.45 * min(5, hist_len)

                    if iou > 0.01 or dist <= eff_max_dist:
                        sim = (
                            2.5 * iou
                            + 1.0 * ov_min * vol_ratio
                            + 1.5 * max(0.0, 1.0 - dist / eff_max_dist)
                            + 0.8 * vol_ratio
                            + 0.6 * vol_norm
                            + maturity_bonus
                            - 0.05 * (gap - 1)
                        )
                        if sim > 0.20:
                            cost[i, j] = -sim
                            match_meta[(i, j)] = (iou, dist)

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if c < n_cur and cost[r, c] < 0.0:
                    tid = active_tids[r]
                    cp = cur_list[c]
                    iou, dist = match_meta[(r, c)]
                    tracks[tid]["last_v"] = v
                    tracks[tid]["mask"] = cp.mask
                    tracks[tid]["centroid"] = cp.centroid
                    tracks[tid]["vox"] = cp.vox
                    tracks[tid]["history"][v] = (cp.uid, iou, dist, cp.vox, cp.ch1_excess)  # type: ignore[index]
                    assigned_j.add(c)

        for j, cp in enumerate(cur_list):
            if j not in assigned_j:
                tracks[next_tid] = {
                    "last_v": v,
                    "mask": cp.mask,
                    "centroid": cp.centroid,
                    "vox": cp.vox,
                    "history": {v: (cp.uid, 1.0, 0.0, cp.vox, cp.ch1_excess)},
                }
                next_tid += 1

    if not tracks:
        return [(0, 0, 0.0, 0.0, 0, "no_tcell") for _ in range(n_visits)]

    def _score_t_track(tid: int) -> float:
        hist: Dict[int, Tuple[int, float, float, int, float]] = tracks[tid]["history"]  # type: ignore[assignment]
        total_v = sum(float(it[3]) * (1.0 + 0.1 * min(50.0, float(it[4]))) for it in hist.values())
        persistence = float(len(hist)) / float(max(1, n_visits))
        return total_v * (0.5 + 0.5 * persistence)

    sorted_tids = sorted(tracks.keys(), key=_score_t_track, reverse=True)
    best_tid = sorted_tids[0]
    best_hist: Dict[int, Tuple[int, float, float, int, float]] = dict(tracks[best_tid]["history"])  # type: ignore[arg-type]

    # If the primary T-cell track went unassigned on a visit because two T-cell fragments
    # merged into one instance (won by a secondary track in 1-to-1 assignment) or jumped,
    # fill any missing visit from the highest-ranked track present on that visit.
    for other_tid in sorted_tids[1:]:
        other_hist: Dict[int, Tuple[int, float, float, int, float]] = tracks[other_tid]["history"]  # type: ignore[assignment]
        for ov_v, val in other_hist.items():
            if ov_v not in best_hist and val[3] >= min_t_vox:
                best_hist[ov_v] = val

    # Recompute exact sequential IoU and centroid jump along the consolidated primary T-cell track
    sorted_vs = sorted(best_hist.keys())
    prop_by_v_uid: Dict[Tuple[int, int], InstanceProps] = {
        (v, p.uid): p for v in range(n_visits) for p in filtered_props[v]
    }
    for idx_v, v in enumerate(sorted_vs):
        uid, _, _, vox, ch1_ex = best_hist[v]
        if idx_v == 0:
            best_hist[v] = (uid, 1.0, 0.0, vox, ch1_ex)
        else:
            prev_v = sorted_vs[idx_v - 1]
            prev_uid = best_hist[prev_v][0]
            p_prev = prop_by_v_uid[(prev_v, prev_uid)]
            p_cur = prop_by_v_uid[(v, uid)]
            inter = int(np.sum(p_prev.mask & p_cur.mask))
            union = int(np.sum(p_prev.mask | p_cur.mask))
            iou = float(inter) / float(union) if union > 0 else 0.0
            dist = float(np.linalg.norm(p_prev.centroid - p_cur.centroid))
            best_hist[v] = (uid, iou, dist, vox, ch1_ex)

    out: List[Tuple[int, int, float, float, int, str]] = []
    first_v = sorted_vs[0]
    for v in range(n_visits):
        if v in best_hist:
            uid, iou, dist, _, _ = best_hist[v]
            unc = 0
            reasons: List[str] = []
            if v > first_v:
                if iou < UNCERTAIN_MIN_IOU:
                    unc = 1
                    reasons.append(f"t_low_iou({iou:.2f})")
                if dist > UNCERTAIN_MAX_JUMP_PX:
                    unc = 1
                    reasons.append(f"t_jump({dist:.1f}px)")
            out.append((int(uid), 1, round(float(iou), 4), round(float(dist), 2), unc, ";".join(reasons)))
        else:
            out.append((0, 0, 0.0, 0.0, 1, "t_missing_visit"))
    return out


def _extract_contact_subcore(
    b_mask: np.ndarray,
    t_mask: np.ndarray,
    typical_cell_vox: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """If initial target mask is a multi-cell merge, localize core around the T-cell interface."""
    b_vox = int(b_mask.sum())
    if b_vox <= int(MERGE_VOL_RATIO_THRESH * typical_cell_vox):
        cent = np.argwhere(b_mask).mean(axis=0).astype(np.float64)
        return b_mask, cent, b_vox

    t_dil = binary_dilation(t_mask, structure=STRUCT6, iterations=2)
    iface = b_mask & t_dil
    if not np.any(iface):
        cent = np.argwhere(b_mask).mean(axis=0).astype(np.float64)
        return b_mask, cent, b_vox

    iface_cent = np.argwhere(iface).mean(axis=0).astype(np.float64)
    coords = np.argwhere(b_mask)
    dists = np.linalg.norm(coords - iface_cent[None, :], axis=1)
    k_keep = min(len(dists), max(MIN_TARGET_VOX_ABS, int(round(typical_cell_vox))))
    thresh_d = float(np.partition(dists, k_keep - 1)[k_keep - 1])
    sub_coords = coords[dists <= max(55.0, thresh_d)]
    sub_mask = np.zeros_like(b_mask, dtype=bool)
    sub_mask[sub_coords[:, 0], sub_coords[:, 1], sub_coords[:, 2]] = True
    sub_cent = sub_coords.mean(axis=0).astype(np.float64)
    return sub_mask, sub_cent, int(sub_mask.sum())


def _localize_core_in_merge(
    merged_mask: np.ndarray,
    prior_core_cent: np.ndarray,
    target_vox: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Localize a track's core mask and centroid within a merged multi-cell mask."""
    coords = np.argwhere(merged_mask)
    dists = np.linalg.norm(coords - prior_core_cent[None, :], axis=1)
    k_keep = min(len(dists), max(MIN_TARGET_VOX_ABS, int(target_vox)))
    thresh_d = float(np.partition(dists, k_keep - 1)[k_keep - 1])
    sub_coords = coords[dists <= max(50.0, thresh_d)]
    sub_m = np.zeros_like(merged_mask, dtype=bool)
    sub_m[sub_coords[:, 0], sub_coords[:, 1], sub_coords[:, 2]] = True
    return sub_m, sub_coords.mean(axis=0).astype(np.float64)


def track_target_series(
    all_b_props: List[List[InstanceProps]],
    t_masks: List[np.ndarray],
) -> List[Tuple[int, int, float, float, int, str]]:
    """Pick target at FIRST contact with tracked T cell, then track forward/backward by Hungarian.

    Never re-picks by contact on later visits. Keeps identity through SYTOX death and after contact ends.
    """
    n_visits = len(all_b_props)
    valid_b: List[List[InstanceProps]] = [
        [p for p in all_b_props[v] if p.vox >= MIN_TARGET_VOX_ABS] for v in range(n_visits)
    ]

    all_vols = [p.vox for vlist in valid_b for p in vlist if p.vox >= 10000]
    typical_b_vox = float(np.percentile(all_vols, 40)) if all_vols else 75000.0
    typical_b_vox = float(np.clip(typical_b_vox, 25000.0, 105000.0))

    t_cents: List[Optional[np.ndarray]] = [
        np.argwhere(tm).mean(axis=0).astype(np.float64) if np.any(tm) else None
        for tm in t_masks
    ]

    # 1. Find FIRST contact between tracked T cell and any valid tumour cell
    first_contact_v: Optional[int] = None
    first_contact_uid: int = 0

    for min_faces in (MIN_FIRST_CONTACT_FACES, 25, 1):
        for v in range(n_visits):
            t_m = t_masks[v]
            if not np.any(t_m) or not valid_b[v]:
                continue
            best_u = 0
            best_faces = 0
            for bp in valid_b[v]:
                faces = count_interface_faces(t_m, bp.mask)
                if faces > best_faces:
                    best_faces = faces
                    best_u = bp.uid
            if best_faces >= min_faces:
                first_contact_v = v
                first_contact_uid = best_u
                break
        if first_contact_v is not None:
            break

    if first_contact_v is None:
        return [(0, 0, 0.0, 0.0, 0, "no_target") for _ in range(n_visits)]

    fc_prop = next(bp for bp in valid_b[first_contact_v] if bp.uid == first_contact_uid)
    fc_core_mask, fc_core_cent, fc_core_vox = _extract_contact_subcore(
        fc_prop.mask, t_masks[first_contact_v], typical_b_vox
    )
    fc_is_merged = fc_prop.vox > int(MERGE_VOL_RATIO_THRESH * typical_b_vox)

    res: Dict[int, Tuple[int, int, float, float, int, str]] = {}
    res[first_contact_v] = (
        first_contact_uid,
        1,
        1.0,
        0.0,
        1 if fc_is_merged else 0,
        "b_merged" if fc_is_merged else "",
    )

    def _init_track(bp: InstanceProps, is_target: bool) -> Dict[str, object]:
        is_dead = (bp.dye_vox >= DEAD_DYE_VOX_THRESH) and (bp.dye_frac >= DEAD_DYE_FRAC_THRESH)
        if is_target:
            c_mask, c_cent, c_vox = fc_core_mask, fc_core_cent, fc_core_vox
        else:
            c_mask, c_cent, c_vox = bp.mask, bp.centroid, min(bp.vox, int(typical_b_vox))
        return {
            "is_target": is_target,
            "last_v": first_contact_v,
            "mask": bp.mask,
            "centroid": bp.centroid,
            "vox": bp.vox,
            "core_mask": c_mask,
            "core_centroid": c_cent,
            "core_vox": c_vox,
            "is_dead": is_dead,
        }

    def _propagate(visit_indices: List[int]) -> None:
        tracks: Dict[int, Dict[str, object]] = {}
        next_tid = 2
        for bp in valid_b[first_contact_v]:
            if bp.uid == first_contact_uid:
                tracks[1] = _init_track(bp, is_target=True)
            else:
                tracks[next_tid] = _init_track(bp, is_target=False)
                next_tid += 1

        for v in visit_indices:
            cur_list = valid_b[v]
            if not cur_list:
                res[v] = (0, 0, 0.0, 0.0, 1, "b_missing_visit")
                continue

            active_tids = [
                tid
                for tid, tr in tracks.items()
                if (tid == 1 and abs(v - int(tr["last_v"])) <= TARGET_MAX_GAP_VISITS)
                or (tid != 1 and abs(v - int(tr["last_v"])) <= 2)
            ]
            assigned_j = set()
            tgt_matched = False

            if active_tids:
                n_act = len(active_tids)
                n_cur = len(cur_list)
                cost = np.zeros((n_act, n_cur + n_act), dtype=np.float64)
                match_meta: Dict[Tuple[int, int], Tuple[float, float, bool]] = {}

                for i, tid in enumerate(active_tids):
                    tr = tracks[tid]
                    pm: np.ndarray = tr["mask"]  # type: ignore[assignment]
                    pc: np.ndarray = tr["centroid"]  # type: ignore[assignment]
                    pv = int(tr["vox"])
                    cm_core: np.ndarray = tr["core_mask"]  # type: ignore[assignment]
                    cc_core: np.ndarray = tr["core_centroid"]  # type: ignore[assignment]
                    cv_core = int(tr["core_vox"])
                    was_dead = bool(tr["is_dead"])
                    last_v = int(tr["last_v"])
                    gap = abs(v - last_v)

                    # Compute stage/chamber shift compensation from T-cell centroid motion when big shift occurs
                    t_shift = np.zeros(3, dtype=np.float64)
                    if t_cents[v] is not None and t_cents[last_v] is not None:
                        dt_vec = t_cents[v] - t_cents[last_v]  # type: ignore[operator]
                        if float(np.linalg.norm(dt_vec)) > 45.0:
                            t_shift = 0.65 * dt_vec

                    max_d = TARGET_MAX_DIST_DEAD_PX if was_dead else TARGET_MAX_DIST_LIVE_PX
                    eff_max_dist = max_d * (1.0 + 0.20 * (gap - 1))

                    for j, cp in enumerate(cur_list):
                        cp_dead = (cp.dye_vox >= DEAD_DYE_VOX_THRESH) and (cp.dye_frac >= DEAD_DYE_FRAC_THRESH)
                        inter = int(np.sum(pm & cp.mask))
                        union = int(np.sum(pm | cp.mask))
                        iou = float(inter) / float(union) if union > 0 else 0.0
                        ov_prev = float(inter) / float(pv) if pv > 0 else 0.0
                        dist_raw = float(np.linalg.norm(pc - cp.centroid))
                        dist_rel = float(np.linalg.norm((pc + t_shift) - cp.centroid))
                        dist_eff = min(dist_raw, dist_rel)

                        inter_c = int(np.sum(cm_core & cp.mask))
                        union_c = int(np.sum(cm_core | cp.mask))
                        iou_core = float(inter_c) / float(union_c) if union_c > 0 else 0.0
                        ov_core = float(inter_c) / float(cv_core) if cv_core > 0 else 0.0
                        dist_core = min(
                            float(np.linalg.norm(cc_core - cp.centroid)),
                            float(np.linalg.norm((cc_core + t_shift) - cp.centroid)),
                        )

                        # Rule: once dead (high SYTOX), target cannot resurrect into a non-overlapping live cell
                        if was_dead and (not cp_dead) and (cp.dye_frac < 0.05) and (iou < 0.05):
                            continue

                        vol_r_core = float(min(cv_core, cp.vox)) / float(max(cv_core, cp.vox)) if max(cv_core, cp.vox) > 0 else 0.0
                        vol_r_prev = float(min(pv, cp.vox)) / float(max(pv, cp.vox)) if max(pv, cp.vox) > 0 else 0.0
                        vol_r = max(vol_r_core, vol_r_prev)
                        if cp_dead and cp.vox >= 8000:
                            vol_r = max(vol_r, 0.70)

                        if iou > 0.01 or iou_core > 0.01 or dist_eff <= eff_max_dist or dist_core <= eff_max_dist:
                            prox_imm = (
                                2.4 * iou
                                + 1.2 * ov_prev
                                + 1.4 * max(0.0, 1.0 - dist_eff / eff_max_dist)
                            )
                            prox_core = (
                                1.8 * iou_core
                                + 1.2 * ov_core
                                + 1.2 * max(0.0, 1.0 - dist_core / eff_max_dist)
                            )
                            dead_bonus = 1.0 if (was_dead and cp_dead) else 0.0
                            sim = (prox_imm + 0.75 * prox_core) * (0.35 + 0.65 * vol_r) + 0.6 * vol_r + dead_bonus - 0.05 * (gap - 1)
                            if sim > 0.15:
                                cost[i, j] = -sim
                                is_merged = cp.vox > int(MERGE_VOL_RATIO_THRESH * max(cv_core, typical_b_vox))
                                match_meta[(i, j)] = (iou, dist_raw, is_merged)

                row_ind, col_ind = linear_sum_assignment(cost)
                win_map: Dict[int, int] = {}
                for r, c in zip(row_ind, col_ind):
                    if c < n_cur and cost[r, c] < 0.0:
                        win_map[active_tids[r]] = c

                # Handle multi-cell merge: if tid=1 (target) went unassigned in 1-to-1 assignment because
                # another track tid=k merged with tid=1 into the same instance cp_j (or stole the sole merged instance),
                # assign tid=1 to that merged instance as well while keeping both cores separate.
                if 1 in active_tids and 1 not in win_map:
                    i_tgt = active_tids.index(1)
                    valid_js = [j for j in range(n_cur) if cost[i_tgt, j] < 0.0]
                    if valid_js:
                        best_j = min(valid_js, key=lambda j: cost[i_tgt, j])
                        iou_t, dist_t, is_m_t = match_meta[(i_tgt, best_j)]
                        cp_best = cur_list[best_j]
                        # Check if cp_best is a merged instance or overlaps tid=1's core/mask or is the only cell
                        if is_m_t or iou_t > 0.02 or n_cur == 1 or dist_t <= 75.0:
                            win_map[1] = best_j
                            match_meta[(i_tgt, best_j)] = (iou_t, dist_t, True)

                for tid, c in win_map.items():
                    r = active_tids.index(tid)
                    cp = cur_list[c]
                    iou, dist, is_merged = match_meta[(r, c)]
                    tr = tracks[tid]
                    tr["last_v"] = v
                    tr["mask"] = cp.mask
                    tr["centroid"] = cp.centroid
                    tr["vox"] = cp.vox
                    if (cp.dye_vox >= DEAD_DYE_VOX_THRESH) and (cp.dye_frac >= DEAD_DYE_FRAC_THRESH):
                        tr["is_dead"] = True

                    if not is_merged:
                        tr["core_mask"] = cp.mask
                        tr["core_centroid"] = cp.centroid
                        if not bool(tr["is_dead"]):
                            tr["core_vox"] = int(0.7 * int(tr["core_vox"]) + 0.3 * cp.vox)
                    else:
                        sub_m, sub_c = _localize_core_in_merge(
                            cp.mask,
                            tr["core_centroid"],  # type: ignore[arg-type]
                            int(tr["core_vox"]),
                        )
                        tr["core_mask"] = sub_m
                        tr["core_centroid"] = sub_c

                    assigned_j.add(c)
                    if tid == 1:
                        tgt_matched = True
                        unc = 0
                        reasons: List[str] = []
                        if iou < UNCERTAIN_MIN_IOU:
                            unc = 1
                            reasons.append(f"b_low_iou({iou:.2f})")
                        if dist > UNCERTAIN_MAX_JUMP_PX:
                            unc = 1
                            reasons.append(f"b_jump({dist:.1f}px)")
                        if is_merged:
                            unc = 1
                            reasons.append("b_merged")
                        res[v] = (cp.uid, 1, round(float(iou), 4), round(float(dist), 2), unc, ";".join(reasons))

            # Only spawn new competitor tracks for genuinely separate large cells (far from target core)
            for j, cp in enumerate(cur_list):
                if j not in assigned_j and cp.vox >= int(0.45 * typical_b_vox):
                    if 1 in tracks:
                        d_to_tgt = float(np.linalg.norm(cp.centroid - tracks[1]["core_centroid"]))  # type: ignore[arg-type]
                        if d_to_tgt < 40.0:
                            continue
                    tracks[next_tid] = _init_track(cp, is_target=False)
                    tracks[next_tid]["last_v"] = v
                    next_tid += 1

            if not tgt_matched:
                res[v] = (0, 0, 0.0, 0.0, 1, "b_unmatched")

    if first_contact_v < n_visits - 1:
        _propagate(list(range(first_contact_v + 1, n_visits)))
    if first_contact_v > 0:
        _propagate(list(range(first_contact_v - 1, -1, -1)))

    # Recompute sequential visit-to-visit (v-1 -> v) IoU and jump for target track across the series
    prop_b_map: Dict[Tuple[int, int], InstanceProps] = {
        (v, p.uid): p for v in range(n_visits) for p in valid_b[v]
    }
    final_out: List[Tuple[int, int, float, float, int, str]] = []
    prev_matched_v: Optional[int] = None
    for v in range(n_visits):
        uid, b_tr, _, _, _, old_reason = res[v]
        if uid == 0 or b_tr == 0:
            final_out.append((0, 0, 0.0, 0.0, 1, old_reason if old_reason else "b_unmatched"))
            continue
        is_merged_flag = "b_merged" in old_reason
        if prev_matched_v is None:
            unc = 1 if is_merged_flag else 0
            r_str = "b_merged" if is_merged_flag else ""
            final_out.append((uid, 1, 1.0, 0.0, unc, r_str))
        else:
            p_prev = prop_b_map[(prev_matched_v, res[prev_matched_v][0])]
            p_cur = prop_b_map[(v, uid)]
            inter = int(np.sum(p_prev.mask & p_cur.mask))
            union = int(np.sum(p_prev.mask | p_cur.mask))
            iou = float(inter) / float(union) if union > 0 else 0.0
            dist = float(np.linalg.norm(p_prev.centroid - p_cur.centroid))
            unc = 0
            reasons: List[str] = []
            if iou < UNCERTAIN_MIN_IOU:
                unc = 1
                reasons.append(f"b_low_iou({iou:.2f})")
            if dist > UNCERTAIN_MAX_JUMP_PX:
                unc = 1
                reasons.append(f"b_jump({dist:.1f}px)")
            if is_merged_flag:
                unc = 1
                reasons.append("b_merged")
            final_out.append((uid, 1, round(iou, 4), round(dist, 2), unc, ";".join(reasons)))
        prev_matched_v = v

    return final_out


def track_chamber(
    short_exp: str,
    exp: str,
    roi: str,
    source: str = "C4",
    out_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Run whole-series T-cell and target selection + Hungarian tracking for one chamber."""
    t0 = time.perf_counter()
    visit_rows = load_chamber_visits(exp, roi)
    chamber_id = f"{short_exp}_{roi}"

    all_t_props: List[List[InstanceProps]] = []
    all_b_props: List[List[InstanceProps]] = []
    t_inst_list: List[np.ndarray] = []
    b_inst_list: List[np.ndarray] = []
    dye_list: List[np.ndarray] = []

    for item in visit_rows:
        stem = str(item["stem"])
        npz_path = REPO_ROOT / "data" / "pilot_seg" / source / exp / roi / f"{stem}.npz"
        raw_path = REPO_ROOT / item["rel_path"]

        with np.load(npz_path) as data:
            t_inst = data["t_inst"].astype(np.uint16)
            b_inst = data["b_inst"].astype(np.uint16)
        with h5py.File(raw_path, "r") as f_raw:
            raw = f_raw["raw"][:]

        tp, bp, sytox_dye = extract_visit_props(t_inst, b_inst, raw)
        all_t_props.append(tp)
        all_b_props.append(bp)
        t_inst_list.append(t_inst)
        b_inst_list.append(b_inst)
        dye_list.append(sytox_dye)

    t_res = track_tcell_series(all_t_props)
    t_masks = [
        (t_inst_list[v] == t_res[v][0]) if t_res[v][0] > 0 else np.zeros_like(t_inst_list[v], dtype=bool)
        for v in range(len(visit_rows))
    ]
    b_res = track_target_series(all_b_props, t_masks)

    elapsed = time.perf_counter() - t0
    rows_out: List[Dict[str, object]] = []
    for idx, item in enumerate(visit_rows):
        v = int(item["visit"])
        sw = int(item["sweep"])
        stem = str(item["stem"])

        tid, t_tr, t_iou, t_jump, t_unc, t_reason = t_res[idx]
        bid, b_tr, b_iou, b_jump, b_unc, b_reason = b_res[idx]

        t_mask = t_masks[idx]
        b_mask = (b_inst_list[idx] == bid) if bid > 0 else np.zeros_like(b_inst_list[idx], dtype=bool)

        t_vox = int(t_mask.sum())
        b_vox = int(b_mask.sum())
        faces = count_interface_faces(t_mask, b_mask) if (tid > 0 and bid > 0) else 0
        contact_area_um2 = round(float(faces) * FACE_AREA_UM2, 3)
        dye_vox = int((b_mask & dye_list[idx]).sum()) if bid > 0 else 0

        unc_flag = int(t_unc or b_unc)
        reasons = [r for r in (t_reason, b_reason) if r]
        rows_out.append(
            {
                "chamber": chamber_id,
                "short_exp": short_exp,
                "experiment": exp,
                "roi": roi,
                "source": source,
                "visit": v,
                "sweep": sw,
                "stem": stem,
                "tcell_track_id": int(t_tr),
                "target_track_id": int(b_tr),
                "tracked_t_id": int(tid),
                "tracked_b_id": int(bid),
                "contact_area_um2": contact_area_um2,
                "target_vol_um3": round(b_vox * VOXEL_VOL_UM3, 3),
                "tcell_vol_um3": round(t_vox * VOXEL_VOL_UM3, 3),
                "dye_voxels_in_target": dye_vox,
                "uncertain_flag": unc_flag,
                "uncertain_reason": ";".join(reasons),
                "tcell_iou_prev": t_iou,
                "tcell_jump_px": t_jump,
                "target_iou_prev": b_iou,
                "target_jump_px": b_jump,
                "chamber_runtime_s": round(elapsed, 3),
            }
        )

    df = pd.DataFrame(rows_out)
    if out_dir is None:
        out_dir = REPO_ROOT / "data" / "pilot_seg" / "p5_tracks"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{chamber_id}.csv"
    df.to_csv(csv_path, index=False)
    return df


def run_all_chambers(source: str = "C4") -> pd.DataFrame:
    """Run track_chamber on all 20 chambers (PILOT_CHAMBERS + PILOT4_CHAMBERS)."""
    dfs: List[pd.DataFrame] = []
    for short_exp, exp, roi in ALL_20_CHAMBERS:
        t0 = time.perf_counter()
        df = track_chamber(short_exp, exp, roi, source=source)
        dt = time.perf_counter() - t0
        c_id = f"{short_exp}_{roi}"
        n_t_tr = len(set(df["tcell_track_id"][df["tcell_track_id"] > 0]))
        n_b_tr = len(set(df["target_track_id"][df["target_track_id"] > 0]))
        n_unc = int(df["uncertain_flag"].sum())
        n_no_b = int((df["target_track_id"] == 0).sum())
        print(
            f"[{c_id}] ({len(df)} visits, {dt:.2f}s) -> "
            f"t_tracks={n_t_tr}, b_tracks={n_b_tr}, uncertain={n_unc}, no_target={n_no_b}"
        )
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    out_all = REPO_ROOT / "data" / "pilot_seg" / "p5_tracks" / f"all_p5_tracks_{source}.csv"
    all_df.to_csv(out_all, index=False)
    return all_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 5 cell selection and tracking.")
    parser.add_argument("--source", type=str, default="C4", help="Instance source directory (C4 or C5).")
    args = parser.parse_args()
    run_all_chambers(source=args.source)
