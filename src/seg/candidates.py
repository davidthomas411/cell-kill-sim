"""Candidate 3D instance segmentation methods (C0, C1, C1s, C3) for CAR-T + target chambers.

Voxel pitch assumption:
  xy = 0.1962 um, z treated as 0.1962 um (nominal isotropic anisotropy = 1.0).
  Voxel volume = 0.1962^3 = 0.007552625848 um^3.
  Voxel face area = 0.1962^2 = 0.03849444 um^2 (~0.0385 um^2).
  Sweep 6 has SiR-tubulin in ch2 and must never be used.

Candidates (Pilot P2):
  C0:  Lab masks as deposited (instances via step2_instance_seg.m / make_gallery.py
       without 3D hole repair, preserving raw deposited topology and cavities).
  C1:  Lab masks repaired: 3D fill-holes (fill_solid_cell_mask) and distance-transform
       seeded watershed on the boundary class (classes 3/4) to split merged targets.
  C1s: C1 + raw-signal check: builds a smoothed (sigma=1.0), median+k*MAD thresholded,
       slice/3D hole-filled signal mask on ch1 (k=5.0) and ch2 (k=6.5), drops connected
       parts (2D Z-slice components and 3D watershed lobes) with signal support < 0.25,
       and keeps the largest 3D connected component per cell without trimming boundaries.
  C3:  PlantSeg (generic_light_sheet_3D_unet + interior-seeded watershed) on ch2 and ch1
       within the 3D-filled raw fluorescence envelope, with the outer-halo slice-overwrite
       bug fixed.
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    generate_binary_structure,
    label,
)
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.death_labels import instance_seg_matlab_xyz
from src.mask_qc import bbox_slices, fill_solid_cell_mask
from src.reproduce_contact_area import STRUCT26
from src.review.make_clean_gallery import extract_exact_visit_masks

VOXEL_UM = 0.1962
ANISOTROPY = 1.0
VOXEL_VOL_UM3 = VOXEL_UM ** 3  # 0.007552625848 um^3
FACE_AREA_UM2 = 0.0385

# Signal mask & C1s pruning parameters
SIGNAL_SIGMA_PX = 1.0
SIGNAL_K_TCELL = 5.0    # ch1 (Lifeact-EGFP): bg + 5.0 * (1.4826 * MAD)
SIGNAL_K_TARGET = 6.5   # ch2 (mApple-Mem): bg + 6.5 * (1.4826 * MAD); rejects T-cell ch2 bleed & PSF tails
SIGNAL_SUPPORT_THRESH = 0.25

PILOT_CHAMBERS: List[Tuple[str, str, str]] = [
    ("v01", "v01_6_40_120_0-6_hd1", "MIP_ROI_2"),
    ("v01", "v01_6_40_120_0-6_hd1", "MIP_ROI_32"),
    ("v02", "v02_6_40_120_0-6_hd3", "MIP_ROI_3"),
    ("v02", "v02_6_40_120_0-6_hd3", "MIP_ROI_32"),
    ("v03", "v03_6_40_120_0-6_hd4", "MIP_ROI_6"),
    ("v03", "v03_6_40_120_0-6_hd4", "MIP_ROI_30"),
    ("v04", "v04_6_40_120_0-6_hd6", "MIP_ROI_24"),
    ("v05", "v05_6_40_120_0-6_hd7", "MIP_ROI_8"),
]


def _relabel_sequential(inst: np.ndarray, min_size: int = 1) -> np.ndarray:
    """Relabel non-zero instance IDs sequentially 1..K in descending size order."""
    out = np.zeros(inst.shape, dtype=np.uint16)
    uids, counts = np.unique(inst, return_counts=True)
    pairs = [(int(c), int(u)) for u, c in zip(uids, counts) if u > 0 and c >= min_size]
    pairs.sort(reverse=True)
    for new_id, (_, old_id) in enumerate(pairs, start=1):
        out[inst == old_id] = new_id
    return out


def _active_crop_slice(
    fg: np.ndarray,
    margin: int = 10,
    min_comp_voxels: int = 2000,
    align16: bool = True,
) -> Tuple[slice, slice, slice]:
    """Return 3D bounding box slice around foreground components >= min_comp_voxels.

    When align16 is True, rounds each dimension <= 128 up to a multiple of 16 (min 32)
    so 3D U-Net inference runs in a single aligned patch where possible.
    """
    lab, _ = label(fg, structure=STRUCT26)
    counts = np.bincount(lab.ravel())
    keep_ids = np.where(counts >= min_comp_voxels)[0]
    keep_ids = keep_ids[keep_ids > 0]
    if keep_ids.size > 0:
        clean = np.isin(lab, keep_ids)
    else:
        clean = fg
    nz = np.argwhere(clean)
    if nz.size == 0:
        return (slice(0, fg.shape[0]), slice(0, fg.shape[1]), slice(0, fg.shape[2]))
    lo = np.maximum(0, nz.min(axis=0) - margin)
    hi = np.minimum(fg.shape, nz.max(axis=0) + margin + 1)
    if not align16:
        return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]

    min_spans = (32, 64, 64)
    slices: List[slice] = []
    for d in range(3):
        span = int(hi[d] - lo[d])
        min_s = min_spans[d]
        if span <= 128:
            target_span = min(128, max(min_s, ((span + 15) // 16) * 16))
        else:
            target_span = span
        target_span = min(int(fg.shape[d]), target_span)
        extra = target_span - span
        if extra > 0:
            l_idx = max(0, int(lo[d]) - extra // 2)
            h_idx = min(int(fg.shape[d]), l_idx + target_span)
            l_idx = max(0, h_idx - target_span)
        else:
            l_idx, h_idx = int(lo[d]), int(hi[d])
        slices.append(slice(l_idx, h_idx))
    return (slices[0], slices[1], slices[2])


def build_signal_mask(
    vol: np.ndarray,
    k: float,
    close_iter: int = 2,
    min_vox: int = 250,
    sigma: float = SIGNAL_SIGMA_PX,
) -> np.ndarray:
    """Build a solid 3D raw-signal mask for a fluorescence channel.

    1. Smooth with 3D Gaussian (sigma = 1.0 px).
    2. Estimate background (median) and noise (1.4826 * MAD) of the smoothed volume.
    3. Threshold at bg + k * noise and drop isolated specks < min_vox voxels.
    4. Close small membrane gaps (close_iter iterations) inside the tight bounding box
       and fill holes slice-wise (Z, Y, X) and in 3D via fill_solid_cell_mask so that
       membrane-labelled target cells (ch2) have their interior cytoplasm counted as solid
       signal without inflating the outer boundary.
    """
    sm = gaussian_filter(vol.astype(np.float32), sigma=sigma)
    sm_sub = sm[::2, ::2, ::2]
    bg = float(np.median(sm_sub))
    mad = float(np.median(np.abs(sm_sub - bg)))
    noise = 1.4826 * mad
    fg = sm > (bg + k * noise)

    lab, _ = label(fg, structure=STRUCT26)
    cnt = np.bincount(lab.ravel())
    keep = np.where(cnt >= min_vox)[0]
    keep = keep[keep > 0]
    if keep.size == 0:
        return np.zeros(vol.shape, dtype=bool)
    fg = np.isin(lab, keep)

    sl = bbox_slices(fg, close_iter + 2, fg.shape)
    fg_sub = fg[sl]
    if close_iter > 0:
        closed = binary_closing(fg_sub, structure=STRUCT26, iterations=close_iter)
    else:
        closed = fg_sub
    filled = fill_solid_cell_mask(closed)
    interior = filled & (~closed)
    solid_sub = fg_sub | fill_solid_cell_mask(fg_sub | interior)

    out = np.zeros(vol.shape, dtype=bool)
    out[sl] = solid_sub
    return out


def prune_unsupported_parts(
    inst: np.ndarray,
    sig_mask: np.ndarray,
    support_thresh: float = SIGNAL_SUPPORT_THRESH,
    min_size: int = 1500,
) -> np.ndarray:
    """Remove from each cell instance any connected part with signal support < support_thresh,
    then keep the largest 26-connected 3D component per cell without voxel-by-voxel erosion.
    """
    out = np.zeros(inst.shape, dtype=np.uint16)
    struct2d = np.ones((3, 3), dtype=bool)

    for uid in np.unique(inst):
        if uid == 0:
            continue
        cell_full = inst == uid
        n_tot = int(cell_full.sum())
        if n_tot < min_size:
            continue
        if float(np.sum(cell_full & sig_mask)) / float(n_tot) < support_thresh:
            continue

        sl = bbox_slices(cell_full, 2, inst.shape)
        cell = cell_full[sl].copy()
        sig = sig_mask[sl]

        # 1. Drop 2D Z-slice connected components with signal support < support_thresh
        #    (severs out-of-focus phantom Z-lobes floating above/below the cell in XZ)
        work = np.zeros_like(cell, dtype=bool)
        for z in range(cell.shape[0]):
            if not np.any(cell[z]):
                continue
            lab_z, n_z = label(cell[z], structure=struct2d)
            for cid in range(1, n_z + 1):
                comp = lab_z == cid
                c_sz = int(comp.sum())
                if c_sz == 0:
                    continue
                if float(np.sum(comp & sig[z])) / float(c_sz) >= support_thresh:
                    work[z] |= comp

        if int(work.sum()) < min_size:
            continue

        # 2. Decompose into 3D distance-transform watershed lobes and drop unsupported lobes
        dt = distance_transform_edt(work)
        dt_sm = gaussian_filter(dt, sigma=1.5)
        pks = peak_local_max(
            dt_sm,
            min_distance=8,
            threshold_abs=2.5,
            labels=work.astype(np.int32),
        )
        if len(pks) > 1:
            seeds = np.zeros(work.shape, dtype=np.int32)
            for idx_p, pk in enumerate(pks, start=1):
                seeds[tuple(int(x) for x in pk)] = idx_p
            ws = watershed(-dt, seeds, mask=work)
            kept_lobes = np.zeros_like(work, dtype=bool)
            for lid in range(1, len(pks) + 1):
                lobe = ws == lid
                l_sz = int(lobe.sum())
                if l_sz == 0:
                    continue
                if float(np.sum(lobe & sig)) / float(l_sz) >= support_thresh:
                    kept_lobes |= lobe
            work = kept_lobes

        if int(work.sum()) < min_size:
            continue

        # 3. Keep largest 26-connected 3D component per cell and fill internal holes
        cc_lab, n_cc = label(work, structure=STRUCT26)
        if n_cc == 0:
            continue
        cc_cnt = np.bincount(cc_lab.ravel())
        cc_cnt[0] = 0
        dom_id = int(np.argmax(cc_cnt))
        if int(cc_cnt[dom_id]) < min_size:
            continue
        dom = fill_solid_cell_mask(cc_lab == dom_id)
        out[sl][dom] = int(uid)

    return _relabel_sequential(out, min_size=min_size)


def run_c0(pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """C0: Lab masks as deposited (step2_instance_seg.m via instance_seg_matlab_xyz).

    Preserves raw deposited 3D instance masks (with imfill_holes_26 only, without
    2D slice hole repair) so cavities and merges in the deposited masks are measured.
    """
    t0 = time.perf_counter()
    t_inst = np.zeros(pred.shape, dtype=np.uint16)
    b_inst = np.zeros(pred.shape, dtype=np.uint16)
    fg_t = (pred == 1) | (pred == 3)
    if np.any(fg_t):
        sl_t = bbox_slices(fg_t, 3, pred.shape)
        sub_p = pred[sl_t]
        t_raw = instance_seg_matlab_xyz(sub_p == 1, sub_p == 3)
        t_inst[sl_t] = np.where((sub_p == 1) | (sub_p == 3), t_raw, 0).astype(np.uint16)
    fg_b = (pred == 2) | (pred == 4)
    if np.any(fg_b):
        sl_b = bbox_slices(fg_b, 3, pred.shape)
        sub_p = pred[sl_b]
        b_raw = instance_seg_matlab_xyz(sub_p == 2, sub_p == 4)
        b_inst[sl_b] = np.where((sub_p == 2) | (sub_p == 4), b_raw, 0).astype(np.uint16)
    dt = time.perf_counter() - t0
    return t_inst, b_inst, dt


def _split_and_fill_class(
    pred: np.ndarray,
    interior_cls: int,
    boundary_cls: int,
    min_comp_size: int = 3000,
    min_peak_dist: int = 18,
    min_peak_abs: float = 7.2,
    min_core_voxels: int = 750,
    split_merged: bool = True,
) -> np.ndarray:
    """3D fill-holes and split merged instances via distance-transform seeded watershed."""
    raw_fg = (pred == interior_cls) | (pred == boundary_cls)
    raw_lab, _ = label(raw_fg, structure=STRUCT26)
    raw_cnt = np.bincount(raw_lab.ravel())
    keep = np.where(raw_cnt >= min_comp_size)[0]
    keep = keep[keep > 0]
    if keep.size == 0:
        return np.zeros(pred.shape, dtype=np.uint16)

    clean_fg = np.isin(raw_lab, keep)
    sl = bbox_slices(clean_fg, 10, pred.shape)
    sub_clean = clean_fg[sl]
    sub_pred = pred[sl]

    # Fill holes per 26-connected component so diagonal contacts do not bridge disjoint components
    raw_sub_lab, n_raw_sub = label(sub_clean, structure=STRUCT26)
    filled_fg = np.zeros_like(sub_clean, dtype=bool)
    for rcid in range(1, n_raw_sub + 1):
        rc_mask = raw_sub_lab == rcid
        s_rc = bbox_slices(rc_mask, 2, filled_fg.shape)
        filled_fg[s_rc] |= fill_solid_cell_mask(rc_mask[s_rc])

    if not split_merged:
        comp_lab, n_comp = label(filled_fg, structure=STRUCT26)
        out_sub = np.zeros(sub_pred.shape, dtype=np.uint16)
        for cid in range(1, n_comp + 1):
            cmask = comp_lab == cid
            if int(cmask.sum()) >= min_comp_size:
                s_c = bbox_slices(cmask, 2, out_sub.shape)
                out_sub[s_c][fill_solid_cell_mask(cmask[s_c])] = cid
        out = np.zeros(pred.shape, dtype=np.uint16)
        out[sl] = out_sub
        return _relabel_sequential(out, min_size=min_comp_size // 2)

    dt_raw = distance_transform_edt(filled_fg)
    dt_smooth = gaussian_filter(dt_raw, sigma=2.5)
    # Use 6-connectivity to match skimage.segmentation.watershed default connectivity=1
    struct6 = generate_binary_structure(3, 1)
    comp_lab, n_comp = label(filled_fg, structure=struct6)
    core_lab, _ = label(filled_fg & (sub_pred != boundary_cls), structure=STRUCT26)
    core_cnt = np.bincount(core_lab.ravel())

    seeds = np.zeros(sub_pred.shape, dtype=np.int32)
    next_id = 1
    for cid in range(1, n_comp + 1):
        cmask = comp_lab == cid
        if int(cmask.sum()) < min_comp_size:
            continue
        pks = peak_local_max(
            dt_smooth,
            min_distance=min_peak_dist,
            threshold_abs=min_peak_abs,
            labels=cmask.astype(np.int32),
        )
        if len(pks) == 0:
            pks = [np.unravel_index(int(np.argmax(dt_smooth * cmask)), dt_smooth.shape)]
        for pk in pks:
            pk_t = tuple(int(x) for x in pk)
            c_id = int(core_lab[pk_t])
            if c_id > 0 and core_cnt[c_id] >= min_core_voxels and np.all(seeds[core_lab == c_id] == 0):
                seeds[core_lab == c_id] = next_id
            else:
                seeds[pk_t] = next_id
            next_id += 1

        # Ensure every interior core (pred == interior_cls, >=1500 vox) inside cmask gets a seed
        int_cores, n_ic = label(cmask & (sub_pred == interior_cls), structure=STRUCT26)
        for ic_id in range(1, n_ic + 1):
            ic_mask = int_cores == ic_id
            if int(ic_mask.sum()) >= 1500 and np.all(seeds[ic_mask] == 0):
                seeds[ic_mask] = next_id
                next_id += 1

    # Watershed on boundary class: boundary voxels (class 3/4) have higher traversal cost
    cost = -dt_raw + 4.0 * (sub_pred == boundary_cls).astype(np.float32)
    ws = watershed(cost, seeds, mask=filled_fg)

    out_sub = np.zeros(sub_pred.shape, dtype=np.uint16)
    for k in range(1, next_id):
        inst_mask = ws == k
        if int(inst_mask.sum()) >= 1500:
            s_inst = bbox_slices(inst_mask, 2, out_sub.shape)
            filled_inst = fill_solid_cell_mask(inst_mask[s_inst])
            out_sub[s_inst][inst_mask[s_inst] | (filled_inst & (out_sub[s_inst] == 0) & (ws[s_inst] == 0))] = k
    out = np.zeros(pred.shape, dtype=np.uint16)
    out[sl] = out_sub
    return _relabel_sequential(out, min_size=1500)


def run_c1(pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """C1: Lab masks repaired (3D fill-holes + distance-transform watershed on boundary class)."""
    t0 = time.perf_counter()
    t_inst = _split_and_fill_class(
        pred,
        interior_cls=1,
        boundary_cls=3,
        min_comp_size=2500,
        min_peak_dist=20,
        min_peak_abs=8.0,
        min_core_voxels=1500,
        split_merged=False,
    )
    b_inst = _split_and_fill_class(
        pred,
        interior_cls=2,
        boundary_cls=4,
        min_comp_size=3000,
        min_peak_dist=18,
        min_peak_abs=7.2,
        min_core_voxels=750,
        split_merged=True,
    )
    dt = time.perf_counter() - t0
    return t_inst, b_inst, dt


def run_c1s(
    pred: np.ndarray,
    raw: np.ndarray,
    sig_t: Optional[np.ndarray] = None,
    sig_b: Optional[np.ndarray] = None,
    c1_res: Optional[Tuple[np.ndarray, np.ndarray, float]] = None,
    support_thresh: float = SIGNAL_SUPPORT_THRESH,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """C1s = C1 + raw-signal check.

    Builds filled raw-signal masks on ch1 (k=5.0) and ch2 (k=6.5), drops connected parts
    with signal support < support_thresh (0.25), and keeps the largest connected component
    per cell without eroding cell boundaries.
    """
    t0 = time.perf_counter()
    if c1_res is not None:
        t_c1, b_c1, dt_c1 = c1_res
    else:
        t_c1, b_c1, dt_c1 = run_c1(pred)
        t0 = time.perf_counter()

    if sig_t is None:
        sig_t = build_signal_mask(raw[1], k=SIGNAL_K_TCELL, close_iter=1, min_vox=250)
    if sig_b is None:
        sig_b = build_signal_mask(raw[2], k=SIGNAL_K_TARGET, close_iter=2, min_vox=250)

    t_inst = prune_unsupported_parts(t_c1, sig_t, support_thresh=support_thresh, min_size=1250)
    b_inst = prune_unsupported_parts(b_c1, sig_b, support_thresh=support_thresh, min_size=1500)
    dt_prune = time.perf_counter() - t0
    return t_inst, b_inst, float(dt_c1 + dt_prune)


def _plantseg_channel(
    vol_ch: np.ndarray,
    fg_hint: np.ndarray,
    device: str = "mps",
    min_peak_dist: int = 18,
    min_peak_dt: float = 5.5,
    min_size: int = 2500,
    otsu_factor: float = 0.68,
    pm_seed_thresh: float = 0.65,
) -> np.ndarray:
    """Run PlantSeg 3D U-Net + interior-seeded watershed on a single fluorescence channel.

    Fixes the P1 C3 striped-rectangle and merged-target bugs:
      1. Suppresses outer cell-to-background boundary probability near the exterior of env
         (dt_env <= 5.0) so PlantSeg's boundary prediction only acts on internal cell-cell
         interfaces and does not split a cell's outer PSF halo into a hollow concentric ring.
      2. Seeds watershed from interior distance-transform maxima of env & (pm_internal < pm_seed_thresh).
      3. Fills internal 3D holes per instance via 3D binary_fill_holes restricted to unassigned
         background voxels ((out_sub == 0) & (ws_clean == 0)) instead of 2D slice-wise
         fill_solid_cell_mask + unconditional overwrite.
    """
    from plantseg.functionals.prediction.prediction import unet_prediction

    sl = _active_crop_slice(fg_hint, margin=10, min_comp_voxels=2000, align16=True)
    sub = vol_ch[sl].astype(np.float32)

    # Compute 3D filled outer envelope from raw fluorescence above camera read noise
    vol_sub = vol_ch[::2, ::2, ::2]
    med = float(np.median(vol_sub))
    mad = float(np.median(np.abs(vol_sub - med)))
    sm = gaussian_filter(sub, sigma=1.2)
    sm_clip = np.clip(sm, None, float(np.percentile(sm, 99.5)))
    above = sm_clip[sm_clip > (med + 3.0 * 1.4826 * mad)]
    if above.size > 100:
        otsu = float(threshold_otsu(above))
    else:
        otsu = med + 4.0 * 1.4826 * mad
    th = med + otsu_factor * (otsu - med)
    wall = sm > th
    wlab, _ = label(wall, structure=STRUCT26)
    wcnt = np.bincount(wlab.ravel())
    keep_w = np.where(wcnt >= 500)[0]
    keep_w = keep_w[keep_w > 0]
    if keep_w.size > 0:
        wall = np.isin(wlab, keep_w)
    env = fill_solid_cell_mask(binary_closing(wall, structure=STRUCT26, iterations=1))

    min_p = (32, 64, 64)
    patch = tuple(min(112, max(min_p[i], (d // 16) * 16)) for i, d in enumerate(sub.shape))
    pm = unet_prediction(
        raw=sub,
        input_layout="ZYX",
        model_name="generic_light_sheet_3D_unet",
        model_id=None,
        patch=patch,
        patch_halo=(8, 16, 16),
        single_batch_mode=True,
        device=device,
        disable_tqdm=True,
    )
    if pm.ndim == 4:
        pm = pm[0]
    if device == "mps":
        torch.mps.empty_cache()
    gc.collect()

    dt_env = distance_transform_edt(env)
    pm_internal = np.where(dt_env > 5.0, pm, 0.0)
    wall_int = pm_internal >= pm_seed_thresh
    if np.any(wall_int):
        wi_lab, _ = label(wall_int, structure=STRUCT26)
        wi_cnt = np.bincount(wi_lab.ravel())
        keep_wi = np.where(wi_cnt >= 250)[0]
        keep_wi = keep_wi[keep_wi > 0]
        wall_int = np.isin(wi_lab, keep_wi) if keep_wi.size > 0 else np.zeros_like(wall_int)
    interior = env & (~wall_int)
    dt_int = distance_transform_edt(interior)
    dt_sm = gaussian_filter(dt_int, sigma=2.5)

    comp_lab, n_comp = label(env, structure=STRUCT26)
    seeds = np.zeros(sub.shape, dtype=np.int32)
    next_id = 1
    for cid in range(1, n_comp + 1):
        cmask = comp_lab == cid
        if int(cmask.sum()) < min_size:
            continue
        pks = peak_local_max(
            dt_sm,
            min_distance=min_peak_dist,
            threshold_abs=min_peak_dt,
            labels=cmask.astype(np.int32),
        )
        if len(pks) == 0:
            pks = [np.unravel_index(int(np.argmax(dt_sm * cmask)), sub.shape)]
        for pk in pks:
            seeds[tuple(int(x) for x in pk)] = next_id
            next_id += 1

    cost = -dt_env.astype(np.float32) + 12.0 * pm_internal.astype(np.float32)
    ws = watershed(cost, seeds, mask=env)
    ws_clean = _relabel_sequential(ws.astype(np.uint16), min_size=min_size // 2)

    out_sub = np.zeros(sub.shape, dtype=np.uint16)
    for uid in np.unique(ws_clean):
        if uid == 0:
            continue
        cc_lab, n_cc = label(ws_clean == uid, structure=STRUCT26)
        if n_cc > 1:
            cc_cnt = np.bincount(cc_lab.ravel())
            cc_cnt[0] = 0
            dom = cc_lab == int(np.argmax(cc_cnt))
        else:
            dom = ws_clean == uid
        dom_filled = binary_fill_holes(dom, structure=STRUCT26)
        write_mask = dom | (dom_filled & (out_sub == 0) & (ws_clean == 0))
        out_sub[write_mask] = int(uid)

    out = np.zeros(vol_ch.shape, dtype=np.uint16)
    out[sl] = out_sub
    return _relabel_sequential(out, min_size=min_size // 2)


def run_c3(
    raw: np.ndarray,
    pred: np.ndarray,
    device: str = "mps",
    sig_t: Optional[np.ndarray] = None,
    sig_b: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """C3 (fixed): PlantSeg (generic_light_sheet_3D_unet + interior watershed) on ch1 and ch2."""
    t0 = time.perf_counter()
    hint_t = (pred == 1) | (pred == 3)
    if sig_t is not None:
        hint_t_sig = hint_t & sig_t
        hint_t = hint_t_sig if int(hint_t_sig.sum()) >= 2000 else (hint_t | sig_t)
    hint_b = (pred == 2) | (pred == 4)
    if sig_b is not None:
        hint_b_sig = hint_b & sig_b
        hint_b = hint_b_sig if int(hint_b_sig.sum()) >= 2000 else (hint_b | sig_b)

    t_inst = _plantseg_channel(
        raw[1],
        fg_hint=hint_t,
        device=device,
        min_peak_dist=25,
        min_peak_dt=9.5,
        min_size=2500,
        otsu_factor=0.68,
        pm_seed_thresh=0.95,
    )
    b_inst = _plantseg_channel(
        raw[2],
        fg_hint=hint_b,
        device=device,
        min_peak_dist=18,
        min_peak_dt=5.5,
        min_size=3000,
        otsu_factor=0.68,
        pm_seed_thresh=0.65,
    )
    dt = time.perf_counter() - t0
    return t_inst, b_inst, dt


def select_tracked_pair(
    t_inst: np.ndarray,
    b_inst: np.ndarray,
    t_ref: np.ndarray,
    b_ref: np.ndarray,
) -> Tuple[int, int]:
    """Identify tracked T cell and contacted target by best overlap with lab's tracked cells.

    For targets, if the lab's reference mask b_ref merges two touching targets,
    weighting overlap by interface contact with the tracked T cell disambiguates
    the actually contacted target cell.
    """
    t_ref_bool = t_ref > 0
    b_ref_bool = b_ref > 0

    best_t_id = 0
    best_t_ov = 0
    for tid in np.unique(t_inst):
        if tid == 0:
            continue
        ov = int(np.sum((t_inst == tid) & t_ref_bool))
        if ov > best_t_ov:
            best_t_ov = ov
            best_t_id = int(tid)
    if best_t_id == 0 and np.any(t_inst > 0):
        best_t_id = 1

    # Dilate tracked T cell (or reference T cell if candidate T cell is empty) to score contact
    t_anchor = (t_inst == best_t_id) if best_t_id > 0 else t_ref_bool
    t_dil = binary_dilation(t_anchor, structure=STRUCT26, iterations=2)

    best_b_id = 0
    best_b_score = -1.0
    for bid in np.unique(b_inst):
        if bid == 0:
            continue
        bmask = b_inst == bid
        ov_ref = int(np.sum(bmask & b_ref_bool))
        contact_vox = int(np.sum(bmask & t_dil))
        if ov_ref == 0 and contact_vox == 0:
            continue
        # Overlap with lab's tracked target + contact with tracked T cell
        score = float(ov_ref) + 25.0 * float(contact_vox)
        if score > best_b_score:
            best_b_score = score
            best_b_id = int(bid)
    if best_b_id == 0 and np.any(b_inst > 0) and np.any(b_ref_bool):
        # Fallback to largest instance overlapping dilated reference
        b_ref_dil = binary_dilation(b_ref_bool, structure=STRUCT26, iterations=3)
        for bid in np.unique(b_inst):
            if bid == 0:
                continue
            ov = int(np.sum((b_inst == bid) & b_ref_dil))
            if ov > best_b_score:
                best_b_score = float(ov)
                best_b_id = int(bid)

    return best_t_id, best_b_id


def count_interface_faces(mask_a: np.ndarray, mask_b: np.ndarray) -> int:
    """Count shared 6-connected voxel faces between two 3D binary masks.

    If masks are separated by a 1-voxel boundary gap, counts faces between
    1-voxel 6-connected dilation of mask_a and mask_b.
    """
    if not np.any(mask_a) or not np.any(mask_b):
        return 0

    def _direct_faces(a: np.ndarray, b: np.ndarray) -> int:
        a_only = a & (~b)
        b_only = b
        faces = 0
        for axis in range(3):
            s0 = [slice(None)] * 3
            s1 = [slice(None)] * 3
            s0[axis] = slice(0, -1)
            s1[axis] = slice(1, None)
            t0, t1 = tuple(s0), tuple(s1)
            faces += int(np.sum(a_only[t0] & b_only[t1]))
            faces += int(np.sum(b_only[t0] & a_only[t1]))
        return faces

    direct = _direct_faces(mask_a, mask_b)
    if direct > 0:
        return direct
    struct6 = generate_binary_structure(3, 1)
    return _direct_faces(binary_dilation(mask_a, structure=struct6, iterations=1), mask_b)


def compute_signal_agreement(
    mask: np.ndarray,
    sig_mask: np.ndarray,
    sig_lab: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Compute (precision, recall) of a 3D cell mask against the filled channel signal mask.

    - precision: fraction of the cell's mask voxels inside the filled signal mask.
    - recall: fraction of the filled signal component(s) overlapping that cell that the mask covers.
    """
    m_vox = int(np.sum(mask))
    if m_vox == 0:
        return 0.0, 0.0
    tp = int(np.sum(mask & sig_mask))
    precision = float(tp) / float(m_vox)
    if tp == 0:
        return round(precision, 4), 0.0

    if sig_lab is None:
        sig_lab, _ = label(sig_mask, structure=STRUCT26)
    ov_ids = np.unique(sig_lab[mask & (sig_lab > 0)])
    if ov_ids.size == 0:
        return round(precision, 4), 0.0
    sig_ov_vox = int(np.sum(np.isin(sig_lab, ov_ids)))
    recall = float(tp) / float(sig_ov_vox) if sig_ov_vox > 0 else 0.0
    return round(precision, 4), round(recall, 4)


def compute_candidate_metrics(
    t_inst: np.ndarray,
    b_inst: np.ndarray,
    tracked_t_id: int,
    tracked_b_id: int,
    runtime_s: float,
    sig_t: Optional[np.ndarray] = None,
    sig_b: Optional[np.ndarray] = None,
    sig_t_lab: Optional[np.ndarray] = None,
    sig_b_lab: Optional[np.ndarray] = None,
    ref_target_cells: Optional[List[np.ndarray]] = None,
) -> Dict[str, float | int]:
    """Compute per-visit segmentation and raw-signal agreement metrics for a candidate."""
    t_mask = (t_inst == tracked_t_id) if tracked_t_id > 0 else np.zeros(t_inst.shape, dtype=bool)
    b_mask = (b_inst == tracked_b_id) if tracked_b_id > 0 else np.zeros(b_inst.shape, dtype=bool)

    t_vox = int(np.sum(t_mask))
    b_vox = int(np.sum(b_mask))

    if t_vox > 0:
        s_t = bbox_slices(t_mask, 2, t_mask.shape)
        t_filled_vox = int(np.sum(fill_solid_cell_mask(t_mask[s_t])))
    else:
        t_filled_vox = 0
    if b_vox > 0:
        s_b = bbox_slices(b_mask, 2, b_mask.shape)
        b_filled_vox = int(np.sum(fill_solid_cell_mask(b_mask[s_b])))
    else:
        b_filled_vox = 0

    t_holes_vox = max(0, t_filled_vox - t_vox)
    b_holes_vox = max(0, b_filled_vox - b_vox)

    n_t_instances = int(len(np.unique(t_inst)) - (1 if 0 in t_inst else 0))
    n_b_instances = int(len(np.unique(b_inst)) - (1 if 0 in b_inst else 0))
    _, n_t_cc = label(t_mask, structure=STRUCT26) if t_vox > 0 else (None, 0)
    _, n_b_cc = label(b_mask, structure=STRUCT26) if b_vox > 0 else (None, 0)

    faces = count_interface_faces(t_mask, b_mask)
    contact_area_um2 = float(faces) * FACE_AREA_UM2

    if sig_b is not None:
        b_prec, b_rec = compute_signal_agreement(b_mask, sig_b, sig_b_lab)
    else:
        b_prec, b_rec = 0.0, 0.0
    if sig_t is not None:
        t_prec, t_rec = compute_signal_agreement(t_mask, sig_t, sig_t_lab)
    else:
        t_prec, t_rec = 0.0, 0.0

    # Check if tracked target b_mask merges >1 distinct real target cell in the chamber
    merged_gt1 = 0
    if b_vox > 0 and ref_target_cells is not None and len(ref_target_cells) >= 2:
        n_cov = 0
        for rc in ref_target_cells:
            rc_vox = int(np.sum(rc))
            if rc_vox > 0 and (float(np.sum(b_mask & rc)) / float(rc_vox)) >= 0.25:
                n_cov += 1
        if n_cov >= 2:
            merged_gt1 = 1

    mean_agree = round(0.25 * (b_prec + b_rec + t_prec + t_rec), 4)

    return {
        "target_precision": b_prec,
        "target_recall": b_rec,
        "tcell_precision": t_prec,
        "tcell_recall": t_rec,
        "mean_agreement": mean_agree,
        "target_merged_gt1": int(merged_gt1),
        "target_vol_um3": round(b_vox * VOXEL_VOL_UM3, 3),
        "tcell_vol_um3": round(t_vox * VOXEL_VOL_UM3, 3),
        "target_vox": b_vox,
        "tcell_vox": t_vox,
        "n_target_instances": n_b_instances,
        "n_tcell_instances": n_t_instances,
        "target_n_cc": int(n_b_cc),
        "tcell_n_cc": int(n_t_cc),
        "target_holes_um3": round(b_holes_vox * VOXEL_VOL_UM3, 3),
        "tcell_holes_um3": round(t_holes_vox * VOXEL_VOL_UM3, 3),
        "holes_um3": round((b_holes_vox + t_holes_vox) * VOXEL_VOL_UM3, 3),
        "holes_vox": int(b_holes_vox + t_holes_vox),
        "contact_faces": int(faces),
        "contact_area_um2": round(contact_area_um2, 3),
        "runtime_s": round(float(runtime_s), 4),
    }


def save_candidate_npz(
    out_path: Path,
    t_inst: np.ndarray,
    b_inst: np.ndarray,
    tracked_t_id: int,
    tracked_b_id: int,
    runtime_s: float,
) -> None:
    """Save compressed uint16 candidate masks and tracked IDs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t_mask = ((t_inst == tracked_t_id).astype(np.uint16) if tracked_t_id > 0 else np.zeros_like(t_inst, dtype=np.uint16))
    b_mask = ((b_inst == tracked_b_id).astype(np.uint16) if tracked_b_id > 0 else np.zeros_like(b_inst, dtype=np.uint16))
    np.savez_compressed(
        out_path,
        t_inst=t_inst.astype(np.uint16),
        b_inst=b_inst.astype(np.uint16),
        t_mask=t_mask,
        b_mask=b_mask,
        tracked_t_id=np.uint16(tracked_t_id),
        tracked_b_id=np.uint16(tracked_b_id),
        runtime_s=np.float32(runtime_s),
    )


def get_reference_masks_for_stems(
    exp: str,
    roi: str,
    stems: List[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Extract reference tracked T-cell and target masks using make_clean_gallery.py's
    extract_exact_visit_masks and step3_tracked/<exp>/per_sweep.csv.

    If the deposited prediction mask loses the target in late visits, propagates the
    most recent non-empty reference mask so raw-based candidates (e.g. C3) can still
    match the tracked cell.
    """
    manifest_df = pd.read_csv(REPO_ROOT / "data" / "scidb_19235_raw" / "manifest.csv")
    m_sub = manifest_df[(manifest_df["experiment"] == exp) & (manifest_df["roi"] == roi)]
    stem_to_vs: Dict[str, Tuple[int, int]] = {}
    for _, r in m_sub.iterrows():
        stem_to_vs[f"{int(r.stem):05d}"] = (int(r.visit_index), int(r.sweep_index))

    sweep_csv = REPO_ROOT / "data" / "step3_tracked" / exp / "per_sweep.csv"
    vs_to_track: Dict[Tuple[int, int], Tuple[int, int, int]] = {}
    if sweep_csv.exists():
        tdf = pd.read_csv(sweep_csv)
        t_sub = tdf[tdf["roi"] == roi]
        for _, r in t_sub.iterrows():
            vs_to_track[(int(r.visit), int(r.sweep))] = (
                int(r.tracked_tcell_id),
                int(r.n_tcell_instances),
                int(r.n_target_instances),
            )

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    last_valid_t: Optional[np.ndarray] = None
    last_valid_b: Optional[np.ndarray] = None

    for stem in stems:
        p_path = REPO_ROOT / "data" / "scidb_19235_masks" / exp / roi / f"{stem}_predictions.h5"
        with h5py.File(p_path, "r") as f:
            pred = f["predictions"][:]
        vs = stem_to_vs.get(stem)
        t_mask: Optional[np.ndarray] = None
        b_mask: Optional[np.ndarray] = None
        if vs is not None and vs in vs_to_track:
            tid, nt, nb = vs_to_track[vs]
            t_full, b_full = extract_exact_visit_masks(pred, tid, nt, nb)
            if np.any(t_full):
                t_mask = t_full.astype(np.uint8)
            if np.any(b_full):
                b_mask = b_full.astype(np.uint8)

        if t_mask is None or b_mask is None:
            t_inst, b_inst, _ = run_c0(pred)
            if t_mask is None and np.any(t_inst > 0):
                t_mask = (t_inst == 1).astype(np.uint8)
            if b_mask is None and np.any(b_inst > 0):
                anchor = (t_mask > 0) if (t_mask is not None and np.any(t_mask > 0)) else (t_inst == 1)
                t_dil = binary_dilation(anchor, structure=STRUCT26, iterations=2)
                best_b = 1
                best_c = -1
                for bid in np.unique(b_inst):
                    if bid == 0:
                        continue
                    c = int(np.sum((b_inst == bid) & t_dil))
                    if c > best_c:
                        best_c = c
                        best_b = int(bid)
                b_mask = (b_inst == best_b).astype(np.uint8)

        if t_mask is None or not np.any(t_mask):
            t_mask = last_valid_t if last_valid_t is not None else np.zeros(pred.shape, dtype=np.uint8)
        else:
            last_valid_t = t_mask

        if b_mask is None or not np.any(b_mask):
            b_mask = last_valid_b if last_valid_b is not None else np.zeros(pred.shape, dtype=np.uint8)
        else:
            last_valid_b = b_mask

        out[stem] = (t_mask, b_mask)
    return out


def _identify_reference_target_cells(
    b_c1s: np.ndarray,
    b_c3: np.ndarray,
    pred: np.ndarray,
) -> List[np.ndarray]:
    """Return list of distinct 3D target cell masks (>=10,000 vox) present in the chamber visit.

    Used to objectively score whether a candidate's tracked target merges >1 target cell.
    Requires either >=2 interior cores in pred==2 or total target volume >=85,000 voxels
    so an occasional over-split of a single small target cell (<80k vox) is not counted as 2 cells.
    """
    c1s_cells = [b_c1s == u for u in np.unique(b_c1s) if u > 0 and int(np.sum(b_c1s == u)) >= 10000]
    c3_cells = [b_c3 == u for u in np.unique(b_c3) if u > 0 and int(np.sum(b_c3 == u)) >= 10000]

    core_lab, _ = label(pred == 2, structure=STRUCT26)
    n_cores = int(np.sum(np.bincount(core_lab.ravel())[1:] >= 1500))
    tot_c1s = sum(int(m.sum()) for m in c1s_cells)
    tot_c3 = sum(int(m.sum()) for m in c3_cells)

    if len(c1s_cells) >= 2 and (n_cores >= 2 or tot_c1s >= 85000):
        return c1s_cells
    if len(c3_cells) >= 2 and (n_cores >= 2 or tot_c3 >= 85000):
        return c3_cells
    return c1s_cells if len(c1s_cells) >= 1 else c3_cells


def run_chamber_visits(
    short_exp: str = "v01",
    exp: str = "v01_6_40_120_0-6_hd1",
    roi: str = "MIP_ROI_32",
    device: str = "mps",
    out_csv: Optional[Path] = None,
    reuse_c3_cache: bool = False,
) -> pd.DataFrame:
    """Run C0, C1, C1s, C3(fixed) on sweep 1 of all 20 visits of a pilot chamber."""
    manifest_df = pd.read_csv(REPO_ROOT / "data" / "scidb_19235_raw" / "manifest.csv")
    sub = manifest_df[
        (manifest_df["experiment"] == exp)
        & (manifest_df["roi"] == roi)
        & (manifest_df["ok"] == True)  # noqa: E712
        & (manifest_df["sweep_index"] != 6)
    ].copy()

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
    stems = [d["stem"] for d in visit_rows]
    ref_map = get_reference_masks_for_stems(exp, roi, stems)

    is_p1_chamber = (exp == "v01_6_40_120_0-6_hd1") and (roi == "MIP_ROI_32")

    all_metrics: List[Dict[str, object]] = []
    for item in visit_rows:
        v = int(item["visit"])
        sw = int(item["sweep"])
        stem = str(item["stem"])
        raw_path = REPO_ROOT / "data" / "scidb_19235_raw" / exp / roi / f"{stem}.h5"
        pred_path = REPO_ROOT / "data" / "scidb_19235_masks" / exp / roi / f"{stem}_predictions.h5"
        with h5py.File(raw_path, "r") as f_raw:
            raw = f_raw["raw"][:]
        with h5py.File(pred_path, "r") as f_pred:
            pred = f_pred["predictions"][:]
        t_ref, b_ref = ref_map[stem]

        # Build per-channel solid signal masks once per volume for C1s and agreement scoring
        t_sig0 = time.perf_counter()
        sig_t = build_signal_mask(raw[1], k=SIGNAL_K_TCELL, close_iter=1, min_vox=250)
        sig_b = build_signal_mask(raw[2], k=SIGNAL_K_TARGET, close_iter=2, min_vox=250)
        dt_sig = time.perf_counter() - t_sig0
        sig_t_lab, _ = label(sig_t, structure=STRUCT26)
        sig_b_lab, _ = label(sig_b, structure=STRUCT26)

        # Run all 4 candidates
        res_c0 = run_c0(pred)
        res_c1 = run_c1(pred)
        t_p0 = time.perf_counter()
        t_c1s = prune_unsupported_parts(res_c1[0], sig_t, support_thresh=SIGNAL_SUPPORT_THRESH, min_size=1250)
        b_c1s = prune_unsupported_parts(res_c1[1], sig_b, support_thresh=SIGNAL_SUPPORT_THRESH, min_size=1500)
        dt_c1s = float(res_c1[2] + dt_sig + (time.perf_counter() - t_p0))
        res_c1s = (t_c1s, b_c1s, dt_c1s)

        npz_c3_cached = REPO_ROOT / "data" / "pilot_seg" / "C3_fixed" / exp / roi / f"{stem}.npz"
        if reuse_c3_cache and npz_c3_cached.exists():
            with np.load(npz_c3_cached) as zc3:
                res_c3 = (zc3["t_inst"], zc3["b_inst"], float(zc3["runtime_s"]))
        else:
            res_c3 = run_c3(raw, pred, device=device, sig_t=sig_t, sig_b=sig_b)

        ref_target_cells = _identify_reference_target_cells(res_c1s[1], res_c3[1], pred)

        cand_results = {
            "C0": res_c0,
            "C1": res_c1,
            "C1s": res_c1s,
            "C3": res_c3,
        }

        for cand, (t_inst, b_inst, rt) in cand_results.items():
            tid, bid = select_tracked_pair(t_inst, b_inst, t_ref, b_ref)
            # Preserve existing P1 .npz files for v01 MIP_ROI_32; store fixed C3 in C3_fixed as well
            if cand == "C3":
                npz_p2 = REPO_ROOT / "data" / "pilot_seg" / "C3_fixed" / exp / roi / f"{stem}.npz"
                save_candidate_npz(npz_p2, t_inst, b_inst, tid, bid, rt)
                if not is_p1_chamber:
                    npz_std = REPO_ROOT / "data" / "pilot_seg" / "C3" / exp / roi / f"{stem}.npz"
                    save_candidate_npz(npz_std, t_inst, b_inst, tid, bid, rt)
            else:
                npz_std = REPO_ROOT / "data" / "pilot_seg" / cand / exp / roi / f"{stem}.npz"
                if not (is_p1_chamber and cand in ("C0", "C1") and npz_std.exists()):
                    save_candidate_npz(npz_std, t_inst, b_inst, tid, bid, rt)

            m = compute_candidate_metrics(
                t_inst,
                b_inst,
                tid,
                bid,
                rt,
                sig_t=sig_t,
                sig_b=sig_b,
                sig_t_lab=sig_t_lab,
                sig_b_lab=sig_b_lab,
                ref_target_cells=ref_target_cells,
            )
            row_out = {
                "short_exp": short_exp,
                "experiment": exp,
                "roi": roi,
                "visit": v,
                "sweep": sw,
                "stem": stem,
                "candidate": cand,
                "n_connected_components": int(m["target_n_cc"]) + int(m["tcell_n_cc"]),
                **m,
            }
            all_metrics.append(row_out)
            print(
                f"[{short_exp} {roi} v{v:02d} s{sw} {stem}] {cand:3s}: "
                f"B_P={m['target_precision']:.2f} B_R={m['target_recall']:.2f} | "
                f"T_P={m['tcell_precision']:.2f} T_R={m['tcell_recall']:.2f} | "
                f"merged={m['target_merged_gt1']} | time={m['runtime_s']:.2f}s"
            )

    df_out = pd.DataFrame(all_metrics)
    if out_csv is None:
        out_csv = REPO_ROOT / "data" / "pilot_seg" / f"{short_exp}_{roi}_p2_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    if not is_p1_chamber:
        alt_csv = REPO_ROOT / "data" / "pilot_seg" / f"{short_exp}_{roi}_metrics.csv"
        df_out.to_csv(alt_csv, index=False)
    print(f"Saved metrics CSV -> {out_csv}")
    return df_out


def run_all_pilot2_chambers(device: str = "mps") -> pd.DataFrame:
    """Run C0, C1, C1s, C3(fixed) across all 8 pilot chambers and write combined CSV."""
    dfs: List[pd.DataFrame] = []
    for short_exp, exp, roi in PILOT_CHAMBERS:
        df_ch = run_chamber_visits(short_exp=short_exp, exp=exp, roi=roi, device=device)
        dfs.append(df_ch)
    combined = pd.concat(dfs, ignore_index=True)
    comb_path = REPO_ROOT / "data" / "pilot_seg" / "pilot2_metrics.csv"
    combined.to_csv(comb_path, index=False)
    print(f"Saved combined P2 metrics CSV -> {comb_path} ({len(combined)} rows)")
    return combined


if __name__ == "__main__":
    run_all_pilot2_chambers(device="mps")
