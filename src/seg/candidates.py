"""Candidate 3D instance segmentation methods (C0..C3) for CAR-T + target chambers.

Voxel pitch assumption:
  xy = 0.1962 um, z treated as 0.1962 um (nominal isotropic anisotropy = 1.0).
  Voxel volume = 0.1962^3 = 0.007552625848 um^3.
  Voxel face area = 0.1962^2 = 0.03849444 um^2 (~0.0385 um^2).
  Sweep 6 has SiR-tubulin in ch2 and must never be used.

Candidates:
  C0: Lab masks as deposited (instances via step2_instance_seg.m / make_gallery.py
      without 3D hole repair, preserving raw deposited topology and cavities).
  C1: Lab masks repaired: 3D fill-holes (fill_solid_cell_mask) and distance-transform
      seeded watershed on the boundary class (classes 3/4) to split merged targets.
  C2: Cellpose-SAM 3D zero-shot (cpsam_v2): T cells on ch1, targets on ch2 (membrane),
      do_3D=True, anisotropy=1.0, default flow_threshold=0.4, cellprob_threshold=0.0.
  C3: PlantSeg (generic_light_sheet_3D_unet + dt_watershed) on ch2 and ch1 within
      the 3D-filled raw fluorescence envelope.
"""

from __future__ import annotations

import os
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
from src.mask_qc import fill_solid_cell_mask
from src.reproduce_contact_area import STRUCT26
from src.review.make_clean_gallery import extract_exact_visit_masks

VOXEL_UM = 0.1962
ANISOTROPY = 1.0
VOXEL_VOL_UM3 = VOXEL_UM ** 3  # 0.007552625848 um^3
FACE_AREA_UM2 = 0.0385


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
) -> Tuple[slice, slice, slice]:
    """Return 3D bounding box slice around foreground components >= min_comp_voxels."""
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
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


def run_c0(pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """C0: Lab masks as deposited (step2_instance_seg.m via instance_seg_matlab_xyz).

    Preserves raw deposited 3D instance masks (with imfill_holes_26 only, without
    2D slice hole repair) so cavities and merges in the deposited masks are measured.
    """
    t0 = time.perf_counter()
    t_raw = instance_seg_matlab_xyz(pred == 1, pred == 3)
    b_raw = instance_seg_matlab_xyz(pred == 2, pred == 4)
    t_inst = np.where((pred == 1) | (pred == 3), t_raw, 0).astype(np.uint16)
    b_inst = np.where((pred == 2) | (pred == 4), b_raw, 0).astype(np.uint16)
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
    filled_fg = fill_solid_cell_mask(clean_fg)

    if not split_merged:
        # 3D fill-holes per connected component of the deposited instances
        comp_lab, n_comp = label(filled_fg, structure=STRUCT26)
        out = np.zeros(pred.shape, dtype=np.uint16)
        for cid in range(1, n_comp + 1):
            cmask = comp_lab == cid
            if int(cmask.sum()) >= min_comp_size:
                out[fill_solid_cell_mask(cmask)] = cid
        return _relabel_sequential(out, min_size=min_comp_size // 2)

    dt_raw = distance_transform_edt(filled_fg)
    dt_smooth = gaussian_filter(dt_raw, sigma=2.5)
    comp_lab, n_comp = label(filled_fg, structure=STRUCT26)
    core_lab, _ = label(filled_fg & (pred != boundary_cls), structure=STRUCT26)
    core_cnt = np.bincount(core_lab.ravel())

    seeds = np.zeros(pred.shape, dtype=np.int32)
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

    # Watershed on boundary class: boundary voxels (class 3/4) have higher traversal cost
    cost = -dt_raw + 4.0 * (pred == boundary_cls).astype(np.float32)
    ws = watershed(cost, seeds, mask=filled_fg)

    out = np.zeros(pred.shape, dtype=np.uint16)
    for k in range(1, next_id):
        inst_mask = ws == k
        if int(inst_mask.sum()) >= 1500:
            out[fill_solid_cell_mask(inst_mask)] = k
    return _relabel_sequential(out, min_size=1500)


def run_c1(pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """C1: Lab masks repaired (3D fill-holes + distance-transform watershed on boundary class)."""
    t0 = time.perf_counter()
    # T cells: 3D fill-holes of each instance (plus boundary watershed if multiple cores)
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
    # Targets: 3D fill-holes + split merged targets with seeded watershed on boundary class 4
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


_CELLPOSE_MODELS: Dict[Tuple[str, bool], object] = {}


def _get_cellpose_model(device: str = "mps"):
    from cellpose import models

    use_bf16 = device == "mps"
    key = (device, use_bf16)
    if key not in _CELLPOSE_MODELS:
        dev = torch.device(device)
        _CELLPOSE_MODELS[key] = models.CellposeModel(
            gpu=(device != "cpu"),
            device=dev,
            use_bfloat16=use_bf16,
        )
    return _CELLPOSE_MODELS[key]


def run_c2(
    raw: np.ndarray,
    pred: np.ndarray,
    device: str = "mps",
    diameter: Optional[float] = 60.0,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """C2: Cellpose-SAM 3D zero-shot on ch1 (T cells) and ch2 (target membrane).

    Runs do_3D=True, anisotropy=1.0, default flow_threshold=0.4, cellprob_threshold=0.0.
    Subtracts camera read-noise baseline (median + 4*MAD) on the active cell bounding box
    so 1..99 percentile normalization does not amplify camera offset noise (~390-400 ADU).
    """
    model = _get_cellpose_model(device=device)
    t0 = time.perf_counter()

    # 1. T cells on ch1
    sl_t = _active_crop_slice((pred == 1) | (pred == 3), margin=10, min_comp_voxels=2000)
    sub_t = raw[1][sl_t].astype(np.float32)
    med_t = float(np.median(raw[1]))
    mad_t = float(np.median(np.abs(raw[1] - med_t)))
    sub_t_clean = np.clip(sub_t - (med_t + 3.5 * 1.4826 * mad_t), 0.0, None)

    masks_t_sub, _, _ = model.eval(
        sub_t_clean,
        z_axis=0,
        do_3D=True,
        anisotropy=ANISOTROPY,
        diameter=diameter,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        normalize={"percentile": [1.0, 99.9]},
        batch_size=batch_size,
        min_size=250,
    )
    t_inst = np.zeros(pred.shape, dtype=np.uint16)
    t_inst[sl_t] = _relabel_sequential(masks_t_sub.astype(np.uint16), min_size=250)

    # 2. Targets on ch2 (membrane)
    sl_b = _active_crop_slice((pred == 2) | (pred == 4), margin=10, min_comp_voxels=2000)
    sub_b = raw[2][sl_b].astype(np.float32)
    med_b = float(np.median(raw[2]))
    mad_b = float(np.median(np.abs(raw[2] - med_b)))
    sub_b_clean = np.clip(sub_b - (med_b + 3.5 * 1.4826 * mad_b), 0.0, None)

    masks_b_sub, _, _ = model.eval(
        sub_b_clean,
        z_axis=0,
        do_3D=True,
        anisotropy=ANISOTROPY,
        diameter=diameter,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        normalize={"percentile": [1.0, 99.9]},
        batch_size=batch_size,
        min_size=250,
    )
    b_inst = np.zeros(pred.shape, dtype=np.uint16)
    b_inst[sl_b] = _relabel_sequential(masks_b_sub.astype(np.uint16), min_size=250)

    dt = time.perf_counter() - t0
    return t_inst, b_inst, dt


def _plantseg_channel(
    vol_ch: np.ndarray,
    fg_hint: np.ndarray,
    device: str = "mps",
    sigma_seeds: float = 7.0,
    min_size: int = 2500,
    otsu_factor: float = 0.68,
) -> np.ndarray:
    """Run PlantSeg 3D U-Net + distance-transform watershed on a single fluorescence channel."""
    from plantseg.functionals.prediction.prediction import unet_prediction
    from plantseg.functionals.segmentation.segmentation import dt_watershed

    sl = _active_crop_slice(fg_hint, margin=10, min_comp_voxels=2000)
    sub = vol_ch[sl].astype(np.float32)

    # Compute 3D filled outer envelope from raw fluorescence above camera read noise
    med = float(np.median(vol_ch))
    mad = float(np.median(np.abs(vol_ch - med)))
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

    patch = tuple(min(128, max(32, (d // 16) * 16)) for d in sub.shape)
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

    ws = dt_watershed(
        pm,
        threshold=0.5,
        sigma_seeds=sigma_seeds,
        stacked=False,
        sigma_weights=2.0,
        min_size=min_size,
        alpha=1.0,
        pixel_pitch=(1.0, 1.0, 1.0),
        apply_nonmax_suppression=False,
        n_threads=4,
        mask=env,
    )
    out = np.zeros(vol_ch.shape, dtype=np.uint16)
    ws_clean = _relabel_sequential(ws.astype(np.uint16), min_size=min_size // 2)
    # Keep dominant connected component and fill internal holes per watershed instance
    uids = np.unique(ws_clean)
    for uid in uids[uids > 0]:
        cc_lab, n_cc = label(ws_clean == uid, structure=STRUCT26)
        if n_cc > 1:
            cc_cnt = np.bincount(cc_lab.ravel())
            cc_cnt[0] = 0
            dom = cc_lab == int(np.argmax(cc_cnt))
        else:
            dom = ws_clean == uid
        out_sub = fill_solid_cell_mask(dom)
        out[sl][out_sub] = int(uid)
    return _relabel_sequential(out, min_size=min_size // 2)


def run_c3(
    raw: np.ndarray,
    pred: np.ndarray,
    device: str = "mps",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """C3: PlantSeg (generic_light_sheet_3D_unet + dt_watershed) on ch1 and ch2."""
    t0 = time.perf_counter()
    t_inst = _plantseg_channel(
        raw[1],
        fg_hint=(pred == 1) | (pred == 3),
        device=device,
        sigma_seeds=6.5,
        min_size=2500,
        otsu_factor=0.68,
    )
    b_inst = _plantseg_channel(
        raw[2],
        fg_hint=(pred == 2) | (pred == 4),
        device=device,
        sigma_seeds=8.0,
        min_size=3000,
        otsu_factor=0.68,
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
        if ov_ref == 0:
            continue
        contact_vox = int(np.sum(bmask & t_dil))
        # Overlap with lab's tracked target + contact with tracked T cell
        score = float(ov_ref) + 8.0 * float(contact_vox)
        if score > best_b_score:
            best_b_score = score
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
        # Exclude interior overlap voxels if masks overlap; count boundary transitions
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


def compute_candidate_metrics(
    t_inst: np.ndarray,
    b_inst: np.ndarray,
    tracked_t_id: int,
    tracked_b_id: int,
    runtime_s: float,
) -> Dict[str, float | int]:
    """Compute per-visit segmentation metrics for a candidate."""
    t_mask = (t_inst == tracked_t_id) if tracked_t_id > 0 else np.zeros(t_inst.shape, dtype=bool)
    b_mask = (b_inst == tracked_b_id) if tracked_b_id > 0 else np.zeros(b_inst.shape, dtype=bool)

    t_vox = int(np.sum(t_mask))
    b_vox = int(np.sum(b_mask))

    t_filled = fill_solid_cell_mask(t_mask) if t_vox > 0 else t_mask
    b_filled = fill_solid_cell_mask(b_mask) if b_vox > 0 else b_mask

    t_holes_vox = max(0, int(np.sum(t_filled)) - t_vox)
    b_holes_vox = max(0, int(np.sum(b_filled)) - b_vox)

    # Connected components: both total non-zero instances in chamber and components of tracked mask
    n_t_instances = int(len(np.unique(t_inst)) - (1 if 0 in t_inst else 0))
    n_b_instances = int(len(np.unique(b_inst)) - (1 if 0 in b_inst else 0))
    _, n_t_cc = label(t_mask, structure=STRUCT26) if t_vox > 0 else (None, 0)
    _, n_b_cc = label(b_mask, structure=STRUCT26) if b_vox > 0 else (None, 0)

    faces = count_interface_faces(t_mask, b_mask)
    contact_area_um2 = float(faces) * FACE_AREA_UM2

    return {
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
    extract_exact_visit_masks and step3_tracked/<exp>/per_sweep.csv."""
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
    for stem in stems:
        p_path = REPO_ROOT / "data" / "scidb_19235_masks" / exp / roi / f"{stem}_predictions.h5"
        with h5py.File(p_path, "r") as f:
            pred = f["predictions"][:]
        vs = stem_to_vs.get(stem)
        if vs is not None and vs in vs_to_track:
            tid, nt, nb = vs_to_track[vs]
            t_full, b_full = extract_exact_visit_masks(pred, tid, nt, nb)
            if np.any(t_full) and np.any(b_full):
                out[stem] = (t_full.astype(np.uint8), b_full.astype(np.uint8))
                continue
        # Fallback if sweep missing from step3_tracked
        t_inst, b_inst, _ = run_c0(pred)
        t_mask = (t_inst == 1).astype(np.uint8)
        t_dil = binary_dilation(t_mask > 0, structure=STRUCT26, iterations=2)
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
        out[stem] = (t_mask, b_mask)
    return out


def run_chamber_visits(
    exp: str = "v01_6_40_120_0-6_hd1",
    roi: str = "MIP_ROI_32",
    device: str = "mps",
    out_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Run C0..C3 on all visits (sweep 1; fallback sweep 2) of a chamber."""
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

    # Warmup Cellpose model before timing
    _ = _get_cellpose_model(device=device)

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

        for cand in ["C0", "C1", "C2", "C3"]:
            npz_path = REPO_ROOT / "data" / "pilot_seg" / cand / exp / roi / f"{stem}.npz"
            if cand == "C0":
                t_inst, b_inst, rt = run_c0(pred)
            elif cand == "C1":
                t_inst, b_inst, rt = run_c1(pred)
            elif cand == "C2":
                t_inst, b_inst, rt = run_c2(raw, pred, device=device, diameter=60.0)
            else:
                t_inst, b_inst, rt = run_c3(raw, pred, device=device)

            tid, bid = select_tracked_pair(t_inst, b_inst, t_ref, b_ref)
            save_candidate_npz(npz_path, t_inst, b_inst, tid, bid, rt)
            m = compute_candidate_metrics(t_inst, b_inst, tid, bid, rt)
            row_out = {
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
                f"[v={v:02d} s={sw} {stem}] {cand}: "
                f"B_vol={m['target_vol_um3']:.1f} T_vol={m['tcell_vol_um3']:.1f} "
                f"holes={m['holes_um3']:.1f} CA={m['contact_area_um2']:.1f} "
                f"n_inst=({m['n_target_instances']},{m['n_tcell_instances']}) "
                f"rt={m['runtime_s']:.2f}s",
                flush=True,
            )

        if out_csv is not None:
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_metrics).to_csv(out_csv, index=False)

    df_out = pd.DataFrame(all_metrics)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(out_csv, index=False)
    return df_out


if __name__ == "__main__":
    out_file = REPO_ROOT / "data" / "pilot_seg" / "v01_MIP_ROI_32_metrics.csv"
    run_chamber_visits(
        exp="v01_6_40_120_0-6_hd1",
        roi="MIP_ROI_32",
        device="mps",
        out_csv=out_file,
    )

