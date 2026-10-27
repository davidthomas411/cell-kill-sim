#!/usr/bin/env python3
"""Render side-by-side MP4 and GIF videos for Single-Cell Twins 4, 5, 6, and 7 in parallel."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from paths import REPO_ROOT as ROOT, EXTERNAL_DIR
from build_all_ground_truth_4d_twins import run_single_cell_lightsheet_twin

SPECS = [
    ("Twin4_MOESM8_FastBurst", "Twin 4 (MOESM8: 2.5s/vol Fast Burst)", "MOESM8_ESM.mp4", 10, 1060, 15.0,
     (slice(430, 1120), slice(195, 885)), (slice(1480, 1840), slice(195, 885)), (1080, 1920), "none",
     "side_by_side_twin4_moesm8_fast_burst"),
    ("Twin5_MOESM5_Chamber1", "Twin 5 (MOESM5: Chamber 1 Continuous)", "MOESM5_ESM.mp4", 365, 588, 10.0,
     (slice(130, 450), slice(700, 1020)), (slice(505, 645), slice(700, 1020)), None, "none",
     "side_by_side_twin5_moesm5_chamber1"),
    ("Twin6_MOESM5_Chamber2", "Twin 6 (MOESM5: Chamber 2 Continuous)", "MOESM5_ESM.mp4", 615, 815, 10.0,
     (slice(130, 450), slice(700, 1020)), (slice(505, 645), slice(700, 1020)), None, "none",
     "side_by_side_twin6_moesm5_chamber2"),
    ("Twin7_MOESM5_SiRTubulin", "Twin 7 (MOESM5: SiR-Tubulin MTOC)", "MOESM5_ESM.mp4", 65, 240, 60.0,
     (slice(375, 595), slice(455, 695)), None, None, "tubulin",
     "side_by_side_twin7_moesm5_sirtubulin_mtoc"),
]


def _worker(spec):
    tid, tlabel, vfile, fs, fe, tmin, xys, xzs, rwh, bmode, ostem = spec
    ls_dir = EXTERNAL_DIR / "feilab_lightsheet_data"
    run_single_cell_lightsheet_twin(
        ls_dir / vfile, tid, tlabel, fs, fe, tmin, xys, xzs,
        resize_wh=rwh, blue_channel_mode=bmode, n_sample_frames=28,
        encode_video=True, out_video_stem=ostem
    )
    return tid, ostem


def main():
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(_worker, s) for s in SPECS]
        for f in as_completed(futs):
            tid, ostem = f.result()
            print(f"[Done] {tid} -> videos/{ostem}.mp4")


if __name__ == "__main__":
    main()
