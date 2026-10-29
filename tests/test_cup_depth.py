#!/usr/bin/env python3
"""
Unit tests for 3D convex-hull cup-depth measurement (`cup_depth_hull_p95_um` in
`src/contact_descriptors.py` lines 709-715) across synthetic geometries (a)-(f) at
`VOXEL_UM = 0.1962 um`:
  a. Flat contact (sphere with planar cut, cap removed, patch radius a = 2, 4, 6 um; expected ~0).
  b. Spherical dent of depth d = 0.5, 1, 2, 3 um over patch radius a = 3, 5 um; expected ~d.
  c. Same as (b) with n_syn tilted by 10, 20, 30 degrees from the true axis.
  d. Wide gently curved contact (sag 0.3 um over a = 6 um); expected ~0.3 um.
  e. Two separate spheres (merged-target case) with T cell between them.
  f. Hollow shell target (thickness 0.8 um) with a dent of 1 um (raw shell and after fill_solid_cell_mask).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from src.constants import VOXEL_UM
from src.validate_cup_depth import (
    build_flat_contact_masks,
    build_hollow_shell_masks,
    build_spherical_dent_masks,
    build_two_spheres_masks,
    build_wide_bowl_masks,
    evaluate_masks_pipeline_and_all_facets,
)


class TestCupDepthSynthetic(unittest.TestCase):
    """Validate `cup_depth_hull_p95_um` and all-facets hull depth on synthetic 3D masks."""

    def test_a_flat_contact_zero_depth(self) -> None:
        """(a) Sphere with planar cut and T cell pressed flat against it gives ~0 hull depth."""
        for R in (4.0, 5.0, 6.0):
            for a in (2.0, 4.0, 6.0):
                if a > R:
                    continue
                b_m, t_m = build_flat_contact_masks(R, a, n_vox=112)
                res = evaluate_masks_pipeline_and_all_facets(b_m, t_m)
                self.assertLessEqual(res["cup_depth_hull_p95_um"], VOXEL_UM + 1e-4)
                self.assertLessEqual(res["cup_depth_all_facets_p95_um"], VOXEL_UM + 1e-4)
                # Reference sphere depth is strictly positive on a planar cut because of cap sag
                self.assertGreater(res["cup_depth_sphere_p95_um"], 0.20)

    def test_b_spherical_dent_tracks_true_depth(self) -> None:
        """(b) Spherical dent of depth d = 0.5, 1, 2, 3 um over a = 3, 5 um matches expected d."""
        for R, a in [(4.0, 3.0), (5.0, 3.0), (6.0, 3.0), (6.0, 5.0)]:
            for d in (0.5, 1.0, 2.0, 3.0):
                b_m, t_m = build_spherical_dent_masks(R, a, d, n_vox=112)
                res = evaluate_masks_pipeline_and_all_facets(b_m, t_m)
                # Max depth is within 1 voxel of d; p95 is within ~1.4 voxels of d
                self.assertAlmostEqual(res["cup_depth_hull_max_um"], d, delta=1.1 * VOXEL_UM)
                self.assertAlmostEqual(res["cup_depth_hull_p95_um"], d, delta=1.5 * VOXEL_UM)
                self.assertAlmostEqual(
                    res["cup_depth_hull_p95_um"],
                    res["cup_depth_all_facets_p95_um"],
                    delta=VOXEL_UM,
                )

    def test_c_tilted_n_syn_invariance(self) -> None:
        """(c) Tilting n_syn by 10, 20, 30 deg does not bias concave dent depth (cos(30 deg)=0.866 > 0.05)."""
        b_m, t_m = build_spherical_dent_masks(5.0, 3.0, 2.0, n_vox=112)
        res_0 = evaluate_masks_pipeline_and_all_facets(b_m, t_m, tilt_deg=0.0)
        for tilt in (10.0, 20.0, 30.0):
            res_t = evaluate_masks_pipeline_and_all_facets(b_m, t_m, tilt_deg=tilt)
            self.assertAlmostEqual(
                res_t["cup_depth_hull_p95_um"],
                res_0["cup_depth_hull_p95_um"],
                delta=1e-4,
            )
            self.assertAlmostEqual(
                res_t["cup_depth_hull_p95_um"],
                res_t["cup_depth_all_facets_p95_um"],
                delta=1e-4,
            )

    def test_d_wide_bowl_sag_0p3(self) -> None:
        """(d) Wide gently curved contact (sag 0.3 um over a = 6 um) gives ~0.3 um hull depth."""
        b_m, t_m = build_wide_bowl_masks(R_um=6.0, a_um=6.0, sag_um=0.3, n_vox=112)
        res = evaluate_masks_pipeline_and_all_facets(b_m, t_m)
        self.assertAlmostEqual(res["cup_depth_hull_p95_um"], 0.3, delta=VOXEL_UM)
        self.assertAlmostEqual(res["cup_depth_all_facets_p95_um"], 0.3, delta=VOXEL_UM)

    def test_e_two_merged_spheres_spurious_cleft_depth(self) -> None:
        """(e) Two merged un-dented spheres with T cell between them produce large bridging hull depth."""
        b_m, t_m = build_two_spheres_masks(R_um=5.0, n_vox=112)
        res = evaluate_masks_pipeline_and_all_facets(b_m, t_m)
        self.assertGreater(res["cup_depth_hull_p95_um"], 2.0)
        self.assertGreater(res["cup_depth_all_facets_p95_um"], 2.0)

    def test_f_hollow_shell_raw_vs_filled(self) -> None:
        """(f) Hollow shell (0.8 um wall, 1.0 um dent) recovers ~1.0 um raw, but collapses under 2D slice hole fill."""
        b_raw, t_raw = build_hollow_shell_masks(
            R_um=5.0, a_um=3.0, d_um=1.0, thickness_um=0.8, apply_fill_solid=False, n_vox=112
        )
        res_raw = evaluate_masks_pipeline_and_all_facets(b_raw, t_raw)
        self.assertAlmostEqual(res_raw["cup_depth_hull_p95_um"], 1.0, delta=VOXEL_UM)
        self.assertAlmostEqual(res_raw["cup_depth_all_facets_p95_um"], 1.0, delta=VOXEL_UM)

        b_fill, t_fill = build_hollow_shell_masks(
            R_um=5.0, a_um=3.0, d_um=1.0, thickness_um=0.8, apply_fill_solid=True, n_vox=112
        )
        res_fill = evaluate_masks_pipeline_and_all_facets(b_fill, t_fill)
        self.assertLess(res_fill["cup_depth_hull_p95_um"], 0.4)


if __name__ == "__main__":
    unittest.main()
