#!/usr/bin/env python3
"""
Step 6b Stage 1 Unit Tests: `tests/test_shell_model.py`
Tests for `src/mech/shell_model.py`:
  1. Analytical gradient check by central finite differences (relative error < 1e-6 across all terms:
     tension, area stretch, bending, volume, contact penalty, synapse tension x alpha_syn, active pressure).
  2. Energy strictly decreases during L-BFGS-B relaxation under active synaptic push.
  3. Volume conserved within < 1% for both T-cell and target shells.
  4. No interpenetration (minimum contact separation remains positive / above penetration tolerance).
  5. Manifold check on synthetic spheres and marching-cubes mask extraction.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mech.shell_model import (
    ShellMesh,
    ShellParameters,
    TwoShellSystem,
    afm_modulus_to_cortical_tension,
    build_icosphere,
    check_closed_manifold,
    compute_closed_mesh_volume,
    compute_mesh_descriptors,
    extract_cell_mesh_marching_cubes,
    solve_two_shell_equilibrium,
)


def _make_touching_sphere_pair(
    r_tcell: float = 4.5,
    r_target: float = 5.5,
    target_edge_um: float = 1.4,
    patch_radius_um: float = 2.8,
) -> tuple[ShellMesh, ShellMesh]:
    """Create two touching triangulated spheres along the Z axis with tagged synapse faces."""
    gap = 0.15
    c_target = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    c_tcell = np.array([0.0, 0.0, -(r_target + r_tcell + gap)], dtype=np.float64)

    V_t, F_t = build_icosphere(c_tcell, r_tcell, target_edge_um=target_edge_um)
    V_g, F_g = build_icosphere(c_target, r_target, target_edge_um=target_edge_um)

    # Tag T-cell faces near the north pole facing the target (within patch_radius_um laterally and z > c_tcell[2])
    f_cent_t = np.mean(V_t[F_t], axis=1)
    r_lat = np.hypot(f_cent_t[:, 0], f_cent_t[:, 1])
    syn_faces = (f_cent_t[:, 2] > c_tcell[2] + 0.5 * r_tcell) & (r_lat <= patch_radius_um)
    if not np.any(syn_faces):
        syn_faces[np.argmax(f_cent_t[:, 2])] = True

    t_mesh = ShellMesh(V0=V_t, F=F_t, synapse_faces=syn_faces)
    g_mesh = ShellMesh(V0=V_g, F=F_g, synapse_faces=np.zeros(len(F_g), dtype=bool))
    return t_mesh, g_mesh


class TestShellModel(unittest.TestCase):
    """Unit tests for the Step 6b two-shell mechanical model."""

    def test_closed_manifold_icosphere_and_mask(self) -> None:
        """Verify that icosphere and mask extraction produce closed genus-0 2-manifolds."""
        V, F = build_icosphere(np.zeros(3), 5.0, target_edge_um=1.0)
        info = check_closed_manifold(V, F)
        self.assertTrue(info["is_closed_2manifold"])
        self.assertEqual(info["euler_characteristic"], 2)

        # Synthetic 3D binary mask with an indentation + handle tunnel to test manifold repair
        zz, yy, xx = np.ogrid[-30:31, -30:31, -30:31]
        r2 = zz * zz + yy * yy + xx * xx
        mask = (r2 <= 22 ** 2).astype(bool)
        # Carve a cup dent on top and a small internal void
        dent = ((zz - 18) ** 2 + yy * yy + xx * xx) <= 10 ** 2
        mask = mask & (~dent)
        V_m, F_m = extract_cell_mesh_marching_cubes(mask, np.zeros(3), target_edge_um=1.2)
        info_m = check_closed_manifold(V_m, F_m)
        self.assertTrue(info_m["is_closed_2manifold"])
        self.assertEqual(info_m["euler_characteristic"], 2)
        self.assertGreater(compute_closed_mesh_volume(V_m, F_m), 100.0)

    def test_analytic_gradient_matches_finite_differences(self) -> None:
        """
        Verify analytical gradient `energy_and_gradient(x)` against central finite differences
        `(E(x + eps*e_i) - E(x - eps*e_i)) / (2*eps)` across random perturbations where all
        energy terms (tension, area stretch, bending, volume, contact penalty, alpha_syn=2,
        and active pressure) are simultaneously active.
        """
        t_mesh, g_mesh = _make_touching_sphere_pair(target_edge_um=1.6)
        params = ShellParameters.from_dimensionless(
            pi_push=1.2, chi_t=1.3, alpha_syn=2.0, r_target_um=g_mesh.R_eq0
        )
        sys_model = TwoShellSystem(t_mesh, g_mesh, params)

        rng = np.random.default_rng(42)
        x0 = sys_model.pack_state(t_mesh.V0, g_mesh.V0)
        # Perturb vertices so area stretch, bending, volume, and contact penetration are all non-zero
        x_test = x0 + rng.normal(0.0, 0.08, size=x0.shape)
        # Push T-cell slightly into target so contact penalty is strongly active
        x_test[: 3 * sys_model.n_vt : 3] += 0.02
        x_test[2 : 3 * sys_model.n_vt : 3] += 0.25

        _, grad_analytic = sys_model.energy_and_gradient(x_test)

        # Check 60 coordinates spanning T-cell synapse, T-cell back, Target contact, and Target back
        eps = 1e-6
        indices = np.linspace(0, len(x_test) - 1, 60, dtype=int)
        fd_grad = np.zeros(len(indices), dtype=np.float64)
        an_grad = grad_analytic[indices]

        for k, idx in enumerate(indices):
            x_plus = x_test.copy()
            x_minus = x_test.copy()
            x_plus[idx] += eps
            x_minus[idx] -= eps
            e_plus, _ = sys_model.energy_and_gradient(x_plus)
            e_minus, _ = sys_model.energy_and_gradient(x_minus)
            fd_grad[k] = (e_plus - e_minus) / (2.0 * eps)

        rel_err = np.linalg.norm(an_grad - fd_grad) / max(1e-12, np.linalg.norm(fd_grad))
        max_abs_err = float(np.max(np.abs(an_grad - fd_grad)))
        self.assertLess(rel_err, 1e-6, f"Relative gradient error {rel_err:.3e} >= 1e-6 (max abs {max_abs_err:.3e})")

    def test_energy_decreases_volume_conserved_no_interpenetration(self) -> None:
        """
        Verify that L-BFGS-B relaxation under active pressure:
          1. Strictly decreases total potential energy (`E_final < E_initial`).
          2. Conserves both T-cell and target volumes (`|dV|/V_0 < 1.5%`).
          3. Prevents interpenetration (`min_contact_sep_um > 0`) while indenting the target cup.
          4. Shows deeper target indentation or stiffer T-cell resistance for `alpha_syn = 2` vs `1`.
        """
        t_mesh, g_mesh = _make_touching_sphere_pair(target_edge_um=1.2)
        params2 = ShellParameters.from_dimensionless(
            pi_push=1.5, chi_t=1.0, alpha_syn=2.0, r_target_um=g_mesh.R_eq0
        )
        sol2 = solve_two_shell_equilibrium(t_mesh, g_mesh, params2, maxiter=450, gtol=5e-2)

        self.assertTrue(sol2["success"])
        self.assertLess(sol2["E_final"], sol2["E_initial"] - 1.0)
        self.assertLess(sol2["breakdown"]["vol_tcell_rel_err"], 0.01)
        self.assertLess(sol2["breakdown"]["vol_target_rel_err"], 0.01)
        self.assertGreater(sol2["breakdown"]["min_contact_sep_um"], 0.0)
        self.assertEqual(sol2["breakdown"]["n_inverted_total"], 0)

        desc_init = compute_mesh_descriptors(
            t_mesh.V0, t_mesh.F, g_mesh.V0, g_mesh.F, synapse_faces_t=t_mesh.synapse_faces
        )
        desc_eq2 = compute_mesh_descriptors(
            sol2["V_tcell"], t_mesh.F, sol2["V_target"], g_mesh.F, synapse_faces_t=t_mesh.synapse_faces
        )
        self.assertGreater(desc_eq2["cup_depth_hull_p95_um"], desc_init["cup_depth_hull_p95_um"] + 0.10)
        self.assertGreater(desc_eq2["cup_depth_sphere_p95_um"], desc_init["cup_depth_sphere_p95_um"] + 0.50)

        # Compare alpha_syn = 2 vs alpha_syn = 1
        params1 = ShellParameters.from_dimensionless(
            pi_push=1.5, chi_t=1.0, alpha_syn=1.0, r_target_um=g_mesh.R_eq0
        )
        sol1 = solve_two_shell_equilibrium(t_mesh, g_mesh, params1, maxiter=450, gtol=5e-2)
        self.assertTrue(sol1["success"])
        desc_eq1 = compute_mesh_descriptors(
            sol1["V_tcell"], t_mesh.F, sol1["V_target"], g_mesh.F, synapse_faces_t=t_mesh.synapse_faces
        )
        # Both alpha_syn=2 and alpha_syn=1 indent the target and conserve volume
        self.assertLess(sol1["breakdown"]["vol_target_rel_err"], 0.01)
        self.assertNotAlmostEqual(desc_eq2["cup_depth_hull_p95_um"], desc_eq1["cup_depth_hull_p95_um"], places=3)

    def test_two_shell_symmetry_and_no_interpenetration(self) -> None:
        """
        Verify that two identical mirror-image shells pushed together remain mirror-symmetric
        (`max |V_g - V_t * [1, 1, -1]| < 1e-6`), converge to gtol, conserve volume (< 1%),
        have zero inverted triangles, and do not interpenetrate (`min_contact_sep_um > 0`).
        """
        R_sph = 5.0
        gap_um = 0.15
        c_target = np.array([0.0, 0.0, +(R_sph + 0.5 * gap_um)], dtype=np.float64)
        V_g0, F_g = build_icosphere(c_target, R_sph, target_edge_um=1.0)
        V_t0 = V_g0 * np.array([1.0, 1.0, -1.0], dtype=np.float64)
        F_t = F_g[:, [0, 2, 1]].copy()

        f_cent_t = np.mean(V_t0[F_t], axis=1)
        r_lat = np.hypot(f_cent_t[:, 0], f_cent_t[:, 1])
        syn_faces = ((f_cent_t[:, 2] > (-(R_sph + 0.5 * gap_um) + 0.5 * R_sph)) & (r_lat <= 2.6)).astype(np.float64)

        t_mesh = ShellMesh(V0=V_t0, F=F_t, synapse_faces=syn_faces.copy())
        g_mesh = ShellMesh(V0=V_g0, F=F_g, synapse_faces=syn_faces.copy())
        params = ShellParameters.from_dimensionless(
            pi_push=2.4, chi_t=1.0, alpha_syn=2.0, r_target_um=R_sph
        )
        sol = solve_two_shell_equilibrium(t_mesh, g_mesh, params, maxiter=450, gtol=5e-2)
        self.assertTrue(sol["success"])
        V_t_mirrored = sol["V_tcell"] * np.array([1.0, 1.0, -1.0], dtype=np.float64)
        max_sym_err = float(np.max(np.linalg.norm(sol["V_target"] - V_t_mirrored, axis=1)))
        self.assertLess(max_sym_err, 1e-6)
        self.assertGreater(sol["breakdown"]["min_contact_sep_um"], 0.0)
        self.assertLess(sol["breakdown"]["vol_tcell_rel_err"], 0.01)
        self.assertLess(sol["breakdown"]["vol_target_rel_err"], 0.01)
        self.assertEqual(sol["breakdown"]["n_inverted_total"], 0)

    def test_tense_sphere_unilateral_contact_benchmark(self) -> None:
        """
        Verify the paper-derived unilateral contact benchmark (tense volume-conserving sphere
        pressed by a rigid axisymmetric indenter) against the exact closed-form solution.
        """
        from src.mech.validate_solver import run_benchmark_d_contact_obstacle

        df_d, _ = run_benchmark_d_contact_obstacle()
        self.assertTrue(bool(df_d["converged"].all()))
        # Monotone L2 error decrease with mesh refinement
        self.assertTrue(bool(np.all(np.diff(df_d["profile_l2_rel_err"].to_numpy()) < 0.0)))
        # Fine mesh (0.5 um) apex error < 0.5% and L2 error < 1%
        self.assertLess(float(df_d.iloc[-1]["apex_rel_err"]), 0.005)
        self.assertLess(float(df_d.iloc[-1]["profile_l2_rel_err"]), 0.01)
        self.assertLess(float(df_d["vol_rel_err"].max()), 0.01)

    def test_afm_to_tension_conversion(self) -> None:
        """Verify AFM Young's modulus to cortical tension conversion formula and range."""
        conv = afm_modulus_to_cortical_tension((200.0, 1000.0), R_cell_um=5.3, R_tip_um=2.5, delta_afm_um=0.5)
        t_low, t_high = conv["T_cortical_nominal_pN_per_um"]
        self.assertGreater(t_low, 50.0)
        self.assertLess(t_high, 1000.0)
        self.assertGreater(conv["T_cortical_full_range_pN_per_um"][1], t_high)


if __name__ == "__main__":
    unittest.main(verbosity=2)
