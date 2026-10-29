---
id: cup-measurement
version: v1
date: "2026-10-28"
status: draft
---

# Protocol: 3D Contact Patch and Cup Depth Measurement (`cup-measurement-v1`)

## Purpose

Quantify macro-scale (`>= 1.4 um` spatial wavelength) 3D synaptic contact geometry and target cell indentation depth from segmented T-cell and target masks (`*_predictions.h5`), as currently implemented in `src/contact_descriptors.py`.

**Status**: `draft` (unverified; blocked at Gates `G1` and `G3` because 3D convex-hull cup depth is inflated when the target mask is fragmented, merged, hollow, or truncated by the FOV edge).

## 1. Resolution Cutoff & Axial Pitch Dual-Evaluation

1. **Spatial wavelength cutoff (`>= 1.4 um`)**:
   Spectral comparison against raw 561-nm membrane stacks (`.agy_step5res_report.md`) showed that the 3D U-Net masks attenuate `> 50%` of surface roughness power below `lambda_50% = 1.38 um` (synapse) / `1.45 um` (non-contact) and dilate the target boundary outward by `~0.72 um`. All geometric descriptors are restricted to whole-cell and macro-cup scales (`>= 1.4 um` wavelength; smallest trustworthy bump diameter `>= 0.7–1.0 um` laterally, `>= 0.9–1.2 um` axially). High-frequency residual variance below `1.4 um` is recorded solely as a discretization noise check (`rough_below_cutoff_noise_um2`, `rough_below_cutoff_frac`).
2. **Dual axial voxel pitch (`z = 0.1962 um` vs `z = 0.2400 um`)**:
   Lateral voxel pitch is `VOXEL_UM = 0.1962 um` (`sqrt(0.0385 um^2)`). Because uncontacted targets span `~19%` fewer Z slices than XY pixels (`Delta Z / Delta XY = 0.807`), every descriptor is evaluated at both nominal isotropic `dz = 0.1962 um` and alternative `dz = 0.2400 um` (`_z024` columns). Descriptors whose median shifts by `> 10%` between the two axial pitches are flagged in `data/step5/descriptor_dictionary.csv`.

## 2. Cell & Contact Surface Extraction (`process_one_chamber_descriptors`)

1. **Instance segmentation**:
   Runs `instance_seg_matlab_xyz` in MATLAB `(X, Y, Z)` dimension order (`transpose(2, 1, 0)`):
   - T-cell instances from nucleus `pred == 1` (`bwareaopen >= 3000` voxels) and membrane `pred == 3` (Voronoi assignment within 25-voxel sphere of nucleus, `>= 1000` voxels, 26-connected hole fill).
   - Target instances from nucleus `pred == 2` and membrane `pred == 4`.
2. **Tracked T cell and contacted target body**:
   - Selects the tracked T-cell instance (`tracked_tcell_id` from `src/reproduce_contact_area_tracked.py`).
   - Computes the Voronoi contact surface `surface_mask = (map_exp == 2) & imdilate(map_exp == 3, sphere(1))` and overwrites the peripheral radius-5 rim (`edge_mask = surface_mask & imdilate(map_exp == 1, sphere(5))`) with `-j`, leaving interior synaptic contact voxels `s_real = (surface == +j)`.
   - Selects the largest 26-connected patch component `s_largest_mask` of `s_real` and identifies the contacted target body `b_body` (`fill_solid_cell_mask`: 26-connected target component containing the contacted target instance, 3D hole-filled plus 2D slice hole-filled along Z, Y, and X).
   - Extracts target surface boundary voxels `b_surf = b_body & ~binary_erosion(b_body, STRUCT6)` and marks contact patch boundary voxels `is_contact_on_b_surf`.

## 3. Geometric & Indentation Descriptors (`compute_descriptors_for_z_pitch`)

Let `pts_b_surf_um` be the physical coordinates `(z * dz, y * dxy, x * dxy)` of all target boundary voxels and `pts_c_um` the physical coordinates of the synaptic contact patch voxels on the target surface:

1. **Contact patch size & orientation**:
   - `contact_area_um2 = s_largest_vox * 0.0385 * area_stretch`
   - `patch_equiv_radius_um = sqrt(contact_area_um2 / pi)` (`a`)
   - Unit synaptic normal `n_syn`: unit vector from target centroid `b_center_um` toward T-cell centroid `t_center_um` (representing the inward indentation direction into the target).
   - Orthonormal tangent plane basis `(e1, e2)` perpendicular to `n_syn`; `patch_elongation = sqrt(lambda_max / lambda_min)` of the 2D covariance of `pts_c_um` in `(e1, e2)`.
2. **3D Convex-Hull Indentation (`cup_depth_hull_p95_um`)**:
   - Computes the 3D convex hull `hull_b = scipy.spatial.ConvexHull(pts_b_surf_um)`.
   - Selects hull facets facing the T cell: `facing = (eqs[:, :3] @ n_syn) > 0.05`.
   - For each contact patch point `x in pts_c_um`, computes the inward perpendicular distance to the closest facing hull plane:
     `d_hull_c(x) = max(0, min_{f in facing} -(n_f . x + offset_f))`.
   - `cup_depth_hull_p95_um`: 95th percentile of `d_hull_c` across the contact patch (`d`).
   - `cup_depth_hull_max_um`: maximum of `d_hull_c`.
   - `cup_volume_hull_um3`: `sum(d_hull_c) * (contact_area_um2 / N_c)`.
   - `cup_depth_over_radius`: dimensionless focusing ratio `cup_depth_hull_p95_um / max(0.20, patch_equiv_radius_um)` (`d / a`).
   - `tcell_protrusion_vol_um3` and `tcell_protrusion_frac`: volume and fraction of tracked T-cell voxels lying strictly inside `hull_b`.
3. **Regularized Reference-Sphere Indentation (`cup_depth_sphere_p95_um`)**:
   - Fits a sphere `(c_sph, R_sph)` to the non-contact target boundary points `pts_nc_um` regularized toward `(b_center_um, r_eq_target)`.
   - `d_sph_pos(x) = max(0, R_sph - ||x - c_sph||)`, summarized by its 95th percentile `cup_depth_sphere_p95_um` and integrated volume `cup_volume_sphere_um3`.
4. **Per-Visit and Per-Chamber Aggregation**:
   - Per-visit descriptors average across contact sweeps (`in_contact == True`) within each visit (`data/step5/per_visit_descriptors.csv`).
   - Pre-death chamber summaries (`src/step5b_association_check.py`) evaluate `(A)` the first contact visit strictly before death and `(B)` the mean across all contact visits strictly before death (`visits 1–20`).

## 4. Known Failure Modes Blocking Gates `G1` & `G3`

Visual review (`status/visual_review_2026-10-28.md`) established that `cup_depth_hull_p95_um` measures false indentation whenever the target mask violates single-cell solidity:
- **Fragmented target masks** (`cup_01`: `10.15 um`, `cup_03`: `9.18 um`, `cup_10`: `5.10 um`): `hull_b` bridges across separate pieces and measures the inter-piece gap.
- **Two merged targets** (`deep_07`: `5.60 um`): `hull_b` bridges the groove between two distinct cells.
- **Hollow shell masks** (`dasatinib_11`: `6.75 um`): interior void inflates hull distance.
- **FOV edge truncation** (`cup_05`, `cup_13`): flat image border distorts hull geometry.
Must be restricted to `clean-cell-rule@v1` verified visits before `G3` signoff.
