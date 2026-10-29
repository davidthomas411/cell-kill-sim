# Definitions and Caveats (`lab/handbook/definitions.md`)

## 1. Voxel Size and Coordinate Conventions

- **Lateral voxel pitch (`VOXEL_UM`)**: `0.1962 um` (`src/constants.py`), derived directly from the Fei lab's contact-area voxel conversion factor `0.0385 um^2` in `external/feilab_CART_analysis/main/analysis/summary/summary.m` (`sqrt(0.0385) = 0.196214 um`).
- **Optical lateral FWHM (`OPTICAL_FWHM_UM`)**: `0.32 um` (`320 nm` isotropic Bessel light-sheet resolution reported in Wang et al. 2024).
- **Array axis ordering**:
  - Deposited HDF5 files (`*_predictions.h5` and raw `<stem>.h5`) store 3D volumes in `(Z, Y, X)` order (`shape = (Z, 256, 256)`).
  - MATLAB reads HDF5 arrays in transposed `(X, Y, Z)` order. Because `bwdist` (3D Euclidean distance transform) breaks equidistant Voronoi ties by linear scan-line index, exact numerical reproduction of the Fei lab's instance segmentation requires transposing `(Z, Y, X) -> (X, Y, Z)` via `.transpose(2, 1, 0)` prior to `distance_transform_edt` (`edt_indices_matlab_xyz` in `src/reproduce_actin_measures.py`).
  - Deposited 2D MIPs satisfy `MIPxy_405 == raw[0].max(axis=0).T` (shape `(256, 256)` in `(X, Y)`) and `MIPxz_405 == raw[0].max(axis=1).T` (shape `(256, Z)` in `(X, Z)`).

## 2. Axial Z-Spacing and U-Net Resolution Caveats

1. **Axial Z-spacing caveat (`z = 0.1962 um` vs `z = 0.2400 um`)**:
   Across uncontacted Nalm6 target cells, the 3D bounding box spans `~19%` fewer Z slices than XY pixels (`Delta Z / Delta XY = 0.807`). This corresponds either to mild physical oblateness of cells resting on the microgrid chamber floor at nominal isotropic `dz = 0.1962 um`, or to an effective post-deskew axial step of `~0.24 um` (`0.1962 / 0.807 = 0.243 um`). Every geometric descriptor in `src/contact_descriptors.py` is computed under both `dz = 0.1962 um` and `dz = 0.2400 um` (`_z024` columns). Relative descriptors (`cup_depth_over_radius`, `tcell_protrusion_frac`, `target_sphericity`) vary by `< 6%` between the two axial pitches, whereas absolute volumes scale by `0.2400 / 0.1962 = 1.223`.
2. **U-Net surface smoothing cutoff (`>= 1.4 um` wavelength)**:
   Comparison of 3D U-Net mask boundaries against raw 561-nm membrane stacks across 20 sweeps (`.agy_step5res_report.md`) established:
   - Outward boundary bias: the U-Net target mask (`(pred == 2) | (pred == 4)`) sits `+0.72 um` (`+3.69 voxels`) outside the raw 561-nm half-maximum membrane contour.
   - Spectral half-power (`-3 dB`, `P_mask / P_raw = 0.50`) cutoff wavelength: `lambda_50% = 1.38 um` on the synaptic contact patch (`1.45 um` away from contact). Below `lambda = 1.15 um`, `> 60%` of raw surface roughness power is erased by U-Net smoothing and voxel staircase discretization.
   - Therefore, sub-micron microvillar fingers (`0.1–0.5 um`, de Jesus et al. 2024) cannot be measured from the deposited masks; all valid mask descriptors are restricted to macro-scale features (`>= 1.4 um` wavelength).

## 3. Contact Patch and Cup Depth Definitions (`cup-measurement@v1`)

- **Interior synaptic contact surface (`s_real`)**:
  In `step2_instance_seg.m` and `cl_factor.m`, the Voronoi interface between tracked T cell `j` and target (`map_exp == 2` adjacent to `map_exp == 3`) has its peripheral radius-5 rim (`imdilate(background, sphere(5))`) overwritten as `s_edge = -j`. Only interior voxels (`s_real = (surface == +j)`) count toward contact area (`CA_um2 = s_largest * 0.0385`). Omitting the `s_edge` rim exclusion inflates `CA_um2` by `25–30%`.
- **Equivalent patch radius (`a`)**:
  `patch_equiv_radius_um = sqrt(contact_area_um2 / pi)`.
- **Convex-hull cup depth (`d = cup_depth_hull_p95_um`)**:
  Given the 3D convex hull `ConvexHull(pts_b_surf_um)` of the solid contacted target cell and the unit synaptic normal `n_syn` pointing from target centroid to T-cell centroid, T-cell-facing hull facets are selected by `(n_facet . n_syn) > 0.05`. For each target boundary voxel on the contact patch (`pts_c_um`), `d_hull_c` is the inward perpendicular distance from the closest facing hull facet. `cup_depth_hull_p95_um` is the 95th percentile of `d_hull_c` across the patch.
- **Focusing ratio (`cup_depth_over_radius`)**:
  `cup_depth_over_radius = cup_depth_hull_p95_um / max(0.20, patch_equiv_radius_um)` (`d / a`).
- **T-cell protrusion fraction (`tcell_protrusion_frac`)**:
  Fraction of tracked T-cell voxels lying strictly inside the contacted target's 3D convex hull.
- **Simple Laplace push proxy (`Route (b)`)**:
  On a spherical target shell of radius `R_target` and cortical tension `T_target` indented to depth `d` over patch radius `a`, quasi-static Laplace normal-stress balance gives dimensionless push `Pi_simple = p * R_target / T_target ≈ 2 + 4 * (R_target / a) * (d / a)` and net indentation push `Delta_Pi_indent = 4 * (R_target / a) * cup_depth_over_radius`.

## 4. Target Death Rule (`death-calling@v1`)

- **2D dead-dye mask (`compute_2d_dead_mask` in `src/death_labels.py`)**:
  2D Gaussian filter (`sigma = 3.0`, `truncate = 2.0`), threshold `> 404.0 ADU` (calibrated to match the Fei lab's 3D `> 398.0 ADU` threshold after the `+4.6` to `+5.5 ADU` 2D max-projection extreme-value shift), morphological closing with `disk(3)`, and 8-connected area opening `>= 20 px`.
- **Per-visit positive call (`dead_visit`)**:
  Both `overlap_xy_px >= 20` and `overlap_xz_px >= 20` between the cleaned 2D dead-dye masks (`d_xy`, `d_xz`) and the 2D projections (`b_xy`, `b_xz`) of the 3D target instance contacted by the tracked T cell, after excluding `MIP_ROI_0` `visit >= 19` saturation-band frames.
- **Per-chamber death rules**:
  - **Any-Visit (`ever_dead_any_visit_200m`)**: `>= 1` positive visit in `visits 1–20` (matches `summary.m` `C = max(B, [], 2)`).
  - **Sustained (`ever_dead_200m`)**: `>= 2` consecutive positive visits in `visits 1–20`.
