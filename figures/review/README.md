# Visual Review Gallery (`figures/review/`)

All 128 panels in `figures/review/<case>/<id>.png` render **real microscopy / segmentation data** from the deposited 3D U-Net masks (`data/scidb_19235_masks/<experiment>/<roi>/<stem>_predictions.h5`), local 3D raw fluorescence volumes (`data/scidb_19235_raw/` and `data/scidb_19235_chambers/` wherever downloaded locally), and 2D 405-nm SYTOX death-dye maximum-intensity projections (`data/scidb_19235_mips/<experiment>/<roi>/<stem>_MIPxy_405.tif` and `*_MIPxz_405.tif`), overlaid with the actual outputs of our measurement and solver code.

## Voxel Pitch and Coordinate Conventions
- Lateral and nominal axial voxel size: `VOXEL_UM = 0.1962 um` (`z = 0.196 um`, from `src/constants.py`).
- Every 3D contact panel shows an **XY slice (`z = z_c`)** and an **XZ slice (`y = y_c`)** passing directly through the 3D centroid `(z_c, y_c, x_c)` of the segmented contact patch (`is_contact_on_b_surf`), with a `5 um` scale bar (`5.0 / 0.1962 = 25.48 px`).

## Per-Case Overlay Definitions & Code Functions

### 1. `cup_measurement` (16 panels)
- **Source data**: `data/step5/per_visit_descriptors.csv` and `data/scidb_19235_masks/<exp>/<roi>/<stem>_predictions.h5` (plus local raw 561-nm membrane + 488-nm actin `.h5` where present).
- **Overlays & functions**:
  - **Target mask outline (`#1F4E79` solid)** and **T-cell mask outline (`#C0561F` solid)**: 2D `0.5`-level contour of the solid connected cell masks extracted by `_extract_tcell_and_target_masks_from_h5` (`src/mech/run_step6b.py`) using `fast_instance_seg_with_hint` and `fill_solid_cell_mask` (`src/contact_descriptors.py`).
  - **Contact patch voxels (`#6B4C9A` squares)**: Target boundary voxels within the contact patch (`is_contact_on_b_surf` in `process_one_chamber_descriptors`, `src/contact_descriptors.py`).
  - **Target convex hull (`#2E7D5B` dashed / dotted)**: 2D slice convex hull (`scipy.spatial.ConvexHull`) and 3D target convex hull cross-section (`hull_b = ConvexHull(pts_b_surf_um)` in `compute_descriptors_for_z_pitch`, `src/contact_descriptors.py`, intersected with the slice plane via `intersect_mesh_with_plane`).
  - **Measured cup depth (`#B3261E` double-headed arrow)**: 95th-percentile inward distance `cup_depth_hull_p95_um` from the target's T-cell-facing 3D convex hull facets (`facing = (eqs[:, :3] @ n_syn) > 0.05`) to the contact patch surface (`compute_descriptors_for_z_pitch` in `src/contact_descriptors.py`).

### 2. `fit_reachable` (16 panels) & 3. `fit_unreachable` (16 panels)
- **Source data**: `data/step6b/gateB_visit_fits.csv` (20 in-depth chambers, 207 pre-death contact visits).
- **Overlays & functions**:
  - Each visit is re-solved at its fitted dimensionless push `Pi = pi_fit_delta` (`fit_reachable`) or at the grid ceiling `Pi = 12.0` (`fit_unreachable`) using `_prepare_visit_meshes_at_edge` (`src/mech/run_gateB_check.py`) and `solve_two_shell_equilibrium` (`src/mech/shell_model.py`, `gtol = 5e-2 pN`, `maxiter = 450`, `alpha_syn = 2.0`, `chi_t = 1.0`, `subtract_prestress = True`).
  - **Real target & T-cell mask outlines (`#1F4E79`, `#C0561F` solid)**: Slices of the true 3D segmentation masks.
  - **Reference (unloaded) target outline (`#2E7D5B` dotted)**: Cross-section of `g_mesh.V0` (pre-contact target mesh if available, or cup-filled reference via `make_cup_filled_reference_mesh`, aligned by `_align_reference_target_to_visit` and overlap-resolved by `resolve_initial_contact_overlap`).
  - **Solved target mesh cross-section (`#6B4C9A` dashed)** and **Solved T-cell mesh cross-section (`#111111` dashed)**: Exact plane intersections (`intersect_mesh_with_plane`) of the equilibrium 3D meshes `sol["V_target"]` and `sol["V_tcell"]`.
  - **Depth-vs-Pi curve inset**: Plots the visit's 9-node Gate-B curve `d_delta(Pi) = d(Pi) - d(0)` and `d_raw(Pi)` across `Pi in [0, 0.6, 1.2, 2.4, 4, 6, 8, 10, 12]` with the observed cup depth horizontal line (`cup_depth_hull_p95_um`) and inverted `Pi` (`_invert_depth_curve_no_extrap` in `src/mech/run_gateB_check.py`).

### 4. `initial_overlap` (12 panels)
- **Source data**: `data/step6b/gateB_visit_fits.csv` and corresponding 3D masks.
- **Overlays & functions**:
  - **Before push-apart (`#B3261E` dashed target, `#C0561F` dotted T cell)**: Reference target mesh `V_g_ref_v` aligned by `_align_reference_target_to_visit` (`src/mech/run_step6b.py`) and segmented T-cell mesh `V_t_v` extracted by `extract_cell_mesh_marching_cubes` (`src/mech/shell_model.py`), prior to overlap resolution (`min_sep_before_push_um` from `TwoShellSystem._contact_energy_and_grad`).
  - **After push-apart (`#1F4E79` solid target, `#2E7D5B` solid T cell)**: Meshes translated apart along `syn_dir` by `push_apart_um` via `resolve_initial_contact_overlap` (`src/mech/shell_model.py`, enforcing `min_sep >= 0.16 um` at `Pi = 0`).

### 5. `mesh_size` (6 panels)
- **Source data**: `data/step6b/gateB_visit_fits.csv` (including the 3 benchmark visits in `data/step6b/gateB.json` plus 3 extreme/stratified visits).
- **Overlays & functions**:
  - Each visit is extracted and solved at both `target_edge_um = 1.0 um` (~250–350 vertices/shell) and `target_edge_um = 0.5 um` (~1,000–1,400 vertices/shell) via `extract_cell_mesh_marching_cubes`, `_prepare_visit_meshes_at_edge`, and `solve_two_shell_equilibrium` at the same `Pi`.
  - Overlays both `1.0 um` (`#1F4E79` target, `#C0561F` T cell) and `0.5 um` (`#2E7D5B` dashed target, `#6B4C9A` dotted T cell) solved mesh cross-sections on the real XY and XZ mask slices.

### 6. `death_labels` (30 panels: 12 killed + 12 surviving + 6 dasatinib, including 3 borderline cases)
- **Source data**: `data/step4/per_chamber_death_labels_all_v2.csv`, `data/step4/per_visit_death_labels_all_v2.csv`, `data/scidb_19235_mips/<exp>/<roi>/<stem>_MIPxy_405.tif`, `*_MIPxz_405.tif`, and `*_predictions.h5`.
- **Overlays & functions**:
  - Displays the 405-nm SYTOX death-dye MIP (`MIPxy_405` and `MIPxz_405`) at 3 visits (before, at, and after `death_visit_200m` for killed chambers; before, at peak 405-nm overlap/intensity, and after for surviving/dasatinib chambers; including the 3 borderline chambers closest to the 20-px overlap / 404.0-ADU threshold).
  - **Target projection outline (`#1F4E79` solid)**: `b_xy = ((pred == 2) | (pred == 4)).max(axis=0).T` and `b_xz = ((pred == 2) | (pred == 4)).max(axis=1).T` (`_process_one_chamber` in `src/death_labels.py`).
  - **T-cell projection outline (`#C0561F` dashed)**: `t_xy = ((pred == 1) | (pred == 3)).max(axis=0).T` and `t_xz = ((pred == 1) | (pred == 3)).max(axis=1).T`.
  - **Cleaned dead-dye mask outline (`#B3261E` solid)**: `compute_2d_dead_mask(mip, THRESH_2D_ADU=404.0, MIN_COMPONENT_AREA_2D=20)` in `src/death_labels.py` (2D Gaussian filter `sigma = 3.0`, threshold `> 404.0 ADU`, `disk(3)` morphological closing, 8-connected area opening `>= 20 px`). A visit is called `dead_visit = True` iff `overlap_xy_px >= 20` and `overlap_xz_px >= 20`; a chamber is called `ever_dead = True` iff `>= 2` consecutive visits are positive.

### 7. `dasatinib_vs_untreated` (16 panels: 8 dasatinib + 8 untreated from matched donors `hd2`, `hd10`, `hd5`)
- **Source data**: `data/step5/per_visit_descriptors.csv` across paired experiments (`v08_hd2` vs `v11_hd2_ds`, `v09_hd10` vs `v12_hd10ds`, `v10_hd5` vs `v13_hd5_ds`).
- **Overlays & functions**: Same real mask slices and `src/contact_descriptors.py` overlays as Case 1, highlighting `cup_depth_hull_p95_um`, `patch_equiv_radius_um`, and `cup_depth_over_radius` across matched donors.

### 8. `deep_vs_shallow` (16 panels: 8 top-decile + 8 bottom-decile of `cup_depth_over_radius`)
- **Source data**: Untreated pre-death contact visits (`visit <= 20`, `trace_num == 1`) from `data/step5/per_visit_descriptors.csv` merged with `ever_dead_200m` from `data/step4/per_chamber_death_labels_all_v2.csv`.
- **Overlays & functions**: Same real mask slices and `src/contact_descriptors.py` overlays as Case 1 for visits in the top decile (`>= p90 = 0.644`) and bottom decile (`<= p10 = 0.128`) of `cup_depth_over_radius`, stratified by target outcome (`Killed` vs `Survived`).
