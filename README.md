# Cell-Kill-Sim

A multi-scale 4D modeling, shape reconstruction, and empirical analysis pipeline for engineered T-cell (CAR-T and TEG) tumor killing across 3D confocal organoid co-cultures and 320-nm isotropic light-sheet single-cell microscopy.

The pipeline reconstructs 3D/4D single-cell and multicellular organoid geometries from deposited microscopy volumes and semantic segmentation masks, couples molecular-clutch synaptic traction and Hertzian/cortical-shell mechanics to tension-potentiated perforin pore delivery and ESCRT membrane repair ODEs, and evaluates out-of-sample perturbation benchmarks (including Lck inhibition by Dasatinib 25/50 nM across 419 CAR-T/Nalm6 cell pairs and volume decay across 16 shrinking breast cancer patient-derived organoids).

At the population and track level, the repository tests whether observed "serial killer" T-cell behavior arises from a persistent cell-intrinsic killing rate advantage ($\sigma > 0$) or from local 3D target geometry, crowding, and finite track fragmentation ($\sigma = 0$). Simulation cohorts and live-imaging T-cell tracks ($N = 6,931$ tracks across 6 BEHAV3D wells) are evaluated through identical measurement functions alongside model-free within-organoid co-residency and fragment-aware renewal null tests.

## Repository Layout

```text
.
├── 4D_DIGITAL_TWINS.md
├── DENT_AND_STIFFNESS_CLAUDE_REVIEW.md
├── PROJECT_DESCRIPTION.md
├── Single-cell 4D digital twin of T cell killing project brief.md
├── Single-cell 4D digital twin of T cell killing project brief 2.md
├── VISUAL_GUIDE.md
├── serial-killer-model-plan.md
├── requirements.txt
├── src/
│   ├── extract_*.py
│   ├── model_*.py
│   ├── run_*.py
│   ├── upgrade_*.py
│   ├── build_*.py
│   ├── render_*.py
│   └── v3_*.py
├── data/        # Gitignored: extracted tables, cached chambers, and canonical JSON metrics
├── videos/      # Gitignored: rendered MP4 and GIF side-by-side animations
├── external/    # Gitignored: external repositories and supplementary spreadsheets
└── *.ims        # Gitignored: raw Imaris HDF5 3D confocal time-lapse containers
```

## Source Scripts (`src/`)

| Script | Category | Purpose |
| :--- | :--- | :--- |
| `src/extract_behav3d_and_ims.py` | Extraction | Extracts 3D surface morphology, metadata, and track statistics from Zenodo `8328441` (`LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims`) and BEHAV3D `.rds` outputs. |
| `src/extract_lightsheet_and_exp28.py` | Extraction | Extracts CD19 CAR-T + B-ALL surfaces from `LSM880_example_dataset_CART_ALL_Exp28.ims` and compiles single-cell light-sheet tables from `Source_Data_Wang2024_NatCommun.xlsx`. |
| `src/explore_multiscale_space.py` | Extraction & Modeling | Extracts the 3,598-track BEHAV3D cohort and 1,814-cell scRNA-seq behavioral states, fits the stochastic population baseline, and runs two-cell deformable mechanics sweeps. |
| `src/model_4d_cell_kill.py` | Modeling | Simulates the 4-layer 4D mechanochemical single-cell, cooperative swarm, and serial-killing model (`CellKillTwin4D`). |
| `src/upgrade_shape_extraction_and_modeling.py` | Modeling | Extracts Fourier and spherical-harmonic ($M = 16$) cell contours from light-sheet and confocal frames and quantifies 2D/3D shape-matching IoU and boundary RMSE. |
| `src/build_physically_constrained_4d_shapes.py` | Modeling | Reconstructs 3D Signed Distance Fields (SDFs) and Taubin-faired Dual-Surface-Net meshes enforcing steric non-interpenetration, volume conservation, and nuclear exclusion. |
| `src/build_multiorganelle_mitospace_4d_twin.py` | Modeling | Adds coupled 3D intracellular mitochondrial networks, microtubule asters, and directed lytic granule traffic in SDF-normalized cytoplasmic coordinates. |
| `src/upgrade_video2_and_extract_forces.py` | Modeling & Analysis | Reconstructs 3D multi-lobular organoid and swarming T-cell meshes for Organoid `1000826950` and extracts time-resolved synaptic forces, tensions, and bleb pressures. |
| `src/v3_part1_deposited_volumes_and_physics.py` | Modeling & Analysis | Rebuilds all 7 single-cell 320-nm light-sheet twins from deposited ScienceDB `19235` 3D U-Net masks and 16-bit MIPs, and evaluates the coupled clutch-mechanics and perforin/ESCRT repair ODE on the 419-pair cohort. |
| `src/run_serial_killer_experiment.py` | Analysis | Runs the Step 0 linkage audit and fits Level-1 ($\sigma = 0$ vs. $\sigma > 0$) and Level-2 bidirectional damage/exhaustion models using a shared track measurement function. |
| `src/run_serial_killer_novelty_and_counterfactuals.py` | Analysis | Deconstructs high-contact T-cell cohorts into 0-kill, 1-kill, and $\ge 2$-kill subgroups and runs counterfactual parameter perturbations across the 21 tracked organoids. |
| `src/run_model_free_same_organoid_proof.py` | Analysis | Partitions variance in T-cell killing rate within vs. between host organoids and performs paired comparisons of co-resident T cells on the same organoid. |
| `src/run_dent_predict_kill_task.py` | Analysis | Measures 3D target-membrane synaptic indentation ("dent") across ScienceDB `19235` chambers, validates on synthetic pairs, and scores out-of-sample kill prediction rules. |
| `src/run_stiffness_and_force_scale_sweeps_419.py` | Analysis | Runs 25-point logarithmic parameter sweeps over clutch coupling ($\eta_{\text{clutch}}$) and target stiffness ($E_{\text{target}}$) across the 419 CAR-T/Nalm6 pairs. |
| `src/run_dent_stiffness_fixes_all_frames.py` | Analysis | Evaluates 3D synaptic indentation across all 63,053 outlined frames (502 chambers, 13 experiments) with intact-target filters, pre-death time ordering, and small-strain Hertz caps. |
| `src/v3_part2_organoid_heldout_benchmark.py` | Analysis | Performs the exclusion audit (dropping 4 debris fragments $< 3,000\ \mu\text{m}^3$ and 1 growing organoid) and runs matched-parameter ($p_{\text{local}} = 0$ and $p_{\text{local}} = 1$) held-out benchmarks on the 16 shrinking organoids. |
| `src/v3_part3_part4_fragment_nulls_and_hypotheses.py` | Analysis | Validates confocal step-detection precision/recall against crowding, tests the fragment-aware renewal null across all 6 BEHAV3D wells ($N = 6,931$ tracks), and runs hypothesis-labeled sensitivity projections. |
| `src/build_visual_explainer_panels.py` | Rendering & Reports | Generates multi-channel overview figures comparing the 3D confocal organoid, B-ALL leukemia, and 320-nm light-sheet datasets. |
| `src/build_side_by_side_videos.py` | Rendering & Reports | Renders 4-column frame-synchronized comparison videos pairing real microscopy projections, 2D synthetic optical projections, 3D deformable meshes, and biophysical traces. |
| `src/build_generalization_events_and_study_atlas.py` | Rendering & Reports | Renders out-of-sample generalization videos (Events 3–6) and compiles the interaction event inventory. |
| `src/build_all_ground_truth_4d_twins.py` | Rendering & Reports | Builds ground-truth-only 4D visual twins across the 7 single-cell light-sheet events and the 21 confocal organoids. |
| `src/render_remaining_single_cell_videos.py` | Rendering & Reports | Parallel worker script that renders side-by-side MP4 and GIF animations for Single-Cell Twins 4–7. |
| `src/build_all_21_organoid_videos.py` | Rendering & Reports | Renders multi-plane confocal (`XY`, `XZ`, `YZ`, and optical Z-slices) plus 3D Dual-Surface-Net animations for all 21 tracked objects in the `.ims` dataset. |
| `src/build_dent_stiffness_visual_guide.py` | Rendering & Reports | Builds the visual guide figures, 3-regime comparison video, and HTML viewer (`dent_stiffness_visual_guide.html`) for the 63,053-frame synaptic dent and stiffness study. |
| `src/build_app.py` | Rendering & Reports | Compiles the self-contained interactive 4D digital twin and live-imaging comparison web application (`index.html`). |
| `src/v3_build_canonical_report.py` | Rendering & Reports | Merges canonical v3.0 metrics (`data/v3_canonical_metrics.json`), rebuilds the 21-organoid cohort wall video, templates `4D_DIGITAL_TWINS.md` and `index.html`, and runs consistency assertions. |

## Data

Raw microscopy containers, intermediate caches, and rendered video outputs (`data/`, `videos/`, `external/`, and `*.ims`) are excluded from git tracking due to file size. They originate from the following public datasets cited in the project documentation:

1. **Zenodo Record `8328441` (Dekkers *et al.*, *Nature Biotechnology* 2023 / Alieva *et al.*, 2023 — BEHAV3D)**:
   - `LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims` (15.12 GB Imaris HDF5 container: 9 channels, $200 \times 200 \times 62$ voxels at $3.5425 \times 3.5425 \times 1.0500\ \mu\text{m}$ pitch, 669 frames at $1.995\ \text{min/frame}$ over $22.21\ \text{h}$, containing 21 tracked breast cancer organoid surfaces, 1,122 CD4+ TEG tracks, and 798 CD8+ TEG tracks).
   - `LSM880_example_dataset_CART_ALL_Exp28.ims` (24.28 GB Imaris HDF5 container: CD19 CAR-T cells co-cultured with primary B-ALL blasts, 7 channels, $0.91\ \mu\text{m}$ isotropic pitch, 480 frames over $22.5\ \text{h}$).
2. **BEHAV3D Repository (`AlievaRios/BEHAV3D`) & BioImage Archive `S-BIAD448`**:
   - Multi-well T-cell behavioral tracking tables (`DemoDataset1`–`DemoDataset6`, $N = 6,931$ tracks across 6 wells / 1,282,868 cell-frame rows, and `classified_tcell_track_data.rds` / `classified_tcell_track_data_summary.rds`), organoid death dynamics (`Full_percentage_dead_org_over_time.rds`), and paired scRNA-seq behavioral states (`scRNA_seq_dataset.rds`, $n = 1,814$ CD8+ TEGs).
3. **Science Data Bank `10.57760/sciencedb.19235`, Figshare `28303928`, & Wang *et al.* (*Nature Communications* 2024, `10.1038/s41467-024-51039-9`; *Scientific Data* 2026, `10.1038/s41597-026-06829-9`)**:
   - High-throughput Bessel oblique plane light-sheet microscopy (HBOPM, $0.320\ \mu\text{m}$ optical FWHM / $0.1962\ \mu\text{m}$ reconstructed ROI voxel pitch) of CD19 CAR-T cells engaging Nalm6 B-ALL targets across 502 microfluidic chambers and 13 experimental groups ($n = 419$ curated single-cell pairs in `Source_Data_Wang2024_NatCommun.xlsx`, including Untreated High-Killing Groups 1–5, Untreated Low-Killing Groups 6–10, and Dasatinib 25/50 nM Lck-inhibited Groups 8\*, 9\*, 10\*).
   - Deposited 3D raw fluorescence HDF5 stacks (`*.h5`, shape `(3, 166, 256, 256)` `uint16`), 3D U-Net 4-label semantic segmentation masks (`*_predictions.h5`, `uint8`), 16-bit orthogonal maximum-intensity projections (`*_MIPxy_*.tif`, `*_MIPxz_*.tif`), and supplementary videos (`MOESM5_ESM.mp4`–`MOESM9_ESM.mp4`).

## Setup

Create a Python 3.10+ virtual environment and install the required Python packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Note: Video encoding functions in `src/` invoke the `ffmpeg` command-line binary via `subprocess` to encode `.mp4` and `.gif` outputs.

## Path Configuration Note

The scripts in `src/` currently define hardcoded local paths pointing to the author's workspace (e.g., `ROOT = Path("/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim")` and artifact output paths under `/Users/dxt397/.gemini/antigravity/brain/...`). Before running any script locally, update `ROOT` (and `ART_DIR` / `BRAIN_DIR` where present) to match your local checkout directory and place the required `.ims`, ScienceDB `.h5`/`.tif`, and BEHAV3D `.rds`/`.csv` inputs under `data/` and `external/`.
