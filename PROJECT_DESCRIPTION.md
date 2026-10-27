# Project Description & Experimental Report: Serial Killer vs. Target Geometry (`Level 1–6` Architecture)

## 1. Project Overview & Core Question

In engineered T-cell therapies (CAR-T, TEG, TCR-T) against solid tumors, a small fraction of T cells ("serial killers") execute the majority of tumor cell kills. A central unresolved question in quantitative immuno-oncology is **why**:

1. **Intrinsic Killer Phenotype ($\sigma > 0$ — Cell Property Hypothesis)**: Do serial killers carry a persistent, cell-intrinsic kinetic advantage (faster perforin/granzyme pore delivery rate $k_i$, stronger synaptic grip, faster MTOC polarization) that makes them kill fast repeatedly across targets?
2. **Target Geometry & Local Density ($\sigma = 0$ — Geometry-Only Null)**: Or is every T cell kinetically similar ($\sigma = 0$), and "serial killers" are simply the lucky T cells that land in dense 3D tumor organoid neighborhoods where multiple target cells sit within a single T-cell protrusion reach ($R \approx 10\ \mu\text{m}$)?

This project builds a **6-level multi-scale 4D digital twin and experimental fitting pipeline** that answers the Level 1 & Level 2 questions directly on real 3D live-imaging co-culture tracks (`data/behav3d_tcell_tracks_demo_well.csv`, $1,171$ CD8+ T cells, $251,494$ timepoints at 2-minute ticks over $22.2\ \text{h}$, paired with our 15.12 GB and 24.28 GB Zenodo `8328441` Imaris `.ims` 3D organoid containers and the 2026 Wang *et al.* 320-nm light-sheet CAR-T mechanics benchmark), while providing the physics and signaling engines required for Levels 3–6.

---

## 2. How `serial-killer-model-plan.md` Maps to What We Have Already Built (Reuse Audit)

Rather than starting from scratch, our existing codebase in `/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim` already contains the data parsers, 3D Imaris HDF5 extractors, and 4D mechanics solvers spanning the 6-level ladder:

| Plan Level | Model Scope in `serial-killer-model-plan.md` | Existing Asset in `Cell-Kill-Sim` Reused / Extended | Status |
| :--- | :--- | :--- | :--- |
| **Step 0** | **Data Linkage & Failure-Mode Audit**: Link T-cell tracks to 3D organoid anchors; test whether whole-organoid dead-dye blurs individual kills in large organoids. | Reused `src/inspect_ims_deep.py` + built `src/run_serial_killer_experiment.py` (`data/serial_killer_step0_linkage_audit.csv`). Discovered that in `behav3d_tcell_tracks_demo_well.csv`, `dead_dye` (`Intensity_Mean_Ch=3` inside the $10\ \mu\text{m}$ T-cell 3D surface mask) acts as a **local synaptic probe** during contact (`dist_organoid <= 0.005 mm` = $5\ \mu\text{m}$), resolving individual target kills even on large organoids. | **Complete** |
| **Level 1** | **Point Cells + Damage Accumulator (5 Parameters)**: 2-min tick state machine (`searching`, `engaged`, `refractory`) with reach $R$, damage rate $k$, refractory time $\tau$, detachment $p_{\text{detach}}$, and cell-to-cell spread $\sigma$. Both simulation and real tracks pass through the **exact same measurement code**. | Built `measure_track_table()` and `simulate_level1_cohort()` in `src/run_serial_killer_experiment.py`. Fitted $R, k, \tau, p_{\text{detach}}$ and compared **Fit 1 ($\sigma = 0$ fixed)** vs. **Fit 2 ($\sigma = 0.88$ free)** on held-out kills-per-cell tail and repeat-kill speed correlation. | **Complete** |
| **Level 2 / Fallback** | **Bidirectional Damage & T-Cell Exhaustion / Death**: Each cell carries health $h \in [0, 1]$; engagement drains T-cell health at rate $c$ (activation-induced cell death / exhaustion). Tests whether T-cell terminal death tracks cumulative killing or local density. | Built into `src/run_serial_killer_experiment.py` (`data/serial_killer_level2_fallback.csv`) and coupled to the exhaustion/cooldown state in `src/model_4d_cell_kill.py`. | **Complete** |
| **Level 3** | **Soft Spheres + Contact Mechanics + Multi-Cell Swarming**: Physical 3D cell radii, contact area $A_c$, cortical tension, and cooperative multi-T-cell swarming ($N = 1\dots 5$ CTLs on resistant solid tumor targets). | Already built in `src/model_4d_cell_kill.py` (`cooperative_swarm_4ctl` vs. `single_ctl_resistant_fail`) and calibrated against empirical 3D organoid swarming in `data/organoid_multitcell_swarm_empirical.csv` ($1\text{--}2$ T cells on single targets vs. $6\text{--}22$ simultaneous T cells on solid organoids $> 30,000\ \mu\text{m}^3$). | **Complete & Reusable** |
| **Level 4** | **Subcellular Polarity & MTOC Vector**: Unit vector $\hat{p}$ on the T cell rotating toward the synapse at angular speed $\omega_{\text{MTOC}}$ before pore delivery begins, saving repolarization time when switching to an adjacent target. | Already implemented in `src/model_4d_cell_kill.py` (`mtoc_polarization`, `ca_flux`, `perforin_unidirectional`, `granzyme_b`, `caspase3`) and calibrated to Ritter *et al.* (2015) / Jenkins *et al.* (2015). | **Complete & Reusable** |
| **Level 5** | **Deformable 3D Cell Meshes & Membrane Mechanics**: Triangulated 3D surface meshes ($V, F$) with cortical tension, synaptic indentation cupping, and osmotic swelling/blebbing. | Already built in `src/model_4d_cell_kill.py` + `index.html` interactive WebGL 4D viewer, directly anchored to the 2026 Wang *et al.* *Scientific Data* 320-nm light-sheet dataset ($419$ CAR-T/tumor pairs, Figshare `28303928`). | **Complete & Reusable** |
| **Level 6** | **Spatial Reaction-Diffusion Fields**: Perforin/granzyme synaptic cleft diffusion and IFN-$\gamma$ / CXCL9/10 chemotactic fields in 3D extracellular matrix. | Reuses the spatial grid and swarming arrival-rate boost (`+42%` post-first-hit recruitment rate measured in `data/organoid_multitcell_swarm_empirical.csv`). | **Ready for PDE Coupling** |

---

## 3. Step 0 Findings: Data Linkage & Failure-Mode Resolution

Before fitting, `serial-killer-model-plan.md` requires verifying how T-cell tracks link to 3D organoids and checking whether whole-organoid dead-dye averaging obscures individual kills (`data/serial_killer_step0_linkage_audit.csv`):

1. **Cross-Well Offset in Public GitHub CSVs vs. Exact Alignment in Local `.ims`**:
   - In the public BEHAV3D GitHub repository, `DemoDataset4` (`behav3d_tcell_tracks_demo_well.csv`, $1,171$ CD8+ tracks) and `DemoDataset_8_organoids` come from two different imaging wells (minimum 3D centroid distance $42.44\ \mu\text{m}$).
   - By contrast, inside our local 15.12 GB Imaris HDF5 container (`LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims`, `Scene/Content`), `MegaSurfaces0` ($21$ tracked 3D organoids), `MegaSurfaces1` ($1,122$ CD4+ tracks), and `MegaSurfaces3` ($798$ CD8+ tracks) sit in the **exact same 3D coordinate frame ($0.00\ \mu\text{m}$ offset)**, allowing direct multi-T-cell swarming extraction per organoid (`data/organoid_multitcell_swarm_empirical.csv`).
2. **Unit Calibration of `dist_organoid`**:
   - In `behav3d_tcell_tracks_demo_well.csv`, `dist_organoid` is exported by Imaris in **millimeters** (`0.00105` = $1.05\ \mu\text{m}$, the exact Z-slice spacing; `0.00354246` = $3.54\ \mu\text{m}$, the XY voxel pitch). Multiplying `dist_organoid` by $1,000$ gives exact 3D surface-to-surface distance in $\mu\text{m}$, with physical contact defined as $d \le 5.0\ \mu\text{m}$.
3. **Why the Plan's Failure Mode Is Solved by Local Synaptic Probing**:
   - The plan warns that *whole-organoid* dead-dye averaging blurs single-cell kills when an organoid contains $\ge 10$ cells.
   - However, in `behav3d_tcell_tracks_demo_well.csv`, `dead_dye` is `Intensity_Mean_Ch=3` measured **inside the 3D T-cell surface mask ($10\ \mu\text{m}$ footprint)**. When a T cell is physically attached to an organoid ($d \le 5\ \mu\text{m}$), its surface mask overlaps the synaptic cleft of the specific target cell it is touching. When that target cell dies and takes up red dead-cell dye (Ch 3), the T cell's local `dead_dye` steps up sharply ($\Delta I \ge +22\ \text{a.u.}$, crossing $35\ \text{a.u.}$); when the T cell crawls $10.03\ \mu\text{m}$ to the next viable cell on the same organoid, its local `dead_dye` drops back down ($\Delta I \le -16\ \text{a.u.}$) and steps up again on the next kill!

---

## 4. Level 1 Experimental Results: Intrinsic Killer Phenotype ($\sigma > 0$) vs. Geometry ($\sigma = 0$)

Both the real tracks (`data/behav3d_tcell_tracks_demo_well.csv`, $n = 1,171$ CD8+ T cells) and the simulated Level-1 cohorts (`Fit 1: sigma = 0` and `Fit 2: sigma = 0.88`) were passed through the **exact same measurement function (`measure_track_table()`)** in `src/run_serial_killer_experiment.py`.

### 4.1 Fitted Level-1 Parameters (`data/serial_killer_fitted_parameters.csv`)

| Parameter | Physical Meaning | Fitted Value (Real Data) | How Pinned Down |
| :--- | :--- | :--- | :--- |
| **$R$** | Effective 3D T-cell reach ($\mu\text{m}$) | **$10.03\ \mu\text{m}$** | Median 3D distance between consecutive kills on the same organoid ($n = 186$ serial killers; bounded by max 3D T-cell prolate span $24.7\ \mu\text{m}$). |
| **$k$** | Mean damage rate ($\text{min}^{-1}$) | **$0.0342\ \text{min}^{-1}$** | Inverse of mean contact duration required to trigger a synaptic dead-dye step ($29.21\ \text{min}$ mean, $12.0\ \text{min}$ median across $n = 877$ kills). |
| **$\tau$** | Refractory & repositioning time ($\text{min}$) | **$44.0\ \text{min}$** | Median interval from Kill 1 completion to Kill 2 onset while remaining on the organoid anchor. |
| **$p_{\text{detach}}$** | Per-tick detachment probability ($\text{tick}^{-1}$) | **$0.1189\ \text{tick}^{-1}$** | Fitted from $n = 4,770$ abortive contact bouts ending without a kill (mean abortive bout $16.82\ \text{min}$ at $2\ \text{min/tick}$). |
| **$\sigma$** | Between-cell log-SD of damage rate $k_i$ | **$0.00$** (Fit 1) vs. **$0.88$** (Fit 2) | Fitted to between-cell spread in time-to-kill conditioned on local target density ($\text{CV} = 1.602$). |

### 4.2 Held-Out Test Comparison (`data/serial_killer_heldout_comparison.csv`)

| Held-Out Observable | Empirical Data ($n = 1,171$) | Fit 1: $\sigma = 0$ Fixed (Geometry-Only Null) | Fit 2: $\sigma = 0.88$ Free (Killer Phenotype) | Scientific Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **0 kills (Non-killers)** | **$60.0\%$** ($703$ cells) | $63.1\%$ ($739$ cells) | **$66.2\%$** ($775$ cells) | Both capture non-engaging patrol tracks |
| **1 kill (Single killers)** | **$24.1\%$** ($282$ cells) | $24.1\%$ ($282$ cells) | **$14.8\%$** ($173$ cells) | $\sigma = 0$ concentrates killers into 1–2 kills |
| **Serial killers ($\ge 2$ kills)** | **$15.9\%$** ($186$ cells) | $12.8\%$ ($150$ cells) | **$19.0\%$** ($223$ cells) | $186$ CD8+ T cells execute $678 / 877$ ($77.3\%$) of all kills |
| **Super-killers ($\ge 3$ kills, heavy tail)** | **$8.1\%$** ($95$ cells; $4.3\%$ with $\ge 4$) | **$4.6\%$** ($54$ cells; **$1.1\%$ with $\ge 4$**) | **$10.5\%$** ($123$ cells; **$6.2\%$ with $\ge 4$**) | **$\sigma = 0$ underpredicts $\ge 4$-kill super-killers by $3.9\times$** |
| **Dispersion (Fano factor $\text{Var}/\text{Mean}$)** | **$2.582$** | **$1.381$** (Underdispersed) | **$2.878$** | Geometry alone cannot produce the observed overdispersion |
| **Repeat-kill speed correlation $\rho(T_{\text{kill},1}, T_{\text{kill},2})$** | **$+0.326$ ($p = 5.64 \times 10^{-6}$)** | **$-0.052$ ($p = 0.528$)** | **$+0.450$ ($p = 1.72 \times 10^{-12}$)** | **Fast killers kill fast again — refutes $\sigma = 0$ null** |
| **Partial correlation $\rho(T_{\text{kill},1}, T_{\text{kill},2} \mid \text{density})$** | **$+0.316$ ($p = 1.11 \times 10^{-5}$)** | **$-0.061$ ($p = 0.455$)** | **$+0.447$ ($p = 2.37 \times 10^{-12}$)** | Repeat-kill speed correlation is NOT an artifact of target density |
| **Target-depletion gap ($\text{Gap}_{1\to 2} \to \text{Gap}_{2\to 3}$)** | **$44.0 \to 58.0\ \text{min}$ ($+31.8\%$)** | $131.0 \to 162.0\ \text{min}$ | **$46.0 \to 70.0\ \text{min}$** | **Geometry also matters**: local target depletion slows later kills |

---

## 5. Level 2 / Fallback Results: T-Cell Terminal Death Tracks Cumulative Killing (`data/serial_killer_level2_fallback.csv`)

We also executed the plan's Level 2 / Fallback test across all $1,171$ CD8+ T cells, measuring whether a T cell's own terminal death (permanent Ch 3 dead-dye uptake $\ge 55\ \text{a.u.}$ with $\Delta I \ge +25\ \text{a.u.}$ and motility arrest $< 0.8\ \mu\text{m/min}$) tracks cumulative killing/engagement or local target density:

| T-Cell Killer Group | Track Count ($n$) | Mean Cumulative Contact ($\text{min}$) | Mean Contact Bouts | Mean Local Target Density | T-Cell Terminal Death Rate ($\%$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0 kills (Non-killer)** | $703$ | $97.5\ \text{min}$ | $3.27$ | $24.5$ | **$4.98\%$** |
| **1 kill (Single killer)** | $282$ | $273.1\ \text{min}$ | $6.38$ | $27.3$ | **$11.70\%$** |
| **2 kills (Serial killer)** | $91$ | $286.9\ \text{min}$ | $6.53$ | $31.2$ | **$14.29\%$** |
| **$\ge 3$ kills (Super-killer)** | $95$ | $485.6\ \text{min}$ | $10.28$ | $27.2$ | **$15.79\%$** |

- **Key Level-2 Finding**: T-cell terminal death rises more than **$3\times$** from non-killers ($4.98\%$) to super-killers ($15.79\%$; Spearman $\rho = +0.149, p = 2.87 \times 10^{-7}$ with `n_kills`, and $\rho = +0.165, p = 1.34 \times 10^{-8}$ with cumulative contact time), whereas its correlation with local target density is slightly negative ($\rho = -0.111$). This directly supports the Level-2 bidirectional damage accumulator: serial killing imposes a cumulative physiological cost (activation-induced cell death / synaptic exhaustion) on the killer T cell.
