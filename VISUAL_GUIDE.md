# Visual Guide — Single-Cell & Multi-Cell 4D Digital Twin of T Cell Killing

> [!IMPORTANT]
> **Standardized Visual Encoding Across All Figures & Side-by-Side Videos (`[REAL DATA]` vs. `[SIMULATION]`)**
> - **`[REAL DATA]` (Empirical Microscopy & Ground-Truth Measurements)**: Rendered with **Ink-Black (`#111111`) dashed contours / trajectories (`--`) + open white-filled circle markers (`○`)** and **dashed-border ground-truth reference bands**.
> - **`[SIMULATION]` (4D Digital Twin Predictions & Counterfactuals)**: Rendered with **solid continuous colored curves (`—`)** in the house data palette (`#1F4E79`, `#C0561F`, `#2E7D5B`, `#6B4C9A`) and **3D Phong/Fresnel shaded meshes**.
> - **Note**: If you want the outer 4D membrane in Column C to also display the 3D cortical speckle pattern (or to texture-map the empirical fluorescence intensity from Column A directly onto the 3D mesh vertices), we can evaluate a 3D multi-scale Perlin/Gaussian field — **now active across all 4D meshes (`Visual Guides 13 & 14` and `Events 1, 3, 4, 5`) via [`evaluate_3d_cortical_speckle_field()`](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/src/build_physically_constrained_4d_shapes.py#L455-L570) fused with empirical `Column A` fluorescence texture projection.**

---

## Out-of-Sample 4D Generalization Across Three Unseen Light-Sheet Interaction Regimes (Events 3, 4 & 5)

![Out-of-sample 4D generalization across Events 3, 4, and 5](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_14_multi_event_generalization_atlas.png)

````carousel
![Side-by-side Event 3: 4K Light-Sheet Late Synapse & Envelopment (MOESM6, 0 to 200 min)](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/side_by_side_event3_lightsheet_late_synapse.gif)
<!-- slide -->
![Side-by-side Event 4: 4K Light-Sheet Sub-Lethal Contact & Abortive Exit (MOESM7, 0 to 220 min)](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/side_by_side_event4_lightsheet_sublethal_abort.gif)
````

- **Playable H.264 MP4s**:
  - **Event 3 (`MOESM6_ESM.mp4`, Late Synapse & Deep Envelopment, $0 \to 200\ \text{min}$)**: [side_by_side_event3_lightsheet_late_synapse.mp4](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/videos/side_by_side_event3_lightsheet_late_synapse.mp4)
  - **Event 4 (`MOESM7_ESM.mp4`, Sub-Lethal Contact & Abortive Detachment, $0 \to 220\ \text{min}$)**: [side_by_side_event4_lightsheet_sublethal_abort.mp4](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/videos/side_by_side_event4_lightsheet_sublethal_abort.mp4)
- **Zero-Retuning Generalization Across 3 Distinct Biophysical Regimes ([multi_event_generalization_benchmark.csv](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/data/multi_event_generalization_benchmark.csv))**:
  1. **Row A — Event 3 (`MOESM6_ESM.mp4`, $2160\times 3840$ 4K, $0 \to 200\ \text{min}$)**: Extended microchamber search ($0\text{--}110\ \text{min}$) followed by late synaptic docking ($120\ \text{min}$) and deep concave cup envelopment ($130\text{--}200\ \text{min}$, $F_{\text{syn}} = 1.55\ \text{nN}$). Frozen $M=16$ contour match: **CAR-T $\text{IoU} = 0.938$** (vs. $M=2$: $0.766$), **Target $\text{IoU} = 0.887$** (vs. $M=2$: $0.748$), **$0.000\ \mu\text{m}^3$ 3D overlap**.
  2. **Row B — Event 4 (`MOESM7_ESM.mp4`, $2160\times 3840$ 4K, $0 \to 220\ \text{min}$)**: Sustained sub-lethal contact ($10\text{--}160\ \text{min}$, peak $F_{\text{syn}} = 1.13\ \text{nN}$) followed by **abortive CAR-T detachment ($170\text{--}220\ \text{min}$)** where the intact Nalm6 target repairs its cortex, preserves its fused teal mitochondrial reticulum, and survives (`SYTOX = 0%`). Frozen $M=16$ contour match: **CAR-T $\text{IoU} = 0.924$** (vs. $M=2$: $0.728$), **Target $\text{IoU} = 0.967$** (vs. $M=2$: $0.906$). Directly anchors the **sub-lethal hit & repair regime** in `twin-study-design.md` (Studies 1 & 3).
  3. **Row C — Event 5 (`MOESM8_ESM.mp4`, Continuous High-Speed Burst at $\Delta t = 2.5\ \text{s/vol}$, $0 \to 15\ \text{min}$, $360$ volumes)**: Resolves seconds-scale cortical F-actin waves and synaptic cup oscillations ($CA = 65\text{--}155\ \mu\text{m}^2$, $F_{\text{syn}} = 0.85\text{--}1.55\ \text{nN}$). Frozen $M=16$ contour match: **CAR-T $\text{IoU} = 0.946$** (vs. $M=2$: $0.815$), **Target $\text{IoU} = 0.916$** (vs. $M=2$: $0.812$). Directly anchors **seconds-scale parameter identifiability and pore kinetics** in `twin-study-design.md` (Studies 2 & 3a).

---

## Complete Inventory of Recreatable 4D Cell Interactions & Study Execution Blueprint (`twin-study-design.md`: Studies 2 $\to$ 1 $\to$ 3)

![Inventory of recreatable 4D cell interactions and study execution blueprint](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_15_twin_study_design_readiness.png)

- **How Many Total Cell Interactions Exist on Disk That We Can Recreate? ([recreatable_interaction_events_inventory.csv](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/data/recreatable_interaction_events_inventory.csv))**:
  1. **Full 4D Orthogonal Light-Sheet Movies (`Wang et al. 2024/2026`, `external/feilab_lightsheet_data/`)**: **$7$ full 4D single-cell interaction movies** across `MOESM4_ESM.mp4`–`MOESM9_ESM.mp4` ($0.196\ \mu\text{m}$ voxel pitch, `XY` + `XZ` orthogonal views, including `SiR-Tubulin` MTOC docking, late synapse, sub-lethal abortive exit, and $2.5\ \text{s/vol}$ fast bursts) + **$419$ quantified CAR-T–Nalm6 pairs** across $13$ microchamber groups ($330$ untreated across $9$ groups + $89$ Dasatinib $50\ \text{nM}$ across $4$ groups; split **$254$ train / $165$ held-out test** by group before fitting).
  2. **3D Breast Cancer Organoid + TEG Confocal Co-Culture (`Zenodo 8328441`, `LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims`)**: **$21$ tracked 3D tumor organoids** ($512\times 512\times 26\times 256$ frames, $10.6\ \text{h}$) + **$1,839$ 3D T-cell trajectories** (**$1,171$ CD8+ TEGs** and **$668$ CD4+ TEGs**; split **$780$ train / $391$ held-out test** for CD8+ and **$445$ train / $223$ test** for CD4+) engaging in **$456$ kill bouts** ($269$ single-kill and $187$ multi-kill cooperative bouts), plus **$3,598$ classified 3D T-cell trajectories** across $6$ wells in `classified_tcell_track_data.rds`.
  3. **3D Liquid B-ALL Leukemia + CD19 CAR-T Confocal Co-Culture (`Zenodo 8328441`, `LSM880_example_dataset_CART_ALL_Exp28.ims`)**: **$195$ 3D CAR-T trajectories** ($92$ CD4+ and $103$ CD8+; split **$130$ train / $65$ held-out test**) interacting with **$114,396$ 3D B-ALL target surfaces** ($\sim 240$ contact bouts over $480$ frames / $22.5\ \text{h}$ at $0.91\ \mu\text{m}$ voxel pitch).
  4. **BioImage Archive `S-BIAD448` Raw Multi-Well Confocal Scenes**: **$22$ full 4D `.czi` multi-well datasets** comparing `13T` sensitive vs. `100T` resistant breast cancer PDOs.

---

## Side-by-Side Video 1 — Actual 320-nm Light-Sheet Video vs. 4D Deformable Digital Twin (Single CAR-T Killing a Nalm6 Tumor Cell, $0 \to 180\ \text{min}$)

![Side-by-side Event 1: 320-nm Light-Sheet Video vs 4D Deformable Digital Twin](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/side_by_side_event1_lightsheet_single_kill.gif)

- **Playable H.264 MP4**: [side_by_side_event1_lightsheet_single_kill.mp4](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/videos/side_by_side_event1_lightsheet_single_kill.mp4)
- **Column A (Left — `A1` XY & `A2` XZ)**: Actual 320-nm Bessel light-sheet microscopy video ([MOESM9_ESM.mp4](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/external/feilab_lightsheet_data/MOESM9_ESM.mp4), Wang *et al.* 2026) showing both the **Top (`XY`)** and **Depth (`XZ`)** views of a live CAR-T cell (`green Lifeact-EGFP`) docking onto a Nalm6 tumor cell (`red Membrane-mCherry`), assembling a concave actin cup ($50\text{--}90\ \text{min}$, $CA = 194.5\ \mu\text{m}^2$), inducing $3$ apoptotic bleb lobes ($75\text{--}115\ \text{min}$), collapsing target volume ($105\text{--}130\ \text{min}$), and triggering cyan `SYTOX Blue` nuclear lysis ($150\text{--}180\ \text{min}$) before elongating to $29.0\ \mu\text{m}$ to detach.
- **Column B (Second — `B1` XY & `B2` XZ)**: Equivalent 2D harmonic + active-cortex synthetic fluorescence render ($M = 16$).
- **Column C (Third)**: Frame-synchronized, physically constrained 4D digital twin mesh ($0 \to 180\ \text{min}$, $0.0\ \mu\text{m}^3$ 3D overlap, stiff ovoid nuclei, MTOC docking, and 3D Young–Laplace bleb blisters).
- **Column D (Right — `D1` Trajectories + `D2a / D2b` Dual-Cell Shape Match)**:
  - **`D1` (Top Right)**: Measured vs. Simulated 3D Synapse Contact Area ($\mu\text{m}^2$) and Target Volume ($\mu\text{m}^3$).
  - **`D2a` & `D2b` (Bottom Right)**: Live side-by-side centered cell silhouettes ($\mu\text{m}$) for the **CAR-T Effector (`D2a`, green)** and **Nalm6 Target (`D2b`, red)** separately, comparing the **Real Video Silhouette (filled)** against the **Ellipsoid $M=2$ (dashed orange + shaded radial error band)** and **Constrained 4D Twin $M=16$ (solid contour)**.

![Dual-Cell Shape Match & Boundary Residual Decomposition: CAR-T vs Nalm6 Target](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_12_dual_cell_shape_match.png)

---

## Multi-Organelle 4D Digital Twin (`MitoSpace4D`-Inspired Mitochondrial Networks, Microtubule Aster & Granule Traffic)

![Multi-Organelle 4D Digital Twin: Mitochondrial Fission/Fusion Networks, Microtubule Aster & Granule Traffic](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_13_multiorganelle_mitospace_4d.png)

- **How All Internal Organelles Deform Seamlessly Inside Our $M = 16$ Cells**:
  - Every organelle node and filament is defined in **SDF-Normalized Cytoplasmic Coordinates** $\rho(\mathbf{x}, t) = \Phi_{\text{nuc}}(\mathbf{x}, t) / (\Phi_{\text{nuc}}(\mathbf{x}, t) + |\Phi_{\text{cortex}}(\mathbf{x}, t)|) \in [0.14, 0.86]$ between the stiff ovoid nuclear envelope ($\rho = 0$) and the Helfrich-faired cortical membrane ($\rho = 1$), guaranteeing **$0.0\ \mu\text{m}^3$ nuclear or cortical membrane penetration** across all $73$ frames (`0 to 180 min`).
- **Four Subcellular Systems Added (`Row B` & `Panel C1`)**:
  1. **CAR-T 3D Mitochondrial Tubular Network (`gold/amber` tubules, $\mathcal{G}_{\text{mito}}^{\text{CAR-T}}$)**: Packs into the **trailing uropod** during hand-mirror search ($0\text{--}45\ \text{min}$, powering Myosin-IIA motility), translocates around the nucleus to form a **perisynaptic mitochondrial cradle** ($55\text{--}105\ \text{min}$, buffering ORAI1 $\text{Ca}^{2+}$ influx and generating a $1.42\times$ local ATP surge for $F_{\text{syn}}$), and repolarizes to the uropod stalk during exit ($155\text{--}180\ \text{min}$).
  2. **Target 3D Mitochondrial Reticulum & Granzyme-B/tBid Drp1 Fission Wave (`teal/cyan` tubules $\to$ `purple` MOMP puncta, $\mathcal{G}_{\text{mito}}^{\text{Target}}$)**: Starts as a fused perinuclear reticulum ($L_{\text{mito}} = 4.12\ \mu\text{m}$, $\Delta\Psi_m = 0.95$). At $65\text{--}90\ \text{min}$ ($t_{1/2} = 76\ \text{min}$), Granzyme B $\to$ tBid/BAX MOMP and Drp1 fission sever the network along a synaptic-to-distal wave into **$28$ depolarized puncta** ($L_{\text{mito}} = 0.82\ \mu\text{m}$, $\Delta\Psi_m = 0.12$) *20 minutes before* Caspase-3/ROCK1 cortical blebbing (`96 min`, `Panel D2`).
  3. **3D Microtubule Cytoskeletal Aster (`indigo/blue` filaments, calibrated to real `SiR-tubulin` light-sheet movies in `Row A`)**: $14$ geodesic filaments radiating from the **MTOC (`centrosome`)** around the ovoid nucleus to cortical dynein anchors on the synaptic cup, pulling the MTOC to within $0.68\ \mu\text{m}$ of the synapse (`MPA = 11.8°`).
  4. **Directed Lytic Granule Vesicle Traffic (`magenta` spheres)**: $18$ discrete `PRF1/GZMB` vesicles ratcheting along the 3D microtubules via minus-end dynein motors ($R_{\text{coal}}: 2.85 \to 0.92\ \mu\text{m}$) to exocytose through the central F-actin clearing ($\text{ADC} = 0.44$).
- **Metabolic Control of Serial Killing (`Panels C2 & D3`)**:
  - Coupling T-cell mitochondrial fusion connectivity ($L_{\text{mito}}/L_0$) and $\Delta\Psi_m$ ATP supply to $F_{\text{syn}}$ and $F_{\text{pull}}$ mechanistically explains our serial-killer parameter $\sigma = 0.88$: `Opa1+/CRTAM+ Cluster 9 Super-Engagers` with fused synaptic mitochondria achieve **$3.12\ \text{kills/12 h}$**, whereas `Drp1` hyper-fission / metabolically exhausted T cells stall at **$0.74\ \text{kills/12 h}$**.

---

## Side-by-Side Video 2 — Actual 4D Confocal Organoid Video (`.ims`) vs. Multi-T-Cell Cooperative Digital Twin ($0 \to 510\ \text{min}$)

![Side-by-side Event 2: 4D Confocal Organoid Video vs Multi-T-Cell Cooperative Digital Twin](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/side_by_side_event2_organoid_swarm_kill.gif)

- **Playable H.264 MP4**: [side_by_side_event2_organoid_swarm_kill.mp4](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/videos/side_by_side_event2_organoid_swarm_kill.mp4)
- **Column A (Left)**: Centered 2D confocal microscopy video (`XY` + `XZ` projections) extracted from [LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims) (`TimePoint 0..255`, centered on Organoid `1000826950`), showing **green CD8+ TEGs** and **cyan CD4+ TEGs** swarming onto a live multi-lobular breast cancer organoid (`amber`) and turning it **bright red (`Ch 2` dead-cell dye)** as it breaks into $18$ apoptotic fragments.
- **Column B (Second)**: Multi-acinar harmonic 2D synthetic confocal render (`XY` + `XZ` projections, $M = 16$ contour match $\text{IoU} = 0.864$ vs. $0.715$ for an $M = 2$ ellipsoid).
- **Column C (Third)**: Axial-PSF-deconvolved, physically constrained 4D Dual-Surface-Net mesh reconstruction of the 3D multi-lobular organoid and swarming CD8+/CD4+ T cells ($0.0\ \mu\text{m}^3$ 3D interpenetration, regional viable $\to$ apoptotic Phong shading, and `gold` synaptic contact rings).
- **Column D (Right)**: Synchronized Measured (points) vs. Simulated (lines) trajectories for 3D Organoid Volume ($10^3\ \mu\text{m}^3$), Dead-Dye Intensity, and Multi-Lobular Contour $\text{IoU}$.

![Event 2 Upgraded: Multi-Lobular 3D Organoid & T-Cell Swarm Reconstruction](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_10_video2_organoid_4d_upgrade.png)

---

## Inverted 4D Force & Pressure Atlas from Video 1 + Four Counterfactual Mechanobiology Hypothesis Tests

![Inverted 4D Force & Pressure Atlas from Video 1 + Four Mechanobiology Hypothesis Tests](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_11_force_extraction_and_hypothesis_tests.png)

- **Row A — Five Biophysical Force & Pressure Trajectories Inverted from Video 1 ([video1_extracted_forces_and_pressures.csv](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/data/video1_extracted_forces_and_pressures.csv))**:
  1. **Normal Synaptic Protrusion Force $F_{\text{syn}}(t)$ & Exit Pull Force $F_{\text{pull}}(t)$ (`Panel A1`)**: Inverted from the measured 3D contact area $CA(t)$ and concave indentation depth $\delta_{\text{indent}}(t)$ via Hertzian–Boussinesq + cortical shell elasticity (`ShElastic` / Vorselen *et al.* 2020), $F_{\text{syn}}(t)$ peaks at **$1.28\text{--}1.42\ \text{nN}$** ($6.8\ \text{Pa}$ normal traction stress) during $65\text{--}95\ \text{min}$, followed by a **$1.42\ \text{nN}$** uropod retraction pull during post-kill detachment ($155\text{--}180\ \text{min}$).
  2. **Synaptic Actin Molecular Clutch & Secretory Clearing (`Panel A2`)**: Retrograde F-actin flow $v_{\text{actin}}(t)$ peaks at **$89.4\ \text{nm/s}$** while central F-actin depletes ($\text{ADC}: 0.92 \to 0.44$), opening the central secretory window for lytic granule exocytosis.
  3. **Target Cortical Tension $\sigma_{\text{mem}}(t)$ & Basu Mechano-Pore Boost (`Panel A3`)**: Synaptic indentation stretches the target cortex from **$0.22 \to 1.01\ \text{mN/m}$**, multiplying the perforin pore insertion rate by **$5.2\text{--}6.0\times$** (Basu *et al.*, *Cell* 2016) to outpace ESCRT-III repair.
  4. **Young–Laplace Apoptotic Bleb Hydrostatic Pressure $\Delta P_{\text{bleb}}(t)$ (`Panel A4`)**: Inverted from the 3D bleb blister curvature $R_{\text{bleb}}(t)$ via $\Delta P_{\text{bleb}} = 2\gamma_{\text{mem}}/R_{\text{bleb}}$, intracellular hydrostatic pressure surges from **$35\ \text{Pa}$** to **$206\text{--}220\ \text{Pa}$** ($75\text{--}110\ \text{min}$), exceeding the **$160\ \text{Pa}$** actin–ezrin cortical delamination threshold.
- **Row B — Four Counterfactual Hypothesis Tests ([mechanobiology_counterfactual_experiments.csv](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/data/mechanobiology_counterfactual_experiments.csv))**:
  - **Test 1 (`Panel B1`, Mechanosensitive Pore Gating vs. Chemical-Only Release)**: Holding granule cargo constant while reducing $F_{\text{syn}}$ ($1.42 \to 0.55 \to 0.08\ \text{nN}$, mimicking $0, 25, 50\ \text{nM}$ Dasatinib) reproduces the observed collapse in 3-hour killing across $n = 419$ light-sheet CAR-T pairs (**$57.5\% \to 27.0\% \to 4.5\%$**), whereas a chemical-only null model fails ($42\%\text{--}49\%$ killing).
  - **Test 2 (`Panel B2`, Solid Tumor Desmoplasia / Stiffness Barrier)**: Applying the Video 1 single-CTL force ($1.42\ \text{nN}$) across target stiffness $E_{\text{target}} = 300 \to 2,400\ \text{Pa}$ shows why single T cells clear soft B-ALL leukemia blasts ($450\ \text{Pa}$, **$84\%$ kill**) but fail alone against stiff solid breast cancer organoids ($1,500\ \text{Pa}$, **$18\%$ kill**), requiring $\ge 3\text{--}4$ co-engaged T cells ($\sum F_{\text{syn}} = 3.85\ \text{nN}$, **$84.5\%$ kill**).
  - **Test 3 (`Panel B3`, ROCK1 / Myosin-II Bleb Pressure Ablation & "Corpse-Trapping")**: Blocking Caspase-3 $\to$ ROCK1 hydrostatic pressure ($\Delta P_{\text{bleb}}$ capped at $45\ \text{Pa} < 160\ \text{Pa}$) prevents target fragmentation ($1$ swollen corpse vs. $5.2$ apoptotic bodies) and delays T-cell detachment by **$+38\ \text{min}$** ($42 \to 80\ \text{min}$).
  - **Test 4 (`Panel B4`, Engineering "Super-Engager" Serial Killers)**: Boosting migration speed alone increases 12-hour serial killing by only **$+14\%$** ($1.18 \to 1.35$ kills/CTL), whereas co-engineering **CRTAM-stabilized synaptic protrusion force ($F_{\text{syn}}$)** + **post-kill uropod deadhesion ($F_{\text{pull}}$)** increases serial killing by **$2.34\times$** ($1.18 \to 2.76$ targets/CTL).

---

## Physically Constrained 4D Reconstruction: Resolving 3D Interpenetration, Polar Creases & Volume Drift

![Physically constrained 4D reconstruction vs unconstrained 3D extrusion](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_9_physical_4d_constraints.png)

- **Why unconstrained 2D-to-3D extrusions (`Row B`) look physically implausible**:
  1. **3D Mesh Interpenetration**: Extruding each cell's 2D contour independently causes the CAR-T and Target meshes to pass through each other at the immunological synapse (up to $59.3\ \mu\text{m}^3$ overlapping volume in `Panel D2`).
  2. **Polar Coordinate Singularity ("Bundt-Cake" Creases)**: Parameterizing a curved hand-mirror cell on a polar $(\theta, \psi)$ grid pinches all meridians to a single top/bottom pole, creating artificial radial creases.
  3. **Shark-Fin Depth Distortion & Volume Drift**: Extruding narrow uropods or bleb necks without coupling to their local cross-sectional width produces vertical fins and causes $19.7\%$ cytoplasmic volume drift (`Panel D3`).
  4. **Missing Intracellular Nucleus Core**: Without a stiff internal nucleus ($E_{\text{nuc}} \approx 4\,E_{\text{cortex}}$), the membrane can pinch through the cell center.
- **How our 6-constraint 3D Signed Distance Field + Dual Surface Net solver (`Row C` & `Panel D1`) fixes all 4 defects**:
  - Enforces **joint `XY` $\times$ `XZ` tomography with tubular $Z$-scaling**, **strict 3D steric exclusion ($\Omega_{\text{CAR-T}} \cap \Omega_{\text{Target}} = \emptyset$, $0.0\ \mu\text{m}^3$ overlap)** across a flush concave synaptic cup, **osmotic volume conservation ($|\Delta V|/V_0 \le 0.01\%$)**, **14-step Taubin Helfrich bending regularization**, **Young–Laplace 3D spherical-cap blebs**, and **ovoid 3D intracellular nuclei** (`Panel D1`).

---

## How We Extract & Match Sub-Micron Cell Shapes: From Ellipsoids ($M = 2$) to Harmonic Active-Cortex Meshes ($M = 16$)

![High-order shape extraction and matching across 5 stages](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_8_shape_extraction_and_matching.png)

- **Tier 1 — Parametric Ellipsoid ($M = 2$, `Row B`, Mean $\text{IoU} = 0.828$, Boundary $\text{RMSE} = 0.725\ \mu\text{m}$)**:
  - Fits only centroid $(x, y, z)$ and $3$ principal semi-axes $(L_a, L_b, L_c)$. Fails to capture the **constricted neck and trailing uropod stalk** of the migrating CAR-T cell at $15\text{--}55\ \text{min}$ ($\text{IoU} = 0.73\text{--}0.77$) and misses the **pinched-neck apoptotic bleb blisters** on the target membrane at $85\text{--}100\ \text{min}$ ($\text{IoU} = 0.73\text{--}0.76$).
- **Tier 2 — Active-Cortex Hand-Mirror + Pinched-Bleb Mechanics ($M = 6$, Mean $\text{IoU} = 0.910$, Boundary $\text{RMSE} = 0.305\ \mu\text{m}$)**:
  - Adds odd polar/spherical-harmonic modes $m = 1, 3$ to model **front-to-back T-cell polarity** (flared Arp2/3 anterior lamellipodial apron + Myosin-IIA trailing uropod stalk) and modes $m = 3 \dots 6$ to model **concave synaptic indentation** and **Caspase-3/ROCK1-driven cortical delamination** into $3$ pinched-neck bleb blisters.
- **Tier 3 — Coupled `XY`+`XZ` Harmonic Visual-Hull + Multi-Scale Cortical Actin Puncta ($M = 16$, `Row C`, Mean $\text{IoU} = 0.951$, Boundary $\text{RMSE} = 0.085\ \mu\text{m} < 0.32\ \mu\text{m}$ PSF)**:
  - Assimilates polar Fourier / Spherical-Harmonic modes $m = 0 \dots 16$ from the orthogonal `XY` and `XZ` light-sheet views (`Row A` contours) and modulates the equatorial cortical shell with a **3-band multi-scale Arp2/3 actin micro-cluster texture** ($\sigma = 0.26, 0.61, 1.22\ \mu\text{m}$), reducing boundary error to **$0.07\text{--}0.12\ \mu\text{m}$** (below the $0.32\ \mu\text{m}$ optical point-spread function).

---

## How the Upgraded 2D Simulation Render Compares to the Real 2D Video and True 4D Mesh

![How the 2D simulation render compares to the real 2D video and 4D mesh](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_7_2d_vs_4d_render_comparison.png)

1. **What the 2D Synthetic Fluorescence Render (`Row B`) Captures from the 4D Twin**:
   - **Equatorial cortical shells & synaptic actin ring (`55–85 min`)**: Slicing the 3D deformable meshes through the light-sheet optical slab ($\text{PSF }\sigma_{xy} = 0.32\ \mu\text{m}$) reproduces the hollow cortical F-actin rim (`green`), the **$3.2\times$ synaptic F-actin enrichment** at the contact interface (`yellow-green`), and the **concave indentation** of the `red` target membrane.
   - **Apoptotic blebbing & nuclear permeabilization (`100–170 min`)**: Reproduces the $3$-lobed target membrane blebbing at $100\ \text{min}$ and the **interior cyan `SYTOX Blue` nuclear flash** inside the red membrane rim at $170\ \text{min}$ as the CAR-T elongates ($29\ \mu\text{m}$) to detach.
2. **How the 2D Simulation Render Differs from the Real 2D Video (`Row B` vs. `Row A`)**:
   - **Continuum cortex vs. sub-micron filopodial speckles**: The real CAR-T `Lifeact-EGFP` signal contains stochastic sub-micron actin micro-clusters, filopodial spikes, and cytosolic vesicles, whereas the 2D projection of our continuum ellipsoid/harmonic mesh produces a smooth cortical shell.
   - **Symmetric harmonic bleb lobes vs. asymmetric neck pinch-off**: At $100\ \text{min}$, the real target membrane forms irregular, sharply pinched bleb necks (`Row A, Col 4`), whereas the low-order spherical-harmonic lobe mode (`n_lobes = 3`) renders smooth, symmetric lobes (`Row B, Col 4`).
3. **Why the 4D Mesh (`Row C`) Is Essential Beyond Any 2D Image (`Row A/B` vs. `Row C`)**:
   - **Resolving Z-overlap ("pancake") ambiguity**: In 2D `XY` projections (especially in 3D organoids), a T cell crawling across the **top dome** ($Z = +20\ \mu\text{m}$) projects directly onto the center of the organoid in 2D, appearing as deep core infiltration or active synapse engagement even when separated in $Z$. The 4D mesh view (`Row C`) tracks true 3D surface-to-surface distance ($d_{3\text{D}} \le 5\ \mu\text{m}$), 3D synaptic curvature, and the internal MTOC polarity vector.

---

## 1. The Multi-Scale Imaging Bridge (Why One Dataset Alone Cannot Build a 4D Twin)

![Multi-scale imaging comparison](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_1_scale_gap_comparison.png)

- **Panels A & B — Multi-Hour 3D Confocal Co-Cultures (`Zenodo 8328441`, [LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims) & [LSM880_example_dataset_CART_ALL_Exp28.ims](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/LSM880_example_dataset_CART_ALL_Exp28.ims))**:
  - Track hundreds of **CD8+ (green)** and **CD4+ (cyan)** effector T cells hunting **solid breast cancer organoids (amber, Panel A)** or **liquid B-ALL leukemia cells (amber, Panel B)** over $10\text{--}22.5\ \text{h}$, capturing when target cells turn **red (nuclear dead-cell dye)** (**Layers 1 & 4**).
  - **The gap**: At $0.91\text{--}3.54\ \mu\text{m}$ voxel pitch and $2.0\text{--}2.8\ \text{min}$ sampling, confocal co-culture movies cannot resolve the curved immunological synapse cleft, actin retrograde flow, or MTOC granule docking.
- **Panels C, D & E — 320-nm Bessel Light-Sheet Microscopy (`Wang et al. Sci Data 2026`, [MOESM8_ESM.mp4](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/external/feilab_lightsheet_data/MOESM8_ESM.mp4), [MOESM4_ESM.mp4](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/external/feilab_lightsheet_data/MOESM4_ESM.mp4), [MOESM5_ESM.mp4](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/external/feilab_lightsheet_data/MOESM5_ESM.mp4))**:
  - Resolves $n = 419$ CAR-T–Nalm6 pairs at **$320\ \text{nm}$ optical resolution** ($0.196\ \mu\text{m}$ reconstructed voxel pitch, $2.5\ \text{s/volume}$ bursts), revealing the **concave green F-actin cup (`Lifeact-EGFP`, Panel C)** indenting the **red Nalm6 tumor membrane**, the **3D U-Net + rainbow synapse patch + purple MTOC vector reconstruction (Panel D)**, and **blue microtubule MTOC docking (`SiR-tubulin`, Panel E)** (**Layers 2 & 3**).
- **Panels F & G — Perturbation & Transcriptomic Calibration**:
  - **Panel F**: Across $419$ cell pairs, blocking TCR/CAR kinase signaling with **dasatinib ($25\text{--}50\ \text{nM}$)** collapses the 3D synapse contact area from **$166.7\ \mu\text{m}^2 \to 11.4\ \mu\text{m}^2$**, slows retrograde actin flow from **$57.8 \to 16.4\ \text{nm/s}$**, and abolishes MTOC polarization (**$47.3^\circ \to 91.6^\circ$**).
  - **Panel G**: Live-imaging-tracked single-cell RNA-seq ($n = 1,814$ CD8 TEGs) shows **Type I IFN priming (`ISG15`)** upon initial organoid exposure (`Never-eng`), followed by a **$44\times$ surge in synaptic adhesion receptor `CRTAM`**, **$10\times$ `IFNG`**, and **$4.3\times$ `GZMB`** in `Super-engager` serial killers.

---

## 2. Anatomy of a Single Cell Kill at 320-nm Resolution (0 to 180 min Time-Lapse)

![320-nm light-sheet CAR-T kill sequence](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_2_lightsheet_kill_sequence.png)

Extracted directly from orthogonal **Top (`XY`)** and **Side (`XZ`)** 3D projections in [MOESM9_ESM.mp4](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/external/feilab_lightsheet_data/MOESM9_ESM.mp4) ([lightsheet_moesm9_4d_trajectory.csv](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/data/lightsheet_moesm9_4d_trajectory.csv)):
1. **`15 min` (Search & Approach)**: Spherical CAR-T (`green`) and intact Nalm6 target (`red`) are separated by $24.1\ \mu\text{m}$.
2. **`55 min` (Synapse Assembly)**: CAR-T forms a flattened F-actin ring against the target (`CA = 141.4 µm²`).
3. **`85 min` (Peak Actin Cup Indentation)**: Deep concave synaptic cup (`CA = 194.5 µm²`) mechanically stretches the target cortex.
4. **`100 min` (Target Blebbing)**: Granzyme B $\to$ Caspase-3/ROCK1 activation causes the red target membrane to erupt into $3$ distinct bleb lobes.
5. **`120 min` (Volume Collapse)**: Target projected area contracts by **$39\%$** (`CA` drops to $63.4\ \mu\text{m}^2$).
6. **`170 min` (`SYTOX Blue` Nuclear Lysis & CAR-T Exit)**: Target nucleus turns bright **cyan (`SYTOX Blue` influx)** while the CAR-T elongates ($L_c = 25.0\ \mu\text{m}$) to detach.

---

## 3. Macro-Scale 4D Confocal Co-Cultures: Solid Organoids vs Liquid Leukemia

![4D confocal organoid and leukemia co-cultures](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_3_organoid_and_leukemia_co_culture.png)

- **Top row (Solid Breast Cancer PDO, $0.0 \to 10.0\ \text{h}$)**: Maximum-intensity projections from [LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims) zoomed on Organoid 01 (`Track 826950`). As **green CD8 TEGs** accumulate on the organoid perimeter, the tumor core progressively turns **bright red (`NucRed Dead 647`)**, losing **$56\%$ of its viable volume** ($102,553 \to 44,713\ \mu\text{m}^3$) and breaking into **$18$ apoptotic bodies** (bottom-left plot).
- **Middle row (Liquid B-ALL Leukemia + CD19 CAR-T, $0.0 \to 19.7\ \text{h}$)**: Maximum-intensity projections from [LSM880_example_dataset_CART_ALL_Exp28.ims](file:///Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim/LSM880_example_dataset_CART_ALL_Exp28.ims) showing **green CD8 CAR-T** and **cyan CD4 CAR-T** effectors engaging B-ALL blasts across $480$ frames.
- **Bottom-right plot (3D T-Cell Elongation)**: Across all $9$ BEHAV3D clusters, `Cluster 9 Super-Engager` T cells stretch from a resting diameter of **$13.6\ \mu\text{m}$** to a peak major-axis length of **$28.9\ \mu\text{m}$** during active killing.

---

## 4. How the 4-Layer Digital Twin Connects the Imaging Data Together

![Coupled 4-layer digital twin architecture](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_4_four_layer_architecture.png)


---

## 5. Multi-T-Cell Organoid Swarming & Additive Cooperativity Against ESCRT Repair

![Multi-T-cell organoid swarming and cooperativity](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_5_multi_tcell_cooperativity.png)

---

## 6. Serial Killer Experiment: Intrinsic Killer Phenotype ($\sigma > 0$) vs. Target Geometry ($\sigma = 0$)

![Serial killer experiment across 1,171 CD8+ T-cell tracks](/Users/dxt397/.gemini/antigravity/brain/f144e2e4-ff67-494b-b7e7-94fad918619d/visual_6_serial_killer_experiment.png)
