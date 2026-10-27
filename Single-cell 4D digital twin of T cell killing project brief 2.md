# Single-cell 4D digital twin of T cell killing: project brief

Oct 23, 2026 · @David Thomas

## Aim

Build a physics-based model of one cytotoxic T cell killing one tumor cell, in 3D over time, that reproduces what live microscopy shows: approach, contact, deformation of the target, death and detachment. Every parameter is tied to a measured quantity, and the output is compared frame by frame with real 4D imaging. That is what "digital twin" means in this project; it does not mean a whole-patient pharmacology model.

Published models of T cell killing stop at the population level, where a kill is a probability on contact. A single-cell mechanical model would let us test which parts of the mechanism (synapse force, target membrane tension, perforin delivery, serial killing) separate effective T cells from ineffective ones. It would also give a template for evaluating lymphocyte products such as the organoid-interacting lymphocytes being developed at Wake Forest.

Starting point: the open BEHAV3D data (Princess Máxima Center, Utrecht) to fit the behavior and timing layer now. The mechanics layer will need higher-resolution imaging that does not exist in public archives; the first tasks below are scoped to what can be done with the open data.

## Origin

A presentation on 2026-10-23 from the [Wake Forest Organoid Research Center (WFORCE)](https://school.wakehealth.edu/research/institutes-and-centers/wake-forest-institute-for-regenerative-medicine/research/wforce) showed microscopy of immune cells attacking and pulling apart tumor cells. WFORCE is the Votanopoulos and Soker program that builds immune-enhanced patient tumor organoids: tumor cells plus matched lymph node or blood cells in a hydrogel.

The footage most likely comes from their February 2026 preprint, [Enhancing Patient Lymphocyte Response to Peritoneal Malignancies Using a Personalized Immunocompetent Microfluidic Co-Culture Platform](https://www.biorxiv.org/content/10.64898/2026.02.19.706377v1.full.pdf). Patient blood cells were circulated for seven days through a chip holding the patient's tumor cells and antigen-presenting cells; the trained product is called organoid-interacting lymphocytes (OILs). A patient-community summary of the preprint ([Appendicure](https://appendicure.com/2026/02/23/a-new-immunotherapy-strategy-emerging-from-wake-forest-for-appendiceal-cancer-tumor-on-a-chip/)) reports mean tumor cell death of 52.2% for OILs, 35.9% for expanded blood cells and 24.9% for TILs; verify these against the preprint before quoting them. The OIL concept was first published in [Annals of Surgical Oncology, 2024](https://link.springer.com/article/10.1245/s10434-023-14858-6). The microscopy itself is not public.

The question the footage raised: does anyone simulate this motion and mechanism at single-cell scale? A literature search on 2026-10-23 found no integrated model. The pieces exist separately (sections below).

## What has been measured

The mechanism in the footage is mechanical as well as chemical, and the key quantities have been measured.

- **Synapse force potentiates perforin.** Cytotoxic T cells exert force across the immunological synapse; the magnitude of force correlates with the speed of perforin pore formation on the target, and raising target membrane tension increases pore formation and killing. Individual pillar deflections required more than 340 pN, about 170 × 10⁻¹⁸ J within about 1 µm². Force exertion and perforin release are coordinated in space and time ([Basu et al., Cell 2016](<https://www.cell.com/cell/fulltext/S0092-8674(16)00061-1>)).
- **Actin protrusions do the pushing.** Perforin is released at the base of actin-rich protrusions that depend on WASP and Arp2/3; the protrusions are required for force exertion and efficient killing, and they physically deform the target cell surface ([Tamzalit et al., Science Immunology 2019](https://janelia.org/publication/interfacial-actin-protrusions-mechanically-enhance-killing-by-cytotoxic-t-cells); [preprint](https://www.biorxiv.org/content/10.1101/443309v1)).
- **Forces can be mapped on a cell-sized target.** Deformable hydrogel microparticles with cell-like stiffness report subcellular force patterns in the cytotoxic synapse; particle shape is resolved to better than 50 nm and tractions are inferred without a reference state ([Vorselen et al., Nature Communications 2020](https://www.nature.com/articles/s41467-019-13804-z); particle synthesis protocol, [Nature Protocols 2025](https://doi.org/10.1038/s41596-025-01281-2)).
- **Centrosome and granules move to the contact.** Electron tomography of a CTL–target conjugate shows the MTOC polarized to the contact site and lytic granules on microtubules leading to the synaptic cleft ([Cell Image Library CIL:40011](https://www.cellimagelibrary.org/images/40011), OME-TIFF download).
- **Killing often needs more than one contact.** In vivo two-photon imaging showed that one CTL contact typically did not suffice to kill, and that targets have a higher probability of dying after multiple contacts ([PLoS Computational Biology 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008428)).
- **Behavior in organoids.** Effective killers ("super engagers") stayed in contact with organoids 48 ± 8 min per hour; single CD8⁺ engineered T cells killed up to 18 cells in 11 h from a fixed anchor point, sending protrusions between epithelial cells and extending their length up to fivefold ([Dekkers et al., Nature Biotechnology 2022](https://www.nature.com/articles/s41587-022-01397-w)).

None of these papers publishes a kinetic relation of perforin pore rate versus membrane tension as a dataset; the numbers are in figures and would have to be extracted.

## What has been modelled

Existing work falls into three groups that have not been joined.

**Population and behavior models (a kill is a rule on contact)**

- ABMACT: agent-based model of NK cell adoptive therapy with heterogeneous virtual cells; cytotoxicity is a probability on contact, with serial-killing capacity as a parameter ([bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.02.17.638701v1.full)).
- 3D agent-based model of CTL killing built to discriminate hypotheses (constant kill probability versus signal integration in CTL or target) against two-photon in vivo data ([PLoS Computational Biology 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008428)).
- Stochastic simulations of single CTLs killing small target populations; shows multiple-hitting produces apparent heterogeneity in killing kinetics ([PLoS Computational Biology 2020](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007972)).
- Multiscale agent-based model in Vivarium with TCR–MHC-I activation, PD-1/PD-L1 inhibition, IFNγ and "cytotoxic packet" secretion, calibrated to multiplexed imaging ([Cell Systems 2024](<https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00089-9>)).
- 3D stochastic agent-based model of T cell and cancer cell growth, migration and interaction in MATLAB ([bioRxiv 2025](https://www.biorxiv.org/content/10.1101/2025.10.21.683801v1.full)).
- PhysiCell 1.14 ships a built-in "attack" model: a persistent synapse with spring adhesion, an attack\_duration parameter and a damage rate; death from accumulated damage is left to the user ([PhysiCell](https://github.com/MathCancer/PhysiCell/releases)).

These reproduce contact statistics and kill kinetics. None represents cell shape, force or the target's membrane.

**Subcellular mechanistic models of one step**

- MTOC repolarization: microtubules as bead-rod chains under constrained Langevin dynamics, pulled by dynein through cortical sliding and capture-shrinkage; cell as a 5 µm sphere with a 3.8 µm nucleus; handles one or two synapses ([Hornak and Rieger, Biophysical Journal 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9034251/); [part I preprint](https://www.biorxiv.org/content/10.1101/822171.full.pdf)).
- Deterministic mechanical model of T-killer polarization with cortical pulling force densities of 20–80 pN/µm, reproducing centrosome oscillation between two targets ([PLoS Computational Biology 2009](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1000260)).
- Agent-based model of immunological synapse formation and TCR cooperativity, with F-actin foci feedback on binding rate ([PMC7555007](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7555007/)).
- Phase-field model of apoptotic cell death: a cytotoxic phase field degrades a cell phase field and reproduces the topological changes of apoptosis; code on [GitHub](https://github.com/Nonlinear-Phenomena-Lab/Phase-field-Apoptosis) ([arXiv 2025](https://arxiv.org/html/2507.09038v3)).

**Data-driven methods (no physics)**

- BEHAV3D: 3D live imaging of engineered T cells in patient-derived organoids, tracking more than 150,000 T cells and classifying nine behaviors; the "super engager" cluster has serial-killing capacity ([Nature Biotechnology 2022](https://www.nature.com/articles/s41587-022-01397-w); protocol in [Nature Protocols 2024](https://www.nature.com/articles/s41596-024-00972-6)).
- TIMING: one effector and one target per nanowell, imaged every 5 min, with automated segmentation and tracking; software open, including a 2026 label-free version ([Single-Cell Lab, University of Houston](https://singlecell.chee.uh.edu/publications/)).
- Video diffusion models that forecast apoptosis from TIMING movies by predicting future frames from past frames ([bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.11.16.567461.full.pdf)) and video diffusion fine-tuned on time-lapse microscopy ([arXiv 2025](https://arxiv.org/abs/2503.18583v2)).

"Digital twin" in the immunotherapy literature currently means a patient-level pharmacology model, for example for [mosunetuzumab](https://pmc.ncbi.nlm.nih.gov/articles/PMC10339700) or [TCR-engineered T cell persistence](https://www.nature.com/articles/s41540-024-00335-7). Those are ordinary differential equations over cell counts, not cells.

## The gap

No published model couples the following in one simulation and validates it frame by frame against 4D microscopy:

1. Two deformable cells, each with membrane, cortex and nucleus.
2. Adhesion and a persistent synapse between them.
3. Actin protrusions that push into the target at the measured force scale (hundreds of pN per protrusion, nN in total).
4. Perforin pore formation whose rate depends on target membrane tension.
5. Granzyme entry, caspase activation and the resulting apoptotic morphology: blebbing and fragmentation.
6. Detachment and movement to the next target (serial killing).

The "pulling apart" seen in the footage is two things at once: the T cell tugging on the target and the target breaking into apoptotic bodies. A model has to produce both from the same mechanics and chemistry.

Three data gaps go with the modelling gap. No public dataset pairs force measurements with time-lapse of the same kill. No one has published perforin kinetics versus membrane tension as data. No subcellular-resolution 4D imaging of T cell–tumor cell conjugates was found in the Image Data Resource or the BioImage Archive on 2026-10-23.

## Proposed model

Four layers, built in order, each tied to data that exists or that we can acquire. The validation loop at the end applies to every layer.

1. **Behavior and timing.** Stochastic rules for search, contact, time to kill, serial killing and detachment. Parameters are fitted to BEHAV3D tracks now, and to TIMING or our own nanowell-style data later. Baseline implementation: PhysiCell's attack model or a short custom simulation. This layer is the reference the physical layers must reproduce.
2. **Mechanics.** A deformable cell model: triangulated surface meshes for the T cell and the target with membrane and cortex tension, a volume constraint and a nucleus; an adhesion spring at the synapse; active protrusions that push with the force scale from Basu et al.; a target whose tension rises under load. Candidate framework: [SimuCell3D](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11052725/), an open-source 3D deformable cell model with subcellular resolution and automatic mesh remodelling. Whether it supports active, localized protrusion forces must be checked; a custom mesh model is the fallback. The surfaces it produces are the same kind of object as the meshes we already track from camera data, so the comparison tooling carries over.
3. **Intracellular machinery.** TCR triggering, MTOC repolarization and granule transport to the contact, with release at protrusion bases. Starting point: the Hornak and Rieger model; Cytosim (open-source cytoskeleton simulator) can run filament and motor dynamics of this kind.
4. **Chemistry and death.** Perforin pore rate as a function of membrane tension, granzyme entry, caspase activation, then blebbing and fragmentation. Starting point: the phase-field apoptosis model, coupled to the mechanics layer so that fragmentation changes the forces.

**Validation loop.** Segment real 4D movies into surfaces, track them, and compare against the simulation on: contact area over time, target deformation field, contact duration, time from contact to death, number of kills per T cell, T cell length change. Agreement on these, not on appearance, is the acceptance test. For BEHAV3D data only the last four are measurable; the first two need the higher-resolution imaging listed under open questions.

## Open data

BEHAV3D is the only open 4D dataset of T cells killing tumor cells found in this search. Everything else is either a sample, a method, or data that has to be requested.

| Source | What it holds | Format and size | Access | Link |
| --- | --- | --- | --- | --- |
| BEHAV3D example datasets (Zenodo, mirrored on Dryad) | Two pre-processed co-culture movies with Imaris segmentation and tracks: TEGs with breast cancer organoids (2019-06-26) and CD19 CAR T cells with ALL cells (Exp28) | Imaris .ims (HDF5), 15.1 GB and 24.3 GB, plus README; MD5 listed | CC0 | [Zenodo 8328441](https://zenodo.org/records/8328441), [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.3n5tb2rp6) |
| BEHAV3D raw reference-map data | Raw imaging behind the nine-behavior map: 22 datasets of TEGs and control LM1 cells with 13T or 100T organoids | Not opened in this search; file list and format to confirm | Public archive | [BioImage Archive S-BIAD448](https://www.ebi.ac.uk/biostudies/studies/s-biad448) |
| BEHAV3D code | R scripts for death dynamics, behavior classification and back-projection; demo track tables; newer pipeline with pretrained Cellpose models for T cells and organoids and btrack tracking | GitHub | Open | [AlievaRios/BEHAV3D](https://github.com/AlievaRios/BEHAV3D), [imAIgene-Dream3D/BEHAV3D](https://github.com/imAIgene-Dream3D/BEHAV3D), [BEHAV3D-Explorer](https://github.com/imAIgene-Dream3D/BEHAV3D-Explorer) |
| BEHAV3D videos | Raw plus segmented movies, including one TEG destroying an organoid over 10 h and one CAR T cell killing five cells in 8 h | MP4 | Open | [Nature Protocols supplementary](https://www.nature.com/articles/s41596-024-00972-6), [Nature Biotechnology video](https://pmc.ncbi.nlm.nih.gov/articles/PMC9849137/) |
| BEHAV3D transcriptomics | scRNA-seq of T cells sorted by behavior, including the serial-killer signature | GEO | Open | [GSE172325](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE172325) |
| CTL–target synapse tomogram | Electron tomography of one conjugate, MTOC polarized, granules on microtubules | OME-TIFF, 0.8 MB | Open | [CIL:40011](https://www.cellimagelibrary.org/images/40011) |
| TIMING nanowell data | One effector and one target per well, 5-min interval; full datasets 200 GB–1.5 TB | Sample block only as supplementary material; rest by request | Software open | [Houston Single-Cell Lab](https://singlecell.chee.uh.edu/publications/) |
| Synapse force data | Pillar-array forces, perforin kinetics versus tension | Source data within the papers; no standalone dataset | Open papers | [Basu 2016](<https://www.cell.com/cell/fulltext/S0092-8674(16)00061-1>), [Tamzalit 2019](https://www.biorxiv.org/content/10.1101/443309v1) |
| Microparticle TFM | Method and analysis for force maps on hydrogel targets | Protocol and code with paper | Open | [Vorselen 2020](https://www.nature.com/articles/s41467-019-13804-z) |
| Wake Forest OIL footage | The microscopy that prompted this project | Not public | Request from Votanopoulos or Soker | [WFORCE](https://school.wakehealth.edu/research/institutes-and-centers/wake-forest-institute-for-regenerative-medicine/research/wforce) |

**BEHAV3D acquisition (from the Nature Biotechnology methods)**

- 20,000 engineered T cells per well with organoids at an effector-to-tumor ratio of 1:30 (1:25 for some serial-killing experiments), in 96-well glass-bottom plates with 2.5% BME and pamidronate.
- Labels: T cells with eFluor 450; a second T cell subset with calcein AM when CD4 and CD8 were imaged together; NucRed Dead 647 plus TO-PRO-3 so that dead cells appear under 633 nm excitation and living cells under 561 nm. Organoids are not genetically labelled; the organoid channel is derived by channel arithmetic.
- Zeiss LSM880, Plan-Apochromat 20×/0.8 dry objective, 60 µm Z-stack at the microscope's optimal step, up to 24 h. Either 30-min intervals at 512 × 512 for up to 60 conditions, or 2-min intervals at 512 × 512 or 200 × 200 for 4 to 10 conditions. Pixel pitch is not stated; read it from the file metadata.
- Processing in Imaris 9.2 to 9.5: Channel Arithmetics, Surfaces, ImarisTrack (autoregressive motion for T cells, connected components for organoids), Distance Transformation for T cell–organoid distance.

**Exported per object**

- T cell: coordinates, speed, mean square displacement, distance to organoids, dead-dye intensity (and NCAM1 intensity where labelled).
- Organoid: coordinates, surface area, volume, mean dead-dye intensity.
- Behavior features: T cell death, organoid contact, speed, square displacement, T cell–T cell interaction; tracks cut to 3.3 h; nine clusters by dynamic time warping and k-means; random forest classifier for new experiments.

**Limits for this project**

- 2-min frames at best, roughly 1 µm pixels, a dry 20× lens: positions, contacts and death times are measurable; contact area, protrusion forces and membrane deformation are not.
- Death is scored by nuclear dye uptake, which marks membrane permeabilization, not caspase activation.
- Cells are engineered T cells (TEGs, CAR T, WT1 TCR) against breast cancer organoids, ALL cells, head and neck cancer and diffuse midline glioma. No patient TILs, no OILs, no peritoneal tumors.

Harvested datasets, including a 2026 light-sheet deposit of CAR-T–tumor cell pairs at synapse resolution: Data registry

## First tasks

In order. Steps 1 to 5 use only the open data; step 6 is the first model; steps 7 and 8 start the physics.

1. **Get the breast cancer file.** Download LSM880\_example\_dataset\_TEG\_BC\_organoids\_2019\_06\_26.ims (15.1 GB) from Zenodo 8328441 and check the MD5 (e24b03a9aed6fbc836db9d135fedc9dc). Clone the three BEHAV3D GitHub repositories and download the Cellpose models from the Zenodo link in the Explorer README.
2. **Read the container without Imaris.** Open the .ims as HDF5 with h5py. List the DataSet, ResolutionLevel, TimePoint and Channel groups; read voxel size, time stamps, channel names and laser lines from DataSetInfo. Record pixel pitch, Z step, number of time points and interval, because the paper does not state them. Export one time point per channel to OME-TIFF or OME-Zarr and view it in napari.
3. **Recover the segmentation.** Check whether the Imaris Surfaces and Tracks are readable from the HDF5 Scene groups. If not, re-segment with the Explorer pipeline: Cellpose models for T cells and organoids, btrack for tracking. Report tracking accuracy against the metrics table in the Nature Protocols supplement.
4. **Reproduce the published features.** Compute per-T-cell speed, mean square displacement, distance to nearest organoid surface and dead-dye intensity; per-organoid area, volume and dead-dye intensity. Run the BEHAV3D R classification on the demo tables, then on our tracks, and confirm the nine clusters appear.
5. **Extract event-level data for calibration.** For every T cell track: contact start and end times, contact durations, time from first contact to target death (dye threshold crossing), number of distinct kills, detachment times, and length change during engagement. Produce distributions and save them as CSV with a data dictionary. These are the calibration targets for the behavior layer.
6. **Build the behavior baseline.** Implement the behavior layer in PhysiCell (attack model) or as a short custom stochastic simulation in Python. Fit it to the step 5 distributions. Deliverable: a simulation whose contact-duration, time-to-kill and kills-per-cell distributions match the data, with the fitted parameters and their uncertainty.
7. **Prototype two deformable cells.** Install SimuCell3D and run a two-cell example: one T cell and one target in contact with an adhesion spring. Add a localized pushing force of a few hundred pN at the contact and record target deformation and contact area. Report whether SimuCell3D can represent active protrusions or whether a custom mesh model is needed.
8. **Define the comparison tooling.** Write the code that turns a segmented 4D movie and a simulation into the same table of metrics (contact area, deformation, contact duration, time to death, kills per cell, length change). Test it on the BEHAV3D movie and on the step 7 prototype, even though the first two metrics will be empty for BEHAV3D.

Working conventions: Python 3.11, one repository, environment pinned, every derived table with a data dictionary, raw data never modified, every number in a report traceable to a script and a file.

## Open questions

- [ ] Pixel pitch, Z step and frame interval of the Zenodo breast cancer movie: read from file metadata (step 2).
- [ ] Contents of BioImage Archive S-BIAD448: raw .czi or converted files, how many movies, which organoid lines and T cell types.
- [ ] Whether Imaris Surfaces and Tracks inside the .ims file can be read outside Imaris, or whether re-segmentation is required.
- [ ] Whether SimuCell3D supports localized active forces on a cell surface; if not, which open deformable cell model does.
- [ ] Perforin pore rate versus membrane tension: extract values from the figures of Basu et al. 2016 and record them with their uncertainty.
- [ ] Access to the Wake Forest OIL microscopy and, if possible, raw image stacks, through Votanopoulos or Soker.
- [ ] Access to full TIMING datasets through the Houston group.
- [ ] Specification for our own imaging for the mechanics layer: single T cell and single target, high-NA objective, frame interval of seconds, a membrane label on the target, and a force readout (microparticle TFM or a hydrogel target). Decide what we can acquire locally and what needs a collaborator.
- [ ] Which T cell product and tumor type the first physical model should represent, given that the open data are engineered T cells on breast cancer organoids.

## References

Pages opened on 2026-10-23.

**Origin**

- Wake Forest Organoid Research Center. [WFORCE page](https://school.wakehealth.edu/research/institutes-and-centers/wake-forest-institute-for-regenerative-medicine/research/wforce).
- Enhancing Patient Lymphocyte Response to Peritoneal Malignancies Using a Personalized Immunocompetent Microfluidic Co-Culture Platform. bioRxiv, February 2026. [10.64898/2026.02.19.706377](https://www.biorxiv.org/content/10.64898/2026.02.19.706377v1.full.pdf).
- Organoid-interacting lymphocytes. Annals of Surgical Oncology, 2024. [10.1245/s10434-023-14858-6](https://link.springer.com/article/10.1245/s10434-023-14858-6).
- Appendicure summary of the preprint, 2026-02-23. [Link](https://appendicure.com/2026/02/23/a-new-immunotherapy-strategy-emerging-from-wake-forest-for-appendiceal-cancer-tumor-on-a-chip/).

**Measurements**

- Basu R et al. Cytotoxic T Cells Use Mechanical Force to Potentiate Target Cell Killing. Cell 2016;165:100–110. [10.1016/j.cell.2016.01.021](<https://www.cell.com/cell/fulltext/S0092-8674(16)00061-1>).
- Tamzalit F et al. Interfacial actin protrusions mechanically enhance killing by cytotoxic T cells. Science Immunology 2019. [Janelia record](https://janelia.org/publication/interfacial-actin-protrusions-mechanically-enhance-killing-by-cytotoxic-t-cells); [preprint 10.1101/443309](https://www.biorxiv.org/content/10.1101/443309v1).
- Vorselen D et al. Microparticle traction force microscopy reveals subcellular force exertion patterns in immune cell–target interactions. Nature Communications 2020;11:20. [10.1038/s41467-019-13804-z](https://www.nature.com/articles/s41467-019-13804-z).
- Hydrogel microparticle force sensors, protocol. Nature Protocols 2025. [10.1038/s41596-025-01281-2](https://doi.org/10.1038/s41596-025-01281-2).
- Cell Image Library. CIL:40011, tomogram of a CTL–target immunological synapse. [Link](https://www.cellimagelibrary.org/images/40011).

**Models**

- Evaluation of CD8 T cell killing models with computer simulations of 2-photon imaging experiments. PLoS Computational Biology 2021. [10.1371/journal.pcbi.1008428](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1008428).
- Heterogeneous, delayed-onset killing by multiple-hitting T cells. PLoS Computational Biology 2020. [10.1371/journal.pcbi.1007972](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1007972).
- Agent-based modeling of cellular dynamics in adoptive cell therapy (ABMACT). bioRxiv 2025. [10.1101/2025.02.17.638701](https://www.biorxiv.org/content/10.1101/2025.02.17.638701v1.full).
- Optimizing T-cell-mediated cancer killing: an agent-based model. bioRxiv 2025. [10.1101/2025.10.21.683801](https://www.biorxiv.org/content/10.1101/2025.10.21.683801v1.full).
- Integrating multiplexed imaging and multiscale modeling identifies tumor phenotype conversion as a critical component of therapeutic T cell efficacy. Cell Systems 2024. [Link](<https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00089-9>).
- Hornak I, Rieger H. Stochastic model of T cell repolarization during target elimination (II). Biophysical Journal 2022. [PMC9034251](https://pmc.ncbi.nlm.nih.gov/articles/PMC9034251/); part I [10.1101/822171](https://www.biorxiv.org/content/10.1101/822171.full.pdf).
- Deterministic mechanical model of T-killer cell polarization reproduces the wandering of aim between simultaneously engaged targets. PLoS Computational Biology 2009. [10.1371/journal.pcbi.1000260](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1000260).
- Agent-based modeling of T cell receptor cooperativity. [PMC7555007](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7555007/).
- A phase-field model for apoptotic cell death. arXiv 2025. [2507.09038](https://arxiv.org/html/2507.09038v3); code [GitHub](https://github.com/Nonlinear-Phenomena-Lab/Phase-field-Apoptosis).
- Runser S, Vetter R, Iber D. SimuCell3D: three-dimensional simulation of tissue mechanics with cell polarization. Nature Computational Science 2024. [10.1038/s43588-024-00620-9](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11052725/).
- PhysiCell releases and attack model. [GitHub](https://github.com/MathCancer/PhysiCell/releases).
- Systems-based digital twins, mosunetuzumab phase I. [PMC10339700](https://pmc.ncbi.nlm.nih.gov/articles/PMC10339700).
- Digital twins elucidate critical role of Tscm in clinical persistence of TCR-engineered cell therapy. npj Systems Biology and Applications 2024. [10.1038/s41540-024-00335-7](https://www.nature.com/articles/s41540-024-00335-7).

**Data-driven methods and data**

- Dekkers JF et al. Uncovering the mode of action of engineered T cells in patient cancer organoids. Nature Biotechnology 2022;41:60–69. [10.1038/s41587-022-01397-w](https://www.nature.com/articles/s41587-022-01397-w); open text at [PMC9849137](https://pmc.ncbi.nlm.nih.gov/articles/PMC9849137/).
- Alieva M et al. BEHAV3D: a 3D live imaging platform for comprehensive analysis of engineered T cell behavior and tumor response. Nature Protocols 2024;19:2052–2084. [10.1038/s41596-024-00972-6](https://www.nature.com/articles/s41596-024-00972-6).
- BEHAV3D example datasets. [Zenodo 8328441](https://zenodo.org/records/8328441); [Dryad 10.5061/dryad.3n5tb2rp6](https://datadryad.org/dataset/doi:10.5061/dryad.3n5tb2rp6); raw data [BioImage Archive S-BIAD448](https://www.ebi.ac.uk/biostudies/studies/s-biad448).
- BEHAV3D code. [AlievaRios/BEHAV3D](https://github.com/AlievaRios/BEHAV3D); [imAIgene-Dream3D/BEHAV3D](https://github.com/imAIgene-Dream3D/BEHAV3D); [BEHAV3D-Explorer](https://github.com/imAIgene-Dream3D/BEHAV3D-Explorer).
- TIMING publications and software, Single-Cell Lab, University of Houston. [Link](https://singlecell.chee.uh.edu/publications/); original method, Bioinformatics 2015. [PDF](https://singlecell.chee.uh.edu/wp-content/uploads/2020/10/Automated-and-Quantitative-Profiling-of-Individual-Cell-Cell-Interactions-from-High-throughput-Time-lapse-Imaging-Microscopy-in-Nanowell-Grids.pdf).
- Video diffusion for early cellular apoptosis forecasting. bioRxiv 2023. [10.1101/2023.11.16.567461](https://www.biorxiv.org/content/10.1101/2023.11.16.567461.full.pdf).
- Holmberg A, Mechtel N, Ouyang W. Adapting video diffusion models for time-lapse microscopy. arXiv 2025. [2503.18583](https://arxiv.org/abs/2503.18583v2).
