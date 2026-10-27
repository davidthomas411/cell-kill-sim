# Single-cell T cell killing model: first target

## Aim

Point an existing cell-interaction model at one question that open data can answer, fit its parameters to that data, and test it against held-out measurements. The question is chosen so that every parameter has something measured to fit against, and so that the module built here is the one later levels of the model reuse.

## The question

When one T cell kills several tumor cells from a single anchor point on an organoid, is that because the T cell differs from its neighbours (grips harder, recovers faster, stays longer), or because it landed in a position with many targets within reach?

The two explanations separate:

- Cell property: a given T cell's interval between kills is similar wherever it sits, and a T cell that kills quickly once kills quickly again.
- Geometry: the interval depends on local target density and lengthens as nearby targets are used up.

## Model, first level

The T cell is a point with a state (searching, engaged, refractory) and a position. The organoid is a set of target positions, each with a damage counter. Time steps in 2-minute ticks to match the data.

Each tick:

- Searching: move, check for targets within reach.
- Engaged: add damage to the current target.
- Target crosses threshold: mark dead, enter refractory, then take the next target within reach, or detach.
- Detach at any time with a fixed probability per tick.

## Parameters and what each is fitted to

| Parameter | Meaning | Fitted against |
|---|---|---|
| R | Reach of the T cell, µm | Distance at which contacts register; reported T cell elongation during engagement |
| k | Damage rate per unit contact time | Time from contact start to the organoid's dead-dye rise |
| tau | Refractory time between kills | Gap between successive dye-rise events on one organoid while the same T cell stays attached |
| p_detach | Probability per tick of letting go | Contact durations that end without a kill |
| sigma | Between-cell variability in k | Spread in time-to-kill across T cells, conditioned on local target density |

sigma is the quantity the question turns on. The others are nuisance parameters that have to be pinned down first.

## The test

Fit twice:

1. sigma fixed at zero. All T cells identical; serial killing can only come from where they land.
2. sigma free.

Compare on measurements not used in the fit: kills per T cell, and whether a T cell that killed fast once kills fast again. If the sigma = 0 fit reproduces the observed spread, serial killing is geometry. If it cannot, there is a real killer phenotype, and sigma measures its size.

## Data

BEHAV3D, engineered T cells with patient-derived tumour organoids, confocal, one frame every 2 minutes, up to 24 h.

- Processed track tables, demo well (CD8 T cells with breast cancer organoids, 2019-06-26): 1,171 tracked T cells, roughly 250,000 rows, one row per cell per frame. Columns: position X/Y/Z in µm, speed, squared displacement, distance to nearest organoid, dead-cell dye intensity inside the T cell. Field is about 707 × 707 µm, 7 to 64 µm deep.
- Organoid tables from the same wells: position, surface area, volume, mean dead-cell dye.
- Source code and demo tables: https://github.com/AlievaRios/BEHAV3D ; newer pipeline with pretrained Cellpose models at https://github.com/imAIgene-Dream3D/BEHAV3D-Explorer
- Raw image volumes, two wells: https://zenodo.org/records/8328441 (CC0, Imaris .ims, 15.1 GB and 24.3 GB). Mirror on Dryad, doi:10.5061/dryad.3n5tb2rp6
- Raw imaging behind the published behaviour map: BioImage Archive accession S-BIAD448
- Single-cell RNA-seq of T cells sorted by behaviour: GEO GSE172325

## Work before fitting

Link the two tables. Each T cell has to be assigned to the organoid it is touching, so that a rise in an organoid's dye signal can be attributed to the T cells on it. This is a nearest-surface join on the position tables, not new imaging. Everything else follows from it.

## Failure mode and fallback

Organoids are segmented as whole objects, so a death appears as a step in that organoid's dye signal rather than as a labelled single cell. In large organoids, individual deaths blur together and tau cannot be measured.

Check this first on a handful of small organoids. If individual steps are not resolvable, switch to the fallback question, which needs only the T cell table and no linkage: does the T cell's own death track cumulative engagement (a cost of killing) or contact count and location (something the target does back)? Same data, same table, one less join.

## What this model does not represent

Stated so that nobody over-reads the fit:

- Shape. A real T cell flattens onto the organoid and sends protrusions between epithelial cells, extending several-fold. Here that is one number, R.
- Force. The real cell pulls and the target deforms. Nothing pushes on anything.
- The target as a cell. Real tumour cells sit in a tissue, resist, and die over minutes with blebbing and fragmentation. Here a target is a counter crossing a threshold.
- Mechanism. k stands in for granule polarization, perforin pores, granzyme entry and caspase activation, each with its own timescale.
- T cell state over time. Real cells exhaust and deplete granules; model cells are identical at hour 1 and hour 20.

This is killing as an accounting problem. It is the right level for the question above, and it is the baseline any physical model has to beat.

## Levels to add, and what each needs

| Level | What is added | Fitted or checked against | Status of the data |
|---|---|---|---|
| 1 | Points, reach, damage rate, refractory, detachment | Contact durations, time to kill, kills per cell | Open, in hand |
| 2 | T cell state: damage rate declining with cumulative kills; own death depending on cumulative engagement | Decline in kill rate along long tracks; dead-dye rise versus contact history | Open, in hand |
| 3 | Shape without mechanics: deformable cell with surface tension, adhesion and a motility drive; reach becomes a consequence of elongation | T cell volume and elongation during engagement | Open, but near the resolution floor (about 1.4 µm per pixel, 2-minute frames) |
| 4 | Target tissue: organoid cells as individual adherent deformable cells; dying cells extruded, neighbours closing the gap | Organoid volume and surface-area curves, in aggregate | Open, but individual tumour cells are not resolved in the segmentation |
| 5 | Mechanics: protrusions pushing with measured forces, target membrane tension, adhesion as a spring | Contact area, membrane deformation, actin flow | Open for one system only (see below) |
| 6 | Kill chemistry: perforin pore rate as a function of membrane tension, granzyme entry, caspase activation, fragmentation | Morphology and timing of death; drug-treated arm as a perturbation | Same dataset as level 5 |

Data for levels 5 and 6: a light-sheet deposit of CAR-T and leukemia cell pairs imaged at 320 nm isotropic resolution in bursts of about 2.5 s, with segmentation masks separating nucleus and cytoplasm for both cells, synapse area and actin flow tabulated, and a drug-treated control arm. Science Data Bank doi:10.57760/sciencedb.19235; data descriptor at https://www.nature.com/articles/s41597-026-06829-9 ; analysis code at https://github.com/feilab-hust/CAR-T-analysis

## Ceiling with open data

Levels 1 and 2 can be fitted and tested properly; the data were produced for these measurements. Level 3 can be fitted but will be loosely constrained. Level 4 can be run and compared in aggregate, not validated cell by cell. Levels 5 and 6 can be built and validated for one system only: CAR-T cells against a suspension leukemia target, not T cells in an organoid.

The gap is therefore in the middle: no open dataset images T cells killing inside an organoid at the resolution needed for mechanics. Closing it means acquiring that imaging. Until then the modelling can proceed through level 4 on open data, and levels 5 and 6 proceed in the simpler geometry.

## Conventions

- Python 3.11, one repository, pinned environment.
- Raw data never modified; every derived table written with a data dictionary.
- Every number in a report traceable to a script and an input file.
- Simulation output and real data pass through the same measurement code, so that a fit compares like with like.
