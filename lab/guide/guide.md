# Cell-Kill-Sim: start here

A plain-language guide to what this project is and what has been done. It assumes
no background in biology or in the code. Every picture comes from the real data.
Last updated 28 October 2026.

<div class="toc" markdown="1">

**Contents:**
[1. The question](#1-the-question) ·
[2. The data](#2-the-data) ·
[3. Turning images into cells](#3-turning-images-into-cells) ·
[4. What we measured](#4-what-we-measured) ·
[5. What we found, and why it is on hold](#5-what-we-found-and-why-it-is-on-hold) ·
[6. How work is checked now](#6-how-work-is-checked-now) ·
[7. What happens next](#7-what-happens-next) ·
[8. Glossary](#8-glossary)

</div>

## 1. The question

CAR-T cells are immune cells engineered to recognise and kill cancer cells.
Some patients' CAR-T cells kill well, others poorly. One idea is that good
killers push on their target in a particular way: they press harder, or press
in a more focused spot, before they kill it.

The goal is to build a **4D digital twin** of each encounter: a 3D model of
the T cell and its target, changing over time. Then we can measure how each
T cell pushes and ask whether that predicts killing.

## 2. The data

The data come from the Fei lab (Wang et al. 2024, *Nature Communications*),
who made it public (ScienceDB 19235). They put T cells and leukaemia cells
(Nalm6) together in tiny wells, called **chambers**, and filmed them in 3D with
a light-sheet microscope.

- Each cell type glows in its own colour. In our pictures the **T cell is green**
  (an actin label), the **tumour cell is magenta** (a membrane label), and a
  **death dye shows yellow** once a cell has died.
- Each chamber was imaged every 10 minutes for about 3 hours: 20 **visits**.
  At each visit the microscope takes a quick burst of about 5 3D images,
  called **sweeps**, a few seconds apart.
- In total: 13 experiments, 10 blood donors, 502 chambers, 63,000 3D images,
  4.7 TB of raw images.

<figure>
<img src="img/f1_one_moment.jpg" alt="One 3D image seen from above: a magenta tumour cell with a green T cell pressed against it">
<figcaption><b>One moment in one chamber</b> (experiment v01, chamber ROI 2, 2 hours in), seen from above.
The 3D image is flattened by showing the brightest point along each line of sight.</figcaption>
</figure>

<figure>
<img src="img/f2_time_strip.jpg" alt="Six snapshots 40 minutes apart showing the T cell moving around the tumour cell">
<figcaption><b>The same chamber over 3 hours.</b> The T cell (green) stays on the tumour cell (magenta) and crawls
around it. In this chamber the tumour cell survives.</figcaption>
</figure>

<figure>
<video src="img/v01_MIP_ROI_32.mp4" controls loop muted playsinline></video>
<figcaption><b>A chamber where the tumour cell is killed</b> (v01, ROI 32), all 20 visits. The death dye
(yellow/white) appears at the end. The coloured outlines are the lab's computer-drawn cell outlines (section 3).</figcaption>
</figure>

<figure>
<img src="img/f3_slice_vs_projection.jpg" alt="Left: a single thin slice showing the tumour cell membrane as a ring. Middle: top view of the whole volume. Right: side view">
<figcaption><b>Three ways to look at one 3D image.</b> One thin slice (left) shows the tumour cell's membrane as a ring,
because the label sits only on the cell's surface. The top and side views (middle, right) show the whole cell.
Early in this project we reviewed single slices only, which can miss what happens elsewhere in the cell.</figcaption>
</figure>

## 3. Turning images into cells

To measure anything, the computer needs to know which voxels (3D pixels) belong
to which cell. The Fei lab trained a neural network to label every voxel, and
published these labels, called **masks**. We started from their masks.

<figure>
<img src="img/f4_good_mask.jpg" alt="Top: raw image with blue and orange outlines that follow the cells closely. Bottom: the lab's mask labels">
<figcaption><b>A good mask.</b> Top: the raw image with the outline of the tumour cell (blue) and T cell (orange).
Bottom: the lab's labels. The outline follows the cell. (The top row shows only the magenta channel, so the T cell's outline has no green under it.)</figcaption>
</figure>

When we looked closely, many masks turned out to be wrong in ways that matter:

<figure>
<img src="img/f5_hollow.jpg" alt="A solid magenta cell whose mask is a broken hollow ring">
<figcaption><b>Hollow mask.</b> The cell is solid, but the mask is a broken ring. Because the label marks only
the membrane, the network sometimes fails to fill the inside. Donor hd10's experiments are mostly like this.</figcaption>
</figure>

<figure>
<img src="img/f6_merged.jpg" alt="Two round tumour cells touching, outlined as one object">
<figcaption><b>Two cells merged.</b> Two round tumour cells are labelled as one object. A T cell sitting in the
groove between them then looks as if it has pushed a deep dent into "one" cell.</figcaption>
</figure>

Other failures: a cell split into fragments, a mask extending where there is no
cell, and cells cut off at the edge of the image. In a random check of visits
that passed our first quality rule, only **4 in 10 masks matched the raw image well**.

## 4. What we measured

**Contact area:** how much of the T cell touches the target. We reproduced the
lab's published numbers exactly (395 of 395 chambers matched; area within 0.5%
for 369 of 419 cells). This part is solid.

**Cup depth:** how deep a dent the T cell makes in the target. It is the
distance from the target's "filled-in" outline (its convex hull) down to the
contact surface.

<figure>
<img src="img/f7_real_cup.jpg" alt="A tumour cell wrapping around a T cell, with a red arrow marking a 3.7 micrometre dent">
<figcaption><b>A real deep cup</b> (3.7 µm). The target visibly wraps around the T cell. The green dashed line is
the filled-in outline; the red arrow is the measured depth.</figcaption>
</figure>

<figure>
<img src="img/f8_cup_test.jpg" alt="Synthetic sphere with a 2 micrometre dent and the measured depth">
<figcaption><b>Testing the ruler.</b> On artificial cells with dents of known depth, the measurement is right to within
one voxel (a 2 µm dent measures 1.96 µm). So the ruler is fine; the trouble is the bad masks it is applied to.</figcaption>
</figure>

**Death:** a target counts as killed when the death dye appears inside it and
stays. Our kill rates match the paper's per group (52% vs 57.5%, 24% vs 27%,
6% vs 4.5%), but the dye is faint and the calls depend on the threshold.

<figure>
<img src="img/f9_death.jpg" alt="Death-dye images before, at and after a called death visit">
<figcaption><b>A death call.</b> The death dye (bright) appears on the target at the called visit and grows after it.</figcaption>
</figure>

**A mechanical model** to turn dents into forces was built and tested, but it
did not fit the real cells reliably: its cells drifted away from the real ones.
It is parked until the measurements are trustworthy.

<figure>
<img src="img/f10_model.jpg" alt="Model outlines displaced from the real cell outlines">
<figcaption><b>Why the model is parked.</b> The model's cells (dashed) sit several micrometres away from the real ones (solid).</figcaption>
</figure>

## 5. What we found, and why it is on hold

| Result | Status |
|---|---|
| Reproduced the lab's contact areas and chamber counts | **Solid** |
| Deeper dents go with faster killing, within each experiment (about 1.4× per standard deviation) | **On hold**: bad masks inflate dents, and they become more common just before death, so this could be an artefact |
| High-killing donors do *not* push in a more focused way than low-killing donors | **On hold**, and the test was inconclusive anyway |
| Mechanical model | **Parked** |

The lesson: earlier stages were accepted on numbers and reports, without
looking at enough real images. Problems in the masks then flowed into every
later result.

## 6. How work is checked now

The project is run like a lab:

- **Galleries and movies:** every step is shown as real images, random examples plus
  the most extreme ones, and every image is reviewed. Each gets a recorded verdict
  (ok / problem / unsure) with a note.
- **Lab notebook:** dated entries recording what was run, what was looked at and what was decided.
- **Data registry:** every dataset has a status. Only verified data may be analysed.
- **Gates:** a stage may feed the next only after its review passes.

So far 256 panels have been reviewed (146 in the first gallery, 110 in the clean-cell gallery).

## 7. What happens next

1. **Better cell outlines.** Instead of trusting the lab's masks, we redraw cells from the
   raw images using several methods: the lab's masks repaired, Cellpose-SAM (a general
   AI cell-outlining model), and a membrane-based method. For each cell we pick the
   outline that matches the raw movie. A pilot on 8 chambers (4 killed, 4 not) is starting.
2. **Movies as the standard view** for every chamber, with the measurements plotted
   alongside so each number can be checked against what the cells do.
3. **Scale up** by processing raw images as they download and keeping only the cell
   outlines and summaries, so all 502 chambers fit on one computer.
4. Then, and only then, re-ask the questions in section 5 and return to the mechanical model.

**Open questions for the Fei lab:** the true spacing between 3D slices (the paper
says 0.6 µm, but the images look like about 0.2 µm), and whether deaths seen on only one visit were discarded.

## 8. Glossary

| Term | Meaning |
|---|---|
| Chamber | One tiny well holding a T cell and target cells; one movie |
| Visit | One time point, every 10 minutes |
| Sweep | One 3D image; about 5 per visit, seconds apart |
| Mask | The computer's labelling of which voxels belong to which cell |
| Projection (top/side view) | A 3D image flattened by keeping the brightest point along each line |
| Cup depth | How deep the T cell dents the target |
| Convex hull | The shape you get by shrink-wrapping a cell; dents show up as gaps under it |
| Donor | The person the T cells came from; 10 donors, 5 high-killing and 5 low-killing |
| Dasatinib | A drug that weakens T-cell pushing; used as a control |
| Digital twin | A 3D model of the cells over time, built from the images |
