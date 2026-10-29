# Cell-Kill-Sim: start here

A plain-language guide to what this project is and what has been done. It assumes
no background in biology or in the code. Every picture comes from the real data.
Last updated 29 October 2026.

<div class="toc" markdown="1">

**Contents:**
[1. The question](#1-the-question) ·
[2. The data](#2-the-data) ·
[3. Turning images into cells](#3-turning-images-into-cells) ·
[4. What we measured](#4-what-we-measured) ·
[5. What we found, and why it is on hold](#5-what-we-found-and-why-it-is-on-hold) ·
[6. How work is checked now](#6-how-work-is-checked-now) ·
[7. Redrawing the cells](#7-redrawing-the-cells-in-progress) ·
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

In every picture below, the **blue line** is the lab's outline of the tumour cell and
the **orange line** is their outline of the T cell, drawn on the raw image.

<figure>
<img src="img/f4_good_mask.jpg" alt="A green T cell and a magenta tumour cell, each closely outlined, seen from above and from the side">
<figcaption><b>A good mask</b> (experiment v01, chamber ROI 5). Seen from above and from the side, each outline
follows its cell.</figcaption>
</figure>

When we looked closely, many masks turned out to be wrong in ways that matter:

<figure>
<img src="img/f5_hollow.jpg" alt="From above the tumour cell outline looks whole; a cut through the middle shows the mask is a broken ring">
<figcaption><b>Hollow mask</b> (v12, ROI 33). From above and from the side the outline looks fine. A cut through the
middle of the cell (right) shows that the mask is only a broken ring, shaded blue: the inside of the cell is not
included. The tumour label marks only the cell's surface, and the lab's network sometimes fails to fill the inside.
Most masks in donor hd10's experiments look like this.</figcaption>
</figure>

<figure>
<img src="img/f6_merged.jpg" alt="One blue outline around two tumour cells; from the side the outline also loops into empty space">
<figcaption><b>Two cells merged, plus a phantom lobe</b> (v01, ROI 32, 40 minutes in). From above, one blue outline
goes around two tumour cells: the large one next to the T cell and a smaller one below it. From the side, the
outline also loops up into empty space where there is no cell. A T cell next to a merged pair can look as if it
has pushed a deep dent into "one" cell.</figcaption>
</figure>

Other failures: a cell split into fragments, and cells cut off at the edge of the
image. In a random check of time points that passed our first quality rule, only
**4 in 10 masks matched the raw image well**. Section 7 also shows that the lab's
masks lose a tumour cell once it has died.

## 4. What we measured

**Contact area:** how much of the T cell touches the target. We reproduced the
lab's published numbers exactly (395 of 395 chambers matched; area within 0.5%
for 369 of 419 cells). This part is solid.

**Cup depth:** how deep a dent the T cell makes in the target. It is the
distance from the target's "filled-in" outline (its convex hull) down to the
contact surface.

<figure>
<img src="img/f7_real_cup.jpg" alt="A magenta tumour cell curving around a green T cell; from the side the blue outline hangs below the cells">
<figcaption><b>A deep cup, measured at 3.7 µm</b> (v10, ROI 23). From above, the tumour cell (magenta) curves around
the T cell (green). But from the side, the blue outline hangs down below the cells where there is no signal,
and that lobe makes the dent look deeper than it is. Many "deep cups" were like this, which is why the cup results are on hold.</figcaption>
</figure>

<figure>
<img src="img/f8_cup_test.jpg" alt="A computer-made round cell (blue) with a 2 micrometre dent where a T cell (orange) presses in; arrows mark the measured depth">
<figcaption><b>Testing the ruler.</b> An artificial cell (blue) with a dent of known depth (2 µm) where an artificial T cell
(orange) presses in, cut through the middle two ways. The arrows show the measured depth: 1.96 µm, right to within one
voxel. So the ruler is fine; the trouble is the bad masks it is applied to.</figcaption>
</figure>

**Death:** a target counts as killed when the death dye appears inside it and
stays. Our kill rates match the paper's per group (52% vs 57.5%, 24% vs 27%,
6% vs 4.5%), but the dye is faint and the calls depend on the threshold.

<figure>
<img src="img/f9_death.jpg" alt="Four snapshots of a magenta tumour cell: at 70 minutes it is clear, then yellow spots appear and by 160 minutes it is filled with yellow death dye">
<figcaption><b>What a death looks like</b> (v02, ROI 3, seen from above). At 70 minutes the tumour cell is clear. Yellow
spots of death dye appear at 110 minutes, and by 160 minutes the cell is full of dye. The T cell (green, right edge)
had touched this cell earlier and moved away. This chamber is labelled "survived" (section 7).</figcaption>
</figure>

**A mechanical model** to turn dents into forces was built and tested, but it
did not fit the real cells reliably: its cells drifted away from the real ones.
It is parked until the measurements are trustworthy.

## 5. What we found, and why it is on hold

| Result | Status |
|---|---|
| Reproduced the lab's contact areas and chamber counts | **Solid** |
| Deeper dents go with faster killing, within each experiment (about 1.4× per standard deviation) | **On hold**: bad masks inflate dents, and they become more common just before death, so this could be an artefact |
| High-killing donors do *not* push in a more focused way than low-killing donors | **On hold**, and the test was inconclusive anyway |
| Mechanical model | **Parked** |
| Which chambers count as "killed" | **In question**: in two of eight chambers we checked, a tumour cell the T cell touched dies but the chamber is labelled "survived" (section 7) |

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

So far 256 gallery panels (146 in the first gallery, 110 in the clean-cell gallery) and 599 movie frames from the redrawing pilot (section 7) have been reviewed.

## 7. Redrawing the cells (in progress)

Instead of trusting the lab's masks, we are redrawing the cells from the raw images
and comparing methods side by side on movies of 8 chambers (4 killed, 4 survived).
Each round, every frame is reviewed and the problems found are fixed in the next round.

**Round 1: four methods on one chamber.** These were the lab's masks, the lab's masks
repaired, Cellpose-SAM (a general AI cell-outlining model) and PlantSeg (a method built
for membrane labels). Cellpose-SAM shattered every cell into pieces, because the tumour
label marks only the membrane and the cells look hollow. PlantSeg had a bug that drew
the T cell as a striped box.

<figure>
<img src="img/f11_p1_four_methods.jpg" alt="Four methods side by side on the same moment: lab masks, repaired lab masks, Cellpose-SAM shattered into pieces, PlantSeg with a striped box">
<figcaption><b>Round 1, one moment, four methods.</b> Blue = tumour cell being tracked, orange = T cell,
grey = other objects. Top row: view from above; bottom row: side view. Third column: Cellpose-SAM breaks
every cell into pieces. Fourth column: PlantSeg's T cell is a striped box (a bug, since fixed).</figcaption>
</figure>

**Round 2: all 8 chambers.** Repairing the lab's masks cut whole round cells into pieces.
More important, the lab's masks drop a tumour cell once it has died: when the cell fills
with death dye, it has no outline. PlantSeg keeps it.

<figure>
<img src="img/f12_oversplit.jpg" alt="One round tumour cell, cut into pieces by the two repaired-mask columns">
<figcaption><b>Over-splitting.</b> One round tumour cell. The lab masks (first column) and PlantSeg (last)
outline it whole; the two repaired versions (middle) cut it into pieces.</figcaption>
</figure>

<figure>
<img src="img/f13_dead_cell_lost.jpg" alt="A dead tumour cell full of yellow dye, outlined only in the last column">
<figcaption><b>A dead cell disappears from the lab's masks.</b> The tumour cell is full of death dye (yellow).
The three lab-based columns have no outline on it; only PlantSeg (right) still outlines it.</figcaption>
</figure>

**Round 3: an improved PlantSeg (called C4).** It knows which colour belongs to which
cell type and handles dead cells separately. It fixes the straight-line cuts and stops
the T cell from swallowing a dead neighbour.

<figure>
<img src="img/f14_c4_fix.jpg" alt="Lab masks, PlantSeg and C4 side by side; PlantSeg's T cell has a straight edge and includes the dead cell, C4's does not">
<figcaption><b>Round 3.</b> Columns: lab masks, PlantSeg, C4. PlantSeg's T cell (orange, middle) has a straight
edge and includes the dead yellow cell; C4's (right) follows the green T cell.</figcaption>
</figure>

**What round 3 showed about killing.** In two chambers labelled "survived", the tumour
cell the T cell touched does die:
- **v02 ROI 3:** the cell dies about 1.5-2 hours after the T cell leaves.
- **v03 ROI 30:** the cell dies at the very last time point, just after contact ends.

Either the lab's masks missed these deaths, or the lab's definition of a kill
excludes them. We are asking the Fei lab.

<figure>
<video src="img/p3_v02_MIP_ROI_3.mp4" controls loop muted playsinline></video>
<figcaption><b>v02 ROI 3, labelled survived.</b> Columns: lab masks, PlantSeg, C4. The T cell (green, right edge)
touches the tumour cell early on, then leaves; the tumour cell later fills with death dye.</figcaption>
</figure>

**Still wrong in C4:**
- touching tumour cells sometimes merge;
- small neighbours get attached;
- the tracking of which cell is which across time slips.

**Round 4: a fair test.** C4 was tuned on those 8 chambers, so we ran it, unchanged,
on 12 chambers it had never seen. Its outlines held up: solid and tight, with none of
the lab masks' lobes hanging into empty space. But two problems now dominate:
- touching tumour cells are often drawn as one;
- the computer often picks the wrong cell as "the target" or "the T cell".

<figure>
<img src="img/f15_heldout.jpg" alt="Lab mask with a large lobe below the cells next to a tight C4 outline, on a chamber C4 had never seen">
<figcaption><b>Round 4, a chamber C4 had never seen</b> (v12, ROI 32). Left: the lab's tumour-cell outline (blue)
hangs far below the cells in the side view. Right: C4 follows the cell.</figcaption>
</figure>

**Round 5: two fixes, tested on all 20 chambers.** We tried a splitter that cuts touching
tumour cells apart, and a tracker that picks the T cell and the target once, from the whole
movie, and follows the same cell to the end. Four people reviewed every frame (the tracker
and the splitter separately).

The splitter did help where two tumour cells touch with a visible neck:

<figure>
<img src="img/f16_p5_split_helps.jpg" alt="Two touching round tumour cells: one outline in the left column, two in the right">
<figcaption><b>Touching cells, separated.</b> Left: one outline around two tumour cells. Right: two.</figcaption>
</figure>

But overall it was better in about 1 frame in 6, the same in 3 in 5, and worse in 1 in 5. It cut
single cells along lines that follow no visible gap, and it deleted some dead cells:

<figure>
<img src="img/f17_p5_split_hurts.jpg" alt="One large faint tumour cell outlined whole on the left, cut into a strip and a remainder on the right">
<figcaption><b>One cell cut in two.</b> A single large, faint tumour cell. Left: outlined whole. Right: cut, and only a strip is tracked.</figcaption>
</figure>

<figure>
<img src="img/f18_p5_dead_lost.jpg" alt="A tumour cell full of yellow death dye outlined on the left and without an outline on the right">
<figcaption><b>A dead cell loses its outline.</b> The tumour cell is full of death dye (yellow). Left: outlined. Right: no outline.</figcaption>
</figure>

We rejected this splitter and rewrote it.

**Round 6: a more careful splitter, and setting aside what is too ambiguous.** The new version
only cuts at a visible gap and never removes a dead cell. It changed little: of 311 frames in
15 chambers it matched the old outlines in about 3 in 4, was better in about 1 in 8 and worse
in about 1 in 16. Where it helps, it removes stray fragments so the target is one clean outline:

<figure>
<img src="img/f19_p6_cleaner.jpg" alt="Grey fragments inside the target on the left, one clean target outline on the right">
<figcaption><b>Cleaner outline.</b> Left: grey fragments cut into the target (551 µm³). Right: one whole target (788 µm³).</figcaption>
</figure>

Where two tumour cells touch it still sometimes cuts along a straight line that follows no gap:

<figure>
<img src="img/f20_p6_bad_cut.jpg" alt="Two touching tumour cells merged on the left, cut by a straight vertical line through one of them on the right">
<figcaption><b>A cut with no gap.</b> Left: two cells in one outline. Right: a straight cut through the right-hand cell.</figcaption>
</figure>

In some chambers two tumour cells both touch the T cell, so even by eye there is no single
"target":

<figure>
<img src="img/f21_p6_ambiguous.jpg" alt="Two tumour cells touch the green T cell; the left column picks the upper one and the right column the lower one">
<figcaption><b>Which one is the target?</b> Two tumour cells touch the T cell. The two methods pick different ones.</figcaption>
</figure>

We have decided to **leave out anything too ambiguous for now**. Ten of the 20 test chambers are
set aside (the five we had already flagged as crowded or faint, plus five where the two methods
follow different cells or the splitter cuts the pair). Eight chambers are clean, and two more are
limited by merged cells that no method has fixed. On the eight clean ones the new outlines plus the
whole-movie tracker are as good as or better than before.

**After that:**
1. Test on a larger sample how often a chamber has an unambiguous target, before redrawing all 502.
2. Agree with the Fei lab on what counts as a kill, then redo the death labels on the clean chambers.
3. **Scale up** by processing raw images as they download, keeping only the cell outlines
   and summaries, so all 502 chambers fit on one computer.
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
| PlantSeg | A published method for outlining cells from membrane images; the basis of our current outlines |
| C4 | Our improved version of PlantSeg (round 3 of the pilot) |
| Tracking | Working out which outline is the same cell from one time point to the next |
