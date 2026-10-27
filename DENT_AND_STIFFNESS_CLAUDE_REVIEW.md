# Dent and Stiffness Analysis: All-63,053-Frame Rerun & Fix Summary (`D1–D8`, `S1–S3`, `R1–R4`)

## 1. Master Results Provenance (`R1`)

Every number in this report, the CSV tables, and the visual figures is generated from a single master pipeline (`src/run_dent_stiffness_fixes_all_frames.py`) and exported to `data/dent_stiffness_master_results.json`.

| Path | Description |
| --- | --- |
| `data/dent_stiffness_master_results.json` | Single master JSON containing all constants, frame survival counts, synthetic validation, time-ordered tests, rules comparisons, group summaries, and stiffness sweeps (`R1`) |
| `src/run_dent_stiffness_fixes_all_frames.py` | End-to-end 63,053-frame dent & stiffness pipeline implementing all fixes `D1–D8`, `S1–S3`, and `R1–R4` |
| `src/build_dent_stiffness_visual_guide.py` | High-resolution visual guide builder using the original 320-nm light-sheet video models (`MOESM9`, `MOESM6`, `MOESM7`, `MOESM8`) paired with our Harmonic `M=16` 2D Sim Render and Physically Constrained 4D Surface-Net Mesh |
| `dent_stiffness_visual_guide.html` | Self-contained interactive HTML visual guide with real light-sheet video models, `M=16` 2D/4D twin viewer, and master tables |
| `data/dent_step2_frame_log.csv` | Frame-level log across all `63,053` outlined frames in the deposit (`D1`) |
| `data/dent_pair_visit_timeseries.csv` | Visit-by-visit time series (`10,510` visits across `502` wells, `0–210 min`, `10 min/visit`) with morphology death time `t_death_min` (`D1, D4`) |
| `data/dent_step3_synthetic_validation.csv` | 50 synthetic 3D pairs (`35` intact dented + `3` cup-wrapped + `12` pathological blebbed/fragmented/swollen/off-patch controls) (`D2–D4`) |
| `data/dent_step3_20_real_frames_handcheck.csv` | 20 hand-checked real 3D frames across distinct wells (`14` accepted intact in-patch contact + `6` rejected pathological controls) (`D2–D4`) |
| `data/dent_step4_chamber_summary_502.csv` | Per-well summary across all `502` wells (`364` wells with intact pre-death frames; `321` wells with intact pre-death contact) |
| `data/dent_step4_aligned_419_pairs.csv` | All `419` non-`'nd'` single-cell pairs aligned with `contact_flag`, `dent_first_contact_um`, `dent_predeath_max_um`, `ca_3d_predeath_max_um2`, and `t_death_min` |
| `data/dent_step5_feature_separation_auc.csv` | Single-feature Mann–Whitney AUCs across all `329` untreated pairs and restricted to within-contact pairs (`contact_flag == 1`, `n = 229`) (`D6`) |
| `data/dent_step5_rules_comparison.csv` | Out-of-sample rules comparison with uniform 4 metrics + 500-bootstrap 95% CIs (`train_high_mae_pct`, `heldout_low_mae_pct`, `dasatinib_mae_pct`, `predicted_donor_gap_pct`) (`D5, D8, R2`) |
| `data/dent_step5_group_summary.csv` | Per-group summary across all `13` groups (`1–10, 8*, 9*, 10*`), highlighting Low-Killing Groups 8 and 10 (`D7`) |
| `data/target_stiffness_sweep_3d_dent.csv` | Small-strain-capped (`delta <= 0.20 * R0`) target stiffness sweep (`E_target = 50..10,000 Pa`) with force plausibility check (`S1–S3`) |
| `figures/visual_19_dent_vs_contact_real_2d_4d_examples.png` | High-resolution original video models (`MOESM9`, `MOESM6`, `MOESM7`) showing first-contact cup (`55m`), peak crater (`85m`), excluded post-hit blebbing (`100m`), and sub-lethal abort (`81.7m`) |
| `figures/visual_20_dent_validation_rules_and_stiffness_evidence.png` | 6-panel all-63,053-frame evidence summary (`D1–D8`, `S1–S3`) |
| `figures/dent_step3_20_real_frames_handcheck.png` | 20 hand-checked real 3D frames (`14` accepted intact + `6` rejected controls) |
| `figures/dent_step6_summary_figure.png` | Pre-death dent vs. kill rate across all 13 groups, contact catch rate vs. dent, and out-of-sample rules comparison |
| `figures/stiffness_and_force_scale_sweeps.png` | Small-strain-capped force plausibility check (`S1`) and single-line scale non-identifiability (`S2, S3`) |

---

## 2. Provenance of Constants and Thresholds (`R4`)

| Constant | Value | Status | Source / Rationale |
| --- | --- | --- | --- |
| `DX_UM` | `0.1962 µm/vox` (optical FWHM `0.320 µm`) | **cited** | `feilab-hust/CAR-T-analysis` `summary.m` line 18 `sqrt(0.0385 µm²/vox)` & line 16 `19.6 nm/s` (`1 vox / 10 s`) on the reconstructed `190×256×256` ROI grid |
| `CA_VOXEL_FACTOR_UM2` | `0.0385 µm²/vox` | **cited** | `feilab-hust/CAR-T-analysis` `summary.m` line 18 contact area calibration factor |
| `VISIT_INTERVAL_MIN` | `10.0 min` (`6 sweeps/visit`) | **cited** | Wang et al. (*Nat. Commun.* 2024) time-lapse cadence (up to 22 visits over `0–210 min`) |
| `LIVE_RADIUS_RANGE_UM` | `[3.8, 6.8] µm` | **chosen** | Intact live Nalm6 lymphoblast radius bounds (`D2`; excludes swollen dying `> 6.8 µm` and shrunken fragments `< 3.8 µm`) |
| `MIN_SPHERICITY_INTACT` | `0.75` (and `rel_RMSE_nc <= 0.105`) | **chosen** | Intact pre-blebbing target roundness floor (`D2`; excludes post-hit blebbing and irregular outlines) |
| `MIN_CONTACT_AREA_UM2` | `5.0 µm²` | **chosen** | Minimum 3D synaptic contact patch area required to define `contact_flag = 1` and measure an in-patch dent (`D3, D6`) |
| `MAX_APEX_PATCH_DIST_UM` | `1.2 µm` | **chosen** | Deepest inward point on the target surface must lie inside the synaptic contact patch (`D3`) |
| `MAX_DENT_FRAC` | `0.45 * R0` | **chosen** | Maximum physical indentation fraction for an intact unfragmented target (`D2`) |
| `SMALL_STRAIN_CAP_FRAC` | `0.20 * R0` | **cited** | Hertzian contact mechanics small-strain limit $\delta_{\text{eff}} = \min(\delta, R_0 / 5)$ (`S1`) |
| `PLAUSIBLE_FORCE_BOUNDS_NN` | `mean <= 1.5 nN, max <= 2.5 nN` | **cited** | Direct BFP / micropillar / DAFM T-cell synaptic force range (Husson 2011; Basu 2016; Vorselen 2020; `S1`) |
| `ETA_CLUTCH_DEFAULT` | `17.1 pN·s/µm³` | **cited** | Actin-clutch frictional coupling scale in baseline synaptic force model |
| `NU_TARGET / T_CORTEX_PN_UM` | `0.45 / 150 pN/µm` | **cited** | Target Poisson ratio and resting cortical tension in Hertz + cortical tension relation |

---

## 3. All-63,053-Frame Dent & Morphology Pipeline (`D1–D4`, `R3`)

### 3.1 Exact Voxel Pitch Calibration (`0.1962 µm/vox` vs Optical FWHM `0.320 µm`)
In `external/feilab_CART_analysis/main/analysis/summary/summary.m`:
- Line 18 converts surface contact voxels to $\mu\text{m}^2$ via `0.0385` ($\Delta x = \sqrt{0.0385} = 0.1962\,\mu\text{m}/\text{vox}$).
- Line 16 converts retrograde actin flow speed from voxels per 10-second sweep to $\text{nm/s}$ via `19.6` ($1\text{ vox} / 10\text{ s} = 19.6\,\text{nm/s} \implies \Delta x = 196.2\,\text{nm}$).
Using the optical diffraction FWHM (`0.320 µm`) on the `256×256` reconstructed ROI grid previously inflated every target radius by $0.320 / 0.1962 = 1.631\times$ (`8.73 µm` instead of `5.35 µm`). At the exact reconstructed ROI voxel pitch `DX_UM = 0.1962 µm/vox` with Euclidean distance-transform peak splitting for touching target doublets, raw target radii across all 63,053 frames stabilize at:
- **Early visits (`0–40 min`):** median $R_0 = 5.611\,\mu\text{m}$
- **Mid visits (`50–120 min`):** median $R_0 = 5.469\,\mu\text{m}$
- **Late visits (`130–210 min`):** median $R_0 = 5.186\,\mu\text{m}$
- **Intact pre-death frames (`D2–D4`):** median $R_0 = 5.758\,\mu\text{m}$

### 3.2 Frame Survival Funnel Across All 63,053 Outlined Frames (`D1, D2, D3, D4`)

| Pipeline Stage / Rejection Reason | Frame Count | Well Count (of 502) | Aligned 1:1 Pairs (of 419) |
| --- | --- | --- | --- |
| **Total outlined frames in deposit (`D1`)** | **63,053** | **502** | **419** |
| Excluded Sweep 6 (`405-nm SiR-Tubulin / SYTOX MTOC sweep`) | 10,508 | — | — |
| **Non-tubulin 3D sweeps evaluated (`Sweeps 1–5 per visit`)** | **52,545** | **502** | **419** |
| Rejected: `non_intact_blebbing_or_low_sphericity` ($\Psi < 0.75$ or $\text{rel\_RMSE}_{\text{nc}} > 0.105$, `D2`) | 15,738 | — | — |
| Rejected: `target_interior_missing_or_unclosed` (`< 3/4` quadrant arc or unclosed shell, `D2`) | 8,485 | — | — |
| Rejected: `non_intact_radius_out_of_live_range` ($R_0 \notin [3.8, 6.8]\,\mu\text{m}$, `D2`) | 7,016 | — | — |
| Rejected: `non_intact_fragmented_target` ($N_{\text{frag}} > 1$, `D2`) | 4,943 | — | — |
| Rejected: `deepest_point_outside_contact_patch` ($> 1.2\,\mu\text{m}$ from contact patch, `D3`) | 4,735 | — | — |
| Rejected: `dent_exceeds_intact_half_radius` ($\delta > 0.45\,R_0$, `D2`) | 4,342 | — | — |
| Rejected: `target_cell_missing_or_too_small` / `both_cells_missing` | 2,778 | — | — |
| Rejected: `post_death_frame_excluded_D4` ($t \ge t_{\text{death}}$, `D4`) | 1,642 | — | — |
| **Accepted intact pre-death frames (`D2–D4`)** | **2,866** | **364** | **294** |
| **Accepted intact pre-death in-patch contact frames ($CA_{3\text{D}} \ge 5\,\mu\text{m}^2$, `D2–D4`)** | **2,849** | **321** | **254** |

Across the `2,849` accepted intact in-patch contact frames, the empirical dent depth distribution is:
- **Median (`q50`):** `1.438 µm`
- **75th percentile (`q75`):** `1.996 µm`
- **90th percentile (`q90`):** `2.353 µm`
- **95th percentile (`q95`):** `2.531 µm`
- **Maximum (`max`):** `3.026 µm` (eliminating the `5.0–5.9 µm` post-death blebbing/corpse artifacts from the unfiltered pipeline).

### 3.3 Expanded 50-Pair Synthetic Validation & 20-Frame Real Hand-Check (`D2–D4`)
In `data/dent_step3_synthetic_validation.csv` (`50` synthetic 3D pairs at $\Delta x = 0.1962\,\mu\text{m}/\text{vox}$):
1. **35 intact dented spheres and ellipsoids** ($R_t \in \{4.8, 5.5, 6.1\}\,\mu\text{m}$, sphere/prolate/oblate, true dent $0.0\text{–}2.0\,\mu\text{m}$ across $Z, Y, X$ push axes): `34 / 35` (`97.1%`) accepted, **$\text{MAE} = 0.164\,\mu\text{m}$** (`0.84 voxels`).
2. **3 cup-wrapped intact targets** ($R_t = 5.5\,\mu\text{m}$, true dent $0.8, 1.3, 1.8\,\mu\text{m}$): `3 / 3` (`100%`) accepted, **$\text{MAE} = 0.175\,\mu\text{m}$**.
3. **12 pathological controls** (`3` blebbed dying, `3` fragmented dying, `3` swollen dying $R_t = 7.4\text{–}8.3\,\mu\text{m}$, `3` off-patch dimples): **`12 / 12` (`100%`) rejected** by `D2/D3`, and **`49 / 50` (`98.0%`) morphology death classification accuracy (`D4`)**.
4. **20 hand-checked real 3D frames (`figures/dent_step3_20_real_frames_handcheck.png`):** Shows `14` accepted intact contact frames across distinct wells (`dent = 0.17` to `2.72 µm`, $R_0 \in [4.5, 6.6]\,\mu\text{m}, \Psi \in [0.83, 1.00]$) alongside `6` rejected pathological real frames (`#15–16` swollen $R_0 = 7.2\text{–}7.4\,\mu\text{m}$, `#17–18` blebbed/irregular, `#19–20` fragmented).

### 3.4 Time-Ordered First-Contact Dent vs. Subsequent Target Death (`D4`)
Among the `254` aligned single-cell pairs with at least one intact pre-death synaptic contact visit ($CA_{3\text{D}} \ge 5\,\mu\text{m}^2$):
- `240` targets subsequently crossed the sustained morphological death threshold ($t_{\text{death}}$), while `14` remained morphologically intact through the end of their time series.
- Mean **first-contact dent** (`dent_first_contact_um`, measured at the first 1–2 intact contact visits before $t_{\text{death}}$) is `1.342 µm` in pairs with subsequent morphological death vs. `1.158 µm` in intact-surviving pairs (**Mann–Whitney $\text{AUC} = 0.545, p = 0.570$**; first-contact $CA_{3\text{D}}$ $\text{AUC} = 0.615, p = 0.149$). Thus, time-ordered first-contact dent alone does not significantly predict which contacted target will subsequently die.

---

## 4. Within-Contact Control (`D6`), Groups 8 & 10 (`D7`), and Out-of-Sample Rules (`D5, D8, R2`)

### 4.1 All-Pairs vs. Within-Contact Donor Separation (`D6`)
Across all `419` pairs, `dent_predeath_max_um` correlates strongly with `ca_3d_predeath_max_um2` (**Spearman $r_s = 0.796$**) because pairs without a detected intact 3D contact patch have `contact_flag = 0`, `CA_3D = 0`, and `dent = 0`. Within contacted pairs (`contact_flag == 1`, `n = 254`), the correlation drops to **$r_s = 0.303$**.

| Feature | Frame Window (`R3`) | High Mean (All, $n=167$) | Low Mean (All, $n=162$) | High vs. Low AUC (All 329 Untreated) | High Mean (Within-Contact, $n=119$) | Low Mean (Within-Contact, $n=110$) | High vs. Low AUC (Within-Contact Only, $n=229$) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Binary 3D Contact Flag ($CA_{3\text{D}} \ge 5\,\mu\text{m}^2$)** | Intact pre-death visits | 0.713 | 0.675 | **0.519** ($p = 0.458$) | 1.000 | 1.000 | **0.500** ($p = 1.000$) |
| **First-Contact Dent Depth (`dent_first_contact_um`, µm)** | First 1–2 intact contact visits (`D4`) | 0.987 | 0.919 | **0.526** ($p = 0.395$) | 1.385 | 1.362 | **0.504** ($p = 0.908$) |
| **Pre-Death Intact Max Dent (`dent_predeath_max_um`, µm)** | All intact pre-death visits (`D2–D4`) | 1.190 | 1.142 | **0.512** ($p = 0.701$) | 1.670 | 1.692 | **0.473** ($p = 0.488$) |
| **Pre-Death Dent at Peak Contact Area (µm)** | Peak-CA intact pre-death visit | 0.938 | 0.930 | **0.510** ($p = 0.748$) | 1.316 | 1.377 | **0.467** ($p = 0.386$) |
| **3D Pre-Death Max Contact Area ($CA_{3\text{D}}$, µm²)** | All intact pre-death visits | 91.18 | 81.88 | **0.521** ($p = 0.509$) | 127.96 | 121.33 | **0.504** ($p = 0.920$) |
| **Published Contact Area (`CA`, µm², xlsx)** | Published synapse frames | 100.16 | 86.25 | **0.582** ($p = 0.010$) | 107.68 | 97.33 | **0.562** ($p = 0.107$) |
| **Actin Depletion Index (`ADC`, xlsx)** | Published synapse frames | 9.170 | 5.086 | **0.573** ($p = 0.021$) | 9.877 | 6.097 | **0.566** ($p = 0.087$) |
| **Retrograde Actin Flow (`ARFS`, µm/min, xlsx)** | Published synapse frames | 5.007 | 4.348 | **0.548** ($p = 0.132$) | 5.297 | 4.712 | **0.542** ($p = 0.278$) |
| **MTOC Angle (`MPA`, deg, xlsx, inverted)** | Published tubulin sweep | 36.73 | 35.16 | **0.501** ($p = 0.968$) | 35.65 | 32.93 | **0.489** ($p = 0.760$) |
| *Unfiltered All-Frame Max Dent (µm, contaminated pre-fix)* | *All 20 visits (unfiltered)* | *4.206* | *3.988* | *0.561 ($p = 0.054$)* | *4.175* | *4.186* | *0.510 ($p = 0.796$)* |

**Conclusion on `D6`:** Once evaluated on all 63,053 frames and restricted to intact pre-death targets (`D2–D4`), **3D dent does not beat the binary contact flag (`AUC = 0.512` vs `0.519` over all pairs, and `AUC = 0.473` within contacted pairs)**. The apparent donor separation in the earlier 3-frame run came from post-kill target swelling/blebbing and differential contact catch rates in the 3 sampled frames. We therefore withdraw the claim that synaptic dent explains half the donor kill gap.

### 4.2 Group-by-Group Summary Highlighting Low-Killing Groups 8 and 10 (`D7`)

| Group | Condition | $n$ | Observed Kill (%) | Contact Flag (%) | First-Contact Dent (µm) | Pre-Death Max Dent (µm) | Within-Contact Max Dent (µm) | 3D Pre-Death CA (µm²) | Published CA (µm²) | Actin Flow (µm/min) | Pre-Death Median $R_0$ (µm) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1` | Untreated high-killing | 33 | 54.5% | 69.7% | 1.102 | 1.281 | 1.838 | 94.2 | 119.9 | 5.24 | 5.60 |
| `2` | Untreated high-killing | 35 | 65.7% | 71.4% | 0.860 | 1.050 | 1.470 | 60.6 | 76.9 | 3.38 | 5.50 |
| `3` | Untreated high-killing | 34 | 55.9% | 61.8% | 0.727 | 0.919 | 1.487 | 61.4 | 90.5 | 4.05 | 5.50 |
| `4` | Untreated high-killing | 31 | 51.6% | 74.2% | 1.058 | 1.282 | 1.728 | 84.4 | 99.9 | 7.01 | 5.50 |
| `5` | Untreated high-killing | 34 | 58.8% | 79.4% | 1.199 | 1.434 | 1.806 | 155.7 | 114.8 | 5.59 | 5.68 |
| `6` | Untreated low-killing | 34 | 35.3% | 61.8% | 0.653 | 0.841 | 1.362 | 75.7 | 73.0 | 3.62 | 5.50 |
| `7` | Untreated low-killing | 24 | 22.2% | 58.3% | 0.719 | 0.856 | 1.467 | 58.2 | 85.1 | 3.21 | 5.55 |
| **`8`** | **Untreated low-killing** | **36** | **24.2%** | **69.4%** | **0.978** | **1.271** | **1.831** | **93.4** | **91.7** | **5.03** | **5.76** |
| `9` | Untreated low-killing | 33 | 25.0% | 54.5% | 0.807 | 0.952 | 1.746 | 55.6 | 74.4 | 4.07 | 5.50 |
| **`10`** | **Untreated low-killing** | **36** | **27.8%** | **88.9%** | **1.348** | **1.661** | **1.868** | **116.0** | **105.1** | **5.37** | **5.68** |
| `8*` | Dasatinib 50 nM | 29 | 6.9% | 41.4% | 0.453 | 0.464 | 1.123 | 22.8 | 41.8 | 2.31 | 5.10 |
| `9*` | Dasatinib 50 nM | 25 | 8.0% | 24.0% | 0.276 | 0.352 | 1.465 | 16.6 | 44.2 | 0.99 | 5.50 |
| `10*` | Dasatinib 50 nM | 35 | 0.0% | 20.0% | 0.104 | 0.104 | 0.520 | 8.6 | 29.7 | 0.39 | 5.50 |

**Conclusion on `D7`:** Low-killing **Group 8 (`24.2%` kill)** and **Group 10 (`27.8%` kill)** have intact pre-death dents (`1.271 µm` and `1.661 µm`), within-contact dents (`1.831 µm` and `1.868 µm`), and 3D contact areas (`93.4 µm²` and `116.0 µm²`) that are as large as or larger than High-Killing Groups 1–5 (`0.919–1.434 µm` pre-death dent, `51.6–65.7%` kill).

### 4.3 Out-of-Sample Rules Comparison with Dasatinib Test (`D5, D8, R2`)
All rules are fitted on Untreated High-Killing Groups 1–5 only and evaluated frozen on Held-Out Low-Killing Groups 6–10 and Held-Out Dasatinib 50 nM Groups 8*–10* (`observed High–Low donor gap = 30.41 pp`):

| Rule (`R2`) | Frame Window (`R3`) | Params | Train High MAE (pp) [95% CI] | Held-Out Low MAE (pp) [95% CI] | Dasatinib 50 nM MAE (pp) [95% CI] | Passes Dasatinib (`<= 10 pp`, `D5`) | Predicted Donor Gap (pp) [95% CI] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **Rule 0: Contact flag alone $\mathbb{I}(CA_{3\text{D}} \ge 5\,\mu\text{m}^2)$** | 3D intact pre-death visits | 1 | 5.72 [3.0, 11.4] | 26.31 [18.4, 33.5] | 17.78 [10.7, 24.8] | **Fail** | 3.76 [−3.9, 11.2] |
| **Rule 1a: Contact Area alone (published xlsx)** | Published synapse frames | 1 | 7.01 [5.4, 9.5] | 25.58 [22.2, 29.2] | 23.38 [17.9, 28.8] | **Fail** | 4.68 [0.9, 8.1] |
| **Rule 1b: 3D Pre-Death Contact Area alone** | 3D intact pre-death visits | 1 | 7.49 [4.0, 13.7] | 26.74 [18.5, 34.9] | 14.38 [7.7, 22.1] | **Fail** | 4.28 [−4.3, 12.4] |
| **Rule 2a: First-Contact Dent + Contact Area (`D4`)** | First 1–2 intact contact visits | 2 | 6.55 [4.1, 9.7] | 25.95 [21.9, 30.0] | 20.35 [15.1, 25.8] | **Fail** | 4.25 [0.2, 8.6] |
| **Rule 2b: Pre-Death Intact Max Dent + Contact Area (`D2–D4`)** | All intact pre-death visits | 2 | 6.46 [3.9, 9.4] | 26.78 [22.4, 30.6] | 19.35 [14.1, 24.2] | **Fail** | 3.71 [−0.3, 8.1] |
| **Rule 2c: First-Contact Dent alone (`D4`)** | First 1–2 intact contact visits | 1 | 6.73 [3.4, 13.4] | 23.23 [16.7, 31.2] | 12.40 [5.5, 20.4] | **Fail** | 7.09 [−1.2, 15.4] |
| **Rule 2d: Pre-Death Intact Max Dent alone (`D2–D4`)** | All intact pre-death visits | 1 | 6.98 [3.6, 12.7] | 23.96 [17.6, 31.2] | 11.14 [4.9, 18.8] | **Fail** | 6.27 [−1.7, 13.7] |
| *Rule 2e: Unfiltered All-Frame Dent + CA (pre-fix baseline)* | *All 20 visits (unfiltered)* | *2* | *4.55 [3.9, 5.6]* | *28.25 [26.6, 29.7]* | *30.00 [26.3, 34.1]* | ***Fail*** | *2.16 [0.3, 4.1]* |
| **Rule 3: Contact Area $\times$ Actin Flow comparator** | Published synapse frames | 1 | 11.38 [8.7, 15.5] | 22.67 [16.8, 28.7] | **9.21 [5.3, 14.9]** | **Pass** | 7.06 [0.9, 12.8] |
| **Rule 4: Clutch-Force Synaptic Physics Model** | Published synapse frames | 2 | 11.83 [8.3, 18.5] | **17.64 [11.9, 26.9]** | **5.23 [1.6, 11.2]** | **Pass** | **10.97 [2.6, 18.9]** |
| **Rule 5: Clutch-Force + Pre-Death Intact Dent** | Intact pre-death visits + xlsx | 3 | 11.02 [7.0, 17.4] | 19.31 [13.4, 27.6] | **5.75 [1.9, 11.7]** | **Pass** | 9.92 [1.3, 18.7] |

**Conclusions on `D5` and `D8`:**
1. **Dasatinib Test (`D5`):** Every rule that relies on contact area or geometric dent without active retrograde actin flow (`Rules 0, 1a, 1b, 2a, 2b, 2c, 2d, 2e`) **fails the Dasatinib test** (`Dasatinib MAE = 11.14 to 30.00 pp`) because Lck-inhibited CAR-T cells in Groups `8*, 9*, 10*` still establish passive physical contacts and modest target indentations (`contact_flag = 20–41%`, within-contact dent `0.52–1.47 µm`) despite having near-zero retrograde actin flow and near-zero killing (`0.0–8.0%`). Therefore, **target dent is not a standalone push rule**. Only rules gated by active retrograde actin flow / clutch force (`Rule 3: CA × Flow = 9.21 pp`, `Rule 4: Clutch Physics = 5.23 pp`) pass the Dasatinib test.
2. **Overlapping Bootstrap Confidence Intervals (`D8`):** With only 5 training donor groups and 5 held-out low-killing donor groups, the 500-bootstrap 95% confidence intervals on Held-Out Low-Killing MAE overlap across rules (e.g., `Rule 4` `[11.9, 26.9] pp` vs `Rule 3` `[16.8, 28.7] pp` vs `Rule 2d` `[17.6, 31.2] pp`). Differences on held-out untreated low-killing donors are therefore **suggestive, not decisive**, whereas the Dasatinib separation (`5.23 pp` vs `19.35–25.58 pp`) is the primary mechanistic discriminator.

---

## 5. Small-Strain-Capped Target Stiffness Sweep (`S1–S3`)

In `src/run_dent_stiffness_fixes_all_frames.py` (`Step 6`), we feed each cell's intact pre-death dent (`dent_predeath_max_um`) into the Hertz + cortical tension contact relation across `25` log-spaced values of $E_{\text{target}} \in [50, 10\,000]\,\text{Pa}$ while enforcing:
1. **Small-strain cap (`S1`):** $\delta_{\text{eff}} = \min(\delta, 0.20\,R_0)$ so no large-strain violation enters $F_{\text{Hertz}} = \frac{4}{3}\frac{E_{\text{target}}}{1-\nu^2}\sqrt{R_{\text{eff}}}\,\delta_{\text{eff}}^{3/2} + \pi T_{\text{cortex}}\delta_{\text{eff}}$.
2. **Force plausibility check (`S1`):** Any $E_{\text{target}}$ where the implied training-cohort mean force exceeds `1.5 nN` or maximum force exceeds `2.5 nN` is flagged as outside measured T-cell synaptic force bounds.

| Assumed $E_{\text{target}}$ (Pa) | Implied Train Mean / Max Force (nN) | Plausible Force (`<= 1.5 / 2.5 nN`, `S1`) | Fixed Threshold: Train MAE (pp) | Fixed Threshold: Held-Out Low MAE (pp) | Fixed Threshold: Dasatinib MAE (pp, `S2`) | Refit Threshold: $G_{\text{crit}}$ / Pinned (`S3`) | Refit Threshold: Held-Out Low MAE (pp) | Refit Threshold: Dasatinib MAE (pp, `S2`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `50.0` | 0.44 / 0.86 | **Yes** | 5.39 | 23.89 | **27.21 (Fail)** | `0.020` (**Pinned**) | 27.73 | 38.97 |
| `97.0` | 0.54 / 1.07 | **Yes** | 5.28 | 24.76 | **27.61 (Fail)** | `0.020` (**Pinned**) | 27.75 | 38.98 |
| `187.9` | 0.72 / 1.45 | **Yes** | 5.32 | 25.66 | **28.11 (Fail)** | `0.029` (Free) | 26.13 | 32.18 |
| `292.1` | 0.93 / 1.90 | **Yes** | 5.56 | 26.17 | **28.44 (Fail)** | `0.026` (Free) | 26.82 | 35.12 |
| `364.6` | 1.08 / 2.22 | **Yes** | 5.75 | 26.40 | **28.60 (Fail)** | `0.020` (**Pinned**) | 27.79 | 39.00 |
| `454.9` | 1.26 / **2.62** | No (`> 2.5 nN`) | 5.98 | 26.59 | **28.76 (Fail)** | `0.056` (Free) | 24.44 | 25.10 |
| `882.0` | **2.12** / **4.52** | No (`> 1.5 nN`) | 6.81 | 26.99 | **29.15 (Fail)** | `0.150` (Free) | 22.49 | 16.51 |
| `10,000.0` | **20.53** / **45.21** | No (`> 1.5 nN`) | 8.21 | 27.26 | **29.60 (Fail)** | `0.214` (Free) | 22.62 | 12.11 |

**Conclusions on `S1–S3`:**
1. **Force Plausibility Bounds (`S1`):** With the small-strain cap $\delta_{\text{eff}} = \min(\delta, 0.20\,R_0)$, only `10 / 25` grid points ($E_{\text{target}} \in [50.0, 364.6]\,\text{Pa}$) keep the implied T-cell forces within measured bounds ($\text{mean} \le 1.5\,\text{nN}, \text{max} \le 2.5\,\text{nN}$). Above `365 Pa`, implied synaptic forces become non-physically large (reaching `45.2 nN` at `10 kPa`).
2. **Fixed-Threshold Dasatinib Failure & Withdrawal of Optimum Claim (`S2`):** Under a fixed literature threshold, the dent-derived Hertz model fails the Dasatinib test across the entire sweep (`fixed_dasatinib_mae_pct = 27.21 to 29.60 pp`) and never improves on the clutch physics model (`23.89–27.26 pp` vs `17.64 pp`). We explicitly withdraw the earlier "shallow optimum at 350 to 1,500 Pa" claim.
3. **Single-Line Scale Non-Identifiability (`S3`):** When the threshold is refitted at each $E_{\text{target}}$, the optimizer either pins $G_{\text{crit}}$ at its lower bound (`0.020`, `8 / 25` points) or scales $G_{\text{crit}}$ proportionally to $E_{\text{target}}$, yielding a flat error curve (`22.44–27.79 pp`). **Within a single target cell line (Nalm6), target stiffness is absorbed by the pore threshold and cannot be identified from kill rates.**
