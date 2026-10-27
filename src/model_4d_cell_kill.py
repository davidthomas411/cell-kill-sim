#!/usr/bin/env python3
"""
4D Mechanochemical Single-Cell, Multi-T-Cell Cooperative Swarm & Serial-Killing Digital Twin (CellKillTwin4D).

Couples:
  Layer 1: 3D chemotactic approach, CCL3/CCL4 swarm recruitment, synapse adhesion, dwell, detachment & serial killing
  Layer 2: Deformable 3D surface mechanics (multi-synapse indentation, WASP/Arp2/3 actin
           protrusions, CTL prolate elongation, target volume collapse & fragmentation)
  Layer 3: Intracellular MTOC & lytic granule polarization to the synaptic cleft
  Layer 4: Additive tension-potentiated perforin pore kinetics across N cooperating T cells
           competing against ESCRT membrane repair, Granzyme B influx, Caspase-3
           bistable activation, and nuclear dead-dye uptake (NucRed/TO-PRO-3/SYTOX)

Calibrated against:
  - Zenodo 8328441 (LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims MegaSurfaces0/1/3)
  - Wang et al., Sci Data 2026 / Nat Commun 2024 (419 CAR-T-Nalm6 pairs at 320 nm light sheet)
  - BEHAV3D DemoDataset1..6 (3,598 T-cell tracks) & scRNA-seq (1,814 CD8 TEGs)
  - Basu et al., Cell 2016; Tamzalit et al., Sci Immunol 2019; Vorselen et al., Nat Commun 2020;
    Weigelin et al., Nat Commun 2021; Hornak & Rieger, Biophys J 2022; Dekkers et al., Nat Biotechnol 2022.
"""

import json
import os
import numpy as np
import pandas as pd

ROOT = "/Users/dxt397/Documents/_PROJECTS/Cell-Kill-Sim"
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Starting 3D coordinates (um) and approach axes for up to 5 cooperating T cells
CTL_STARTS = [
    np.array([-26.0, -12.0, -3.0], dtype=float),
    np.array([-12.0,  25.0,  6.5], dtype=float),
    np.array([ 18.0, -24.0, -5.5], dtype=float),
    np.array([-20.0,  16.0, -9.0], dtype=float),
    np.array([ 22.0, -14.0,  8.5], dtype=float),
]

# Distinct docking unit vectors on the target surface so multiple T cells form separate synapses
CTL_DOCK_DIRS = [
    np.array([-0.89, -0.42, -0.15], dtype=float),
    np.array([-0.42,  0.86,  0.28], dtype=float),
    np.array([ 0.60, -0.76, -0.24], dtype=float),
    np.array([-0.68,  0.55, -0.48], dtype=float),
    np.array([ 0.72, -0.48,  0.50], dtype=float),
]
for _i in range(len(CTL_DOCK_DIRS)):
    CTL_DOCK_DIRS[_i] = CTL_DOCK_DIRS[_i] / np.linalg.norm(CTL_DOCK_DIRS[_i])


def simulate_preset(
    preset_id: str,
    F_syn_max_nN: float = 1.20,
    E_target_Pa: float = 850.0,
    beta_tension: float = 2.40,
    k_repair_min: float = 0.085,
    dwell_cap_min: float = 90.0,
    serial_kill: bool = True,
    n_effectors: int = 1,
    dt_min: float = 1.0,
    t_max_min: float = 180.0,
) -> dict:
    """
    Simulate coupled 4D mechanochemical ODEs + 3D morphological observables
    for 1 to 5 effector T cells engaging 1 or 2 tumor targets.
    """
    n_effectors = int(np.clip(n_effectors, 1, 5))
    steps = int(round(t_max_min / dt_min)) + 1
    t_arr = np.linspace(0.0, t_max_min, steps)

    # Target initial states (Target 1 at origin, Target 2 at (+24, +13, +3.5) um)
    targets = [
        {
            "id": 1,
            "pos": np.array([0.0, 0.0, 0.0]),
            "R0_um": 9.5,  # V0 ~ 3595 um^3 (or multicellular cluster core)
            "V0_um3": 3595.0,
            "V_um3": 3595.0,
            "sigma_mem_mN_m": 0.05,
            "pores": 0.0,
            "granzyme": 0.0,
            "caspase3": 0.0,
            "dead_dye": 0.9,
            "bleb_amp": 0.0,
            "n_frag": 1,
            "indent_um": 0.0,
            "contact_area_um2": 0.0,
            "killed_time_min": None,
        },
        {
            "id": 2,
            "pos": np.array([24.0, 13.0, 3.5]),
            "R0_um": 9.8,
            "V0_um3": 3940.0,
            "V_um3": 3940.0,
            "sigma_mem_mN_m": 0.05,
            "pores": 0.0,
            "granzyme": 0.0,
            "caspase3": 0.0,
            "dead_dye": 0.8,
            "bleb_amp": 0.0,
            "n_frag": 1,
            "indent_um": 0.0,
            "contact_area_um2": 0.0,
            "killed_time_min": None,
        },
    ]

    # Initialize N effector T cells
    # Pioneer cell 0 starts moving immediately; followers 1..N-1 are recruited by CCL3/CCL4
    # with staggered chemotactic activation delays (0, 6, 12, 18, 24 min)
    ctls = []
    for ci in range(n_effectors):
        p0 = CTL_STARTS[ci].copy()
        ax0 = -p0 / max(np.linalg.norm(p0), 1e-6)
        ctls.append({
            "id": ci + 1,
            "subtype": "CD4" if (ci == 3 and n_effectors >= 4) else "CD8",
            "pos": p0,
            "axis": ax0,
            "dock_dir": CTL_DOCK_DIRS[ci].copy(),
            "recruit_delay_min": float(ci * 6.0),
            "active_target_idx": 0,
            "in_contact": 0,
            "dwell_min": 0.0,
            "mtoc_pol": 0.08,
            "F_syn_nN": 0.0,
            "A_syn_um2": 0.0,
            "arfs_nm_s": 0.0,
            "indent_um": 0.0,
            "La_um": 8.3,
            "Lb_um": 10.5,
            "Lc_um": 13.6,
            "sph": 0.89,
            "speed_um_min": 0.35,
        })

    sigma_0 = 0.05  # resting membrane tension (mN/m)
    records = []

    for step_i, t in enumerate(t_arr):
        # Update each effector T cell
        for ci, ctl in enumerate(ctls):
            t_idx = ctl["active_target_idx"]
            tgt = targets[t_idx]
            R_tgt_eff = tgt["R0_um"] * ((tgt["V_um3"] / tgt["V0_um3"]) ** (1.0 / 3.0))
            R_ctl_eff = 5.1

            # Target docking point for this T cell
            dock_pt = tgt["pos"] + ctl["dock_dir"] * (R_tgt_eff + R_ctl_eff - 1.2)
            vec_to_tgt = tgt["pos"] - ctl["pos"]
            dist_center = float(np.linalg.norm(vec_to_tgt))
            unit_dir = vec_to_tgt / max(dist_center, 1e-6)
            surface_gap = max(0.0, dist_center - (R_tgt_eff + R_ctl_eff - 0.8))

            target_dead = tgt["dead_dye"] >= 10.0 and tgt["caspase3"] >= 0.82
            dwell_exceeded = ctl["dwell_min"] >= dwell_cap_min

            if t < ctl["recruit_delay_min"]:
                # Patrolling before chemokine recruitment wave arrives
                ctl["speed_um_min"] = 0.65
                ctl["in_contact"] = 0
            elif not ctl["in_contact"] and not target_dead and not dwell_exceeded:
                vec_dock = dock_pt - ctl["pos"]
                dist_dock = float(np.linalg.norm(vec_dock))
                if surface_gap > 1.2 and dist_dock > 1.8:
                    # Chemotactic approach (followers accelerated slightly by CCL3/CCL4 gradient)
                    speed_um_min = 2.15 + (0.25 if ci > 0 else 0.0)
                    step_dir = vec_dock / max(dist_dock, 1e-6)
                    ctl["pos"] = ctl["pos"] + step_dir * min(dist_dock, speed_um_min * dt_min)
                    ctl["axis"] = 0.78 * ctl["axis"] + 0.22 * unit_dir
                    ctl["axis"] /= max(np.linalg.norm(ctl["axis"]), 1e-6)
                    ctl["speed_um_min"] = speed_um_min
                else:
                    ctl["in_contact"] = 1
                    ctl["speed_um_min"] = 0.35
            elif ctl["in_contact"]:
                ctl["dwell_min"] += dt_min
                tug_offset = 0.0
                if tgt["caspase3"] > 0.45:
                    tug_offset = 1.6 * min(1.0, (tgt["caspase3"] - 0.45) / 0.5)
                desired_dist = R_tgt_eff + R_ctl_eff - 1.35 + tug_offset
                # Dock at assigned orientation around target
                dock_vec = ctl["dock_dir"] if ci > 0 else -unit_dir
                ctl["pos"] = tgt["pos"] + dock_vec * desired_dist
                ctl["axis"] = -dock_vec
                ctl["speed_um_min"] = 0.42 + (0.65 if tug_offset > 0 else 0.0)

                if (target_dead and tgt["caspase3"] >= 0.90) or dwell_exceeded:
                    ctl["in_contact"] = 0
                    if serial_kill and ctl["active_target_idx"] == 0:
                        ctl["active_target_idx"] = 1
                        ctl["dwell_min"] = 0.0
                        ctl["mtoc_pol"] *= 0.35
            else:
                # Post-detachment migration away
                away_dir = ctl["dock_dir"] if ci > 0 else np.array([0.74, -0.64, 0.20])
                away_dir = away_dir / max(np.linalg.norm(away_dir), 1e-6)
                ctl["speed_um_min"] = 1.65
                ctl["pos"] = ctl["pos"] + away_dir * ctl["speed_um_min"] * dt_min
                ctl["axis"] = away_dir

            # Update MTOC polarization (Hornak & Rieger 2022, tau ~ 4.2 min)
            sig_drive = float(np.clip(F_syn_max_nN / 0.65, 0.22, 1.0))
            if ctl["in_contact"]:
                ctl["mtoc_pol"] += (sig_drive - ctl["mtoc_pol"]) * (dt_min / 4.2)
            else:
                ctl["mtoc_pol"] += (0.12 - ctl["mtoc_pol"]) * (dt_min / 6.0)
            ctl["mtoc_pol"] = float(np.clip(ctl["mtoc_pol"], 0.0, 1.0))

            # Synaptic force & contact mechanics per effector
            if ctl["in_contact"]:
                ramp = min(1.0, ctl["dwell_min"] / 5.0) * (0.3 + 0.7 * ctl["mtoc_pol"])
                lysis_decay = max(0.25, 1.0 - 0.65 * tgt["caspase3"])
                # CD4 helper exerts ~0.75x mechanical force of CD8 but sensitizes target via IFNG/TNF
                sub_mult = 0.78 if ctl["subtype"] == "CD4" else 1.0
                F_syn = F_syn_max_nN * sub_mult * ramp * lysis_decay
                A_syn = 14.0 + 74.6 * (F_syn / 1.05) ** 0.78 * np.clip(ramp, 0.25, 1.0)
                arfs = 92.0 * (F_syn / 1.05) ** 1.15 * lysis_decay
                indent = 1.35 * (F_syn / 1.0) * (850.0 / max(E_target_Pa, 200.0)) ** 0.65
                stretch = 1.0 + 0.55 * (F_syn / 1.2) + 0.38 * (1.0 if tgt["caspase3"] > 0.4 else 0.0)
                ctl["Lc_um"] = 13.6 * stretch
                ctl["La_um"] = 8.3 / (stretch ** 0.38)
                ctl["Lb_um"] = 10.5 / (stretch ** 0.22)
                ctl["sph"] = float(np.clip(0.89 - 0.16 * (stretch - 1.0), 0.64, 0.91))
            else:
                F_syn = 0.0
                A_syn = 0.0
                arfs = 0.0
                indent = 0.0
                ctl["Lc_um"] = 13.6
                ctl["La_um"] = 8.3
                ctl["Lb_um"] = 10.5
                ctl["sph"] = 0.89

            ctl["F_syn_nN"] = float(F_syn)
            ctl["A_syn_um2"] = float(A_syn)
            ctl["arfs_nm_s"] = float(arfs)
            ctl["indent_um"] = float(indent)

        # Update both targets' mechanochemistry by summing across all engaged T cells
        for idx, cell in enumerate(targets):
            engaged_ctls = [c for c in ctls if c["in_contact"] == 1 and c["active_target_idx"] == idx]
            total_F = sum(c["F_syn_nN"] for c in engaged_ctls)
            total_A = sum(c["A_syn_um2"] for c in engaged_ctls)
            max_indent = max((c["indent_um"] for c in engaged_ctls), default=0.0)

            cell["indent_um"] = float(max_indent)
            cell["contact_area_um2"] = float(total_A)

            # Local & global cortical membrane tension elevated by multi-synapse pulling
            sigma_mem = sigma_0 + 0.42 * total_F + 0.12 * cell["caspase3"]
            cell["sigma_mem_mN_m"] = float(sigma_mem)

            # Cooperative perforin pore generation summed across all polarized synapses
            pore_gen_total = 0.0
            for c in engaged_ctls:
                if c["mtoc_pol"] > 0.35:
                    Fc = c["F_syn_nN"]
                    sigma_local = sigma_0 + 0.42 * Fc + 0.18 * (total_F - Fc) + 0.12 * cell["caspase3"]
                    tension_boost = np.exp(beta_tension * (sigma_local - sigma_0))
                    hill_force = (Fc ** 2) / (0.45 ** 2 + Fc ** 2)
                    pore_gen_total += 0.042 * c["mtoc_pol"] * tension_boost * hill_force

            # Competing ESCRT membrane repair (Weigelin et al. 2021)
            d_pores = pore_gen_total - k_repair_min * cell["pores"] * (1.0 - 0.85 * cell["caspase3"])
            cell["pores"] = float(max(0.0, cell["pores"] + d_pores * dt_min))

            # Granzyme B accumulation inside target (with intracellular serpin/clearance)
            d_gzmb = 0.085 * cell["pores"] - 0.035 * cell["granzyme"] * (1.0 - 0.7 * cell["caspase3"])
            cell["granzyme"] = float(max(0.0, cell["granzyme"] + d_gzmb * dt_min))

            # Bistable Caspase-3 activation
            gz = cell["granzyme"]
            c3 = cell["caspase3"]
            c3_drive = 0.055 * (gz ** 4) / (0.68 ** 4 + gz ** 4) + 0.065 * (c3 ** 4) / (0.48 ** 4 + c3 ** 4)
            d_c3 = c3_drive * (1.0 - c3) - 0.018 * c3 * (1.0 - c3)
            cell["caspase3"] = float(np.clip(c3 + d_c3 * dt_min, 0.0, 1.0))

            # Nuclear dead-dye uptake (NucRed/TO-PRO-3, threshold = 10.0 a.u.)
            # Sub-lethal pores (P < 0.25) are resealed by ESCRT-III before nuclear dye entry
            supra_pores = max(0.0, cell["pores"] - 0.22)
            dye_influx = 0.08 * supra_pores + 0.85 * (cell["caspase3"] ** 2.2)
            cell["dead_dye"] = float(min(42.5, cell["dead_dye"] + dye_influx * dt_min))
            if cell["dead_dye"] >= 10.0 and cell["caspase3"] >= 0.50 and cell["killed_time_min"] is None:
                cell["killed_time_min"] = float(t)

            # Blebbing & volume collapse
            target_vol_frac = 1.0 - 0.74 * (cell["caspase3"] ** 1.6)
            cell["V_um3"] = float(cell["V0_um3"] * target_vol_frac)
            cell["bleb_amp"] = float(
                np.clip(4.0 * cell["caspase3"] * (1.0 - 0.45 * cell["caspase3"]), 0.0, 1.4)
            )

            if cell["caspase3"] < 0.48:
                cell["n_frag"] = 1
            elif cell["caspase3"] < 0.68:
                cell["n_frag"] = 2
            elif cell["caspase3"] < 0.84:
                cell["n_frag"] = 3
            elif cell["caspase3"] < 0.94:
                cell["n_frag"] = 4
            else:
                cell["n_frag"] = 5

        # Primary/pioneer CTL (index 0) + aggregate multi-CTL observables
        c0 = ctls[0]
        n_engaged_now = sum(1 for c in ctls if c["in_contact"] == 1)
        tot_F_now = sum(c["F_syn_nN"] for c in ctls if c["in_contact"] == 1)
        tot_A_now = sum(c["A_syn_um2"] for c in ctls if c["in_contact"] == 1)
        max_arfs_now = max((c["arfs_nm_s"] for c in ctls), default=0.0)
        max_indent_now = max((c["indent_um"] for c in ctls), default=0.0)
        max_mtoc_now = max((c["mtoc_pol"] for c in ctls), default=c0["mtoc_pol"])
        max_Lc_now = max((c["Lc_um"] for c in ctls), default=c0["Lc_um"])
        min_sph_now = min((c["sph"] for c in ctls), default=c0["sph"])

        mpa_deg = 88.0 - 76.2 * max_mtoc_now if n_engaged_now > 0 else 88.0 - 45.0 * max_mtoc_now
        md_um = 4.20 - 3.52 * max_mtoc_now if n_engaged_now > 0 else 4.20 - 1.80 * max_mtoc_now

        # Compact array of all effector T cells for 3D multi-cell rendering
        ctl_list = []
        for c in ctls:
            ctl_list.append({
                "id": c["id"],
                "sub": c["subtype"],
                "x": round(float(c["pos"][0]), 2),
                "y": round(float(c["pos"][1]), 2),
                "z": round(float(c["pos"][2]), 2),
                "ax": round(float(c["axis"][0]), 3),
                "ay": round(float(c["axis"][1]), 3),
                "az": round(float(c["axis"][2]), 3),
                "La": round(float(c["La_um"]), 2),
                "Lb": round(float(c["Lb_um"]), 2),
                "Lc": round(float(c["Lc_um"]), 2),
                "ic": int(c["in_contact"]),
                "mtoc": round(float(c["mtoc_pol"]), 3),
                "F": round(float(c["F_syn_nN"]), 3),
            })

        records.append({
            "t_min": round(float(t), 1),
            "n_effectors": n_effectors,
            "n_engaged": int(n_engaged_now),
            "active_target": int(c0["active_target_idx"] + 1),
            "in_contact": int(1 if n_engaged_now > 0 else 0),
            "ctl_x": round(float(c0["pos"][0]), 2),
            "ctl_y": round(float(c0["pos"][1]), 2),
            "ctl_z": round(float(c0["pos"][2]), 2),
            "ctl_ax": round(float(c0["axis"][0]), 3),
            "ctl_ay": round(float(c0["axis"][1]), 3),
            "ctl_az": round(float(c0["axis"][2]), 3),
            "ctl_La_um": round(float(c0["La_um"]), 2),
            "ctl_Lb_um": round(float(c0["Lb_um"]), 2),
            "ctl_Lc_um": round(float(max_Lc_now), 2),
            "ctl_sphericity": round(float(min_sph_now), 3),
            "ctl_speed_um_min": round(float(c0["speed_um_min"]), 2),
            "mtoc_polarization": round(float(max_mtoc_now), 3),
            "mpa_deg": round(float(mpa_deg), 2),
            "md_um": round(float(md_um), 3),
            "F_syn_nN": round(float(tot_F_now), 3),
            "contact_area_um2": round(float(tot_A_now), 2),
            "arfs_nm_s": round(float(max_arfs_now), 2),
            "indent_um": round(float(max_indent_now), 3),
            "ctls": ctl_list,
            # Target 1
            "t1_V_um3": round(float(targets[0]["V_um3"]), 1),
            "t1_vol_frac": round(float(targets[0]["V_um3"] / targets[0]["V0_um3"]), 3),
            "t1_sigma_mN_m": round(float(targets[0]["sigma_mem_mN_m"]), 3),
            "t1_pores": round(float(targets[0]["pores"]), 3),
            "t1_caspase3": round(float(targets[0]["caspase3"]), 3),
            "t1_dead_dye": round(float(targets[0]["dead_dye"]), 2),
            "t1_bleb": round(float(targets[0]["bleb_amp"]), 3),
            "t1_n_frag": int(targets[0]["n_frag"]),
            # Target 2
            "t2_V_um3": round(float(targets[1]["V_um3"]), 1),
            "t2_vol_frac": round(float(targets[1]["V_um3"] / targets[1]["V0_um3"]), 3),
            "t2_sigma_mN_m": round(float(targets[1]["sigma_mem_mN_m"]), 3),
            "t2_pores": round(float(targets[1]["pores"]), 3),
            "t2_caspase3": round(float(targets[1]["caspase3"]), 3),
            "t2_dead_dye": round(float(targets[1]["dead_dye"]), 2),
            "t2_bleb": round(float(targets[1]["bleb_amp"]), 3),
            "t2_n_frag": int(targets[1]["n_frag"]),
        })

    return {
        "preset_id": preset_id,
        "parameters": {
            "F_syn_max_nN": F_syn_max_nN,
            "E_target_Pa": E_target_Pa,
            "beta_tension": beta_tension,
            "k_repair_min": k_repair_min,
            "dwell_cap_min": dwell_cap_min,
            "serial_kill": serial_kill,
            "n_effectors": n_effectors,
        },
        "summary": {
            "n_effectors": n_effectors,
            "t1_killed_time_min": targets[0]["killed_time_min"],
            "t2_killed_time_min": targets[1]["killed_time_min"],
            "total_kills": int((targets[0]["killed_time_min"] is not None) + (targets[1]["killed_time_min"] is not None)),
            "max_ctl_length_um": round(max(r["ctl_Lc_um"] for r in records), 2),
            "min_ctl_sphericity": round(min(r["ctl_sphericity"] for r in records), 3),
            "max_F_syn_nN": round(max(r["F_syn_nN"] for r in records), 3),
            "max_contact_area_um2": round(max(r["contact_area_um2"] for r in records), 2),
            "max_arfs_nm_s": round(max(r["arfs_nm_s"] for r in records), 2),
            "min_mpa_deg": round(min(r["mpa_deg"] for r in records), 2),
            "min_md_um": round(min(r["md_um"] for r in records), 3),
        },
        "frames": records,
    }


def main():
    presets = {
        "cooperative_swarm_4ctl": simulate_preset(
            "cooperative_swarm_4ctl",
            F_syn_max_nN=0.48,
            E_target_Pa=1350.0,
            beta_tension=2.10,
            k_repair_min=0.185,
            dwell_cap_min=55.0,
            serial_kill=True,
            n_effectors=4,
        ),
        "single_ctl_resistant_fail": simulate_preset(
            "single_ctl_resistant_fail",
            F_syn_max_nN=0.48,
            E_target_Pa=1350.0,
            beta_tension=2.10,
            k_repair_min=0.185,
            dwell_cap_min=55.0,
            serial_kill=False,
            n_effectors=1,
        ),
        "super_engager_serial": simulate_preset(
            "super_engager_serial",
            F_syn_max_nN=1.25,
            E_target_Pa=850.0,
            beta_tension=2.40,
            k_repair_min=0.085,
            dwell_cap_min=95.0,
            serial_kill=True,
            n_effectors=1,
        ),
        "single_conjugate_13T": simulate_preset(
            "single_conjugate_13T",
            F_syn_max_nN=1.05,
            E_target_Pa=850.0,
            beta_tension=2.30,
            k_repair_min=0.085,
            dwell_cap_min=110.0,
            serial_kill=False,
            n_effectors=1,
        ),
        "transient_subthreshold": simulate_preset(
            "transient_subthreshold",
            F_syn_max_nN=0.30,
            E_target_Pa=850.0,
            beta_tension=2.40,
            k_repair_min=0.110,
            dwell_cap_min=18.0,
            serial_kill=False,
            n_effectors=1,
        ),
        "dasatinib_50nM": simulate_preset(
            "dasatinib_50nM",
            F_syn_max_nN=0.14,
            E_target_Pa=850.0,
            beta_tension=1.50,
            k_repair_min=0.130,
            dwell_cap_min=45.0,
            serial_kill=False,
            n_effectors=1,
        ),
    }

    with open(os.path.join(DATA_DIR, "sim_presets_4d.json"), "w") as fp:
        json.dump(presets, fp, separators=(",", ":"))
    print("Saved data/sim_presets_4d.json")
    for k, v in presets.items():
        print(f"  Preset {k}: {v['summary']}")


if __name__ == "__main__":
    main()
