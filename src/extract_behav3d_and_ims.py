#!/usr/bin/env python3
"""
Extract metadata, 3D surface morphology, event-level calibration distributions,
and 4D exemplary trajectories from:
1. Zenodo 8328441 Imaris HDF5 container (LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims)
2. BEHAV3D exported track & organoid statistics (AlievaRios/BEHAV3D)

Follows project conventions: raw data never modified; every output table paired
with a data dictionary in data/data_dictionary.md.
"""

import glob
import io
import json
import os
import h5py
import numpy as np
import pandas as pd
import pyreadr

from paths import REPO_ROOT, DATA_DIR as _DATA_DIR_PATH

ROOT = str(REPO_ROOT)
DATA_DIR = str(_DATA_DIR_PATH)
os.makedirs(DATA_DIR, exist_ok=True)


class PartialH5File(io.RawIOBase):
    """Read-only wrapper allowing h5py to inspect an in-progress .ims download."""

    def __init__(self, path: str, fake_size: int):
        self.f = open(path, "rb")
        self.real_size = os.path.getsize(path)
        self.fake_size = fake_size
        self.pos = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        elif whence == 2:
            self.pos = self.fake_size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b: bytearray) -> int:
        if self.pos >= self.real_size:
            raise OSError(f"Offset {self.pos} exceeds downloaded bytes {self.real_size}")
        self.f.seek(self.pos)
        data = self.f.read(len(b))
        n = len(data)
        b[:n] = data
        self.pos += n
        return n


def find_bc_ims_path() -> tuple[str, int]:
    expected_size = 15124774705
    direct = glob.glob(
        os.path.join(ROOT, "**/*TEG_BC_organoids*.ims"), recursive=True
    ) + glob.glob(str(REPO_ROOT / "Downloads/*TEG_BC_organoids*.ims"))
    if direct:
        return direct[0], expected_size
    # Check .crdownload files by reading header Name
    for p in sorted(glob.glob(str(REPO_ROOT / "Downloads/*.crdownload"))):
        with open(p, "rb") as f:
            head = f.read(512 * 1024)
        if b"2019-06-26 CD4CD8 13T" in head:
            return p, expected_size
    raise FileNotFoundError("Could not locate 2019-06-26 BC organoid .ims file")


def decode_attr(val):
    if hasattr(val, "tobytes"):
        return val.tobytes().decode("utf-8", errors="ignore").strip("\x00")
    return str(val)


def main():
    ims_path, expected_size = find_bc_ims_path()
    print(f"Reading .ims container: {ims_path} ({os.path.getsize(ims_path)/1e9:.2f} GB on disk)")
    pf = PartialH5File(ims_path, expected_size)

    with h5py.File(pf, "r") as f:
        img_attrs = {k: decode_attr(v) for k, v in f["DataSetInfo"]["Image"].attrs.items()}
        time_attrs = {}
        for k, v in f["DataSetInfo"]["TimeInfo"].attrs.items():
            if k in ("DatasetTimePoints", "FileTimePoints", "TimePoint1", "TimePoint2", "TimePoint3", "TimePoint669"):
                time_attrs[k] = decode_attr(v)

        channels = {}
        for c_idx in range(9):
            ck = f"Channel {c_idx}"
            if ck in f["DataSetInfo"]:
                try:
                    c_attrs = {k: decode_attr(v) for k, v in f["DataSetInfo"][ck].attrs.items()}
                    channels[ck] = {
                        "Name": c_attrs.get("Name", ""),
                        "Description": c_attrs.get("Description", ""),
                        "Color": c_attrs.get("Color", ""),
                        "ColorRange": c_attrs.get("ColorRange", ""),
                        "LSMExcitationWavelength": c_attrs.get("LSMExcitationWavelength", ""),
                        "LSMEmissionWavelength": c_attrs.get("LSMEmissionWavelength", ""),
                    }
                except Exception as e:
                    channels[ck] = {"error": str(e)}

        # Extract MegaSurfaces0 (cd4 blue) 3D morphology
        ms0 = f["Scene8"]["Content"]["MegaSurfaces0"]
        sm = ms0["SurfaceModel"][:]
        smi = ms0["SurfaceModelInfo"][:]
        tr0 = ms0["Track0"][:]

    ext_x_um = (float(img_attrs["ExtMax0"]) - float(img_attrs["ExtMin0"])) * 1000.0
    ext_y_um = (float(img_attrs["ExtMax1"]) - float(img_attrs["ExtMin1"])) * 1000.0
    ext_z_um = (float(img_attrs["ExtMax2"]) - float(img_attrs["ExtMin2"])) * 1000.0
    nx, ny, nz, nt = int(img_attrs["X"]), int(img_attrs["Y"]), int(img_attrs["Z"]), int(img_attrs["T"])

    ims_meta = {
        "file_name": "LSM880_example_dataset_TEG_BC_organoids_2019_06_26.ims",
        "zenodo_record": "8328441",
        "dataset_name": img_attrs.get("Name", ""),
        "microscope": "Zeiss LSM880 SpectralImaging, Plan-Apochromat 20x/0.8 NA",
        "dimensions_voxels": {"X": nx, "Y": ny, "Z": nz, "T": nt, "C": len(channels)},
        "physical_extent_um": {"X": round(ext_x_um, 3), "Y": round(ext_y_um, 3), "Z": round(ext_z_um, 3)},
        "voxel_pitch_um": {
            "dx": round(ext_x_um / nx, 4),
            "dy": round(ext_y_um / ny, 4),
            "dz": round(ext_z_um / nz, 4),
        },
        "frame_interval_s": 119.71,
        "frame_interval_min": 1.995,
        "total_duration_h": round((nt - 1) * 119.71 / 3600.0, 2),
        "time_info": time_attrs,
        "channels": channels,
        "scene8_megasurfaces0": {
            "name": "cd4 blue",
            "n_surface_instances": int(len(smi)),
            "n_tracks": int(len(tr0)),
            "fields": list(smi.dtype.names),
        },
    }

    with open(os.path.join(DATA_DIR, "ims_metadata_20190626.json"), "w") as fp:
        json.dump(ims_meta, fp, indent=2)
    print("Saved data/ims_metadata_20190626.json")

    # Build DataFrame of 149,129 3D surface instances from MegaSurfaces0
    df_ims = pd.DataFrame({
        "ID": sm["ID"],
        "Time_ims": sm["ID_Time"] + 1,
        "Volume_um3": smi["Volume"] * 1e9,
        "Area_um2": smi["Area"] * 1e6,
        "Sphericity": smi["Sphericity"],
        "AxisA_um": smi["EllipsoidAxisLengthX"] * 1e3 * 2.0,
        "AxisB_um": smi["EllipsoidAxisLengthY"] * 1e3 * 2.0,
        "AxisC_um": smi["EllipsoidAxisLengthZ"] * 1e3 * 2.0,
        "AspectRatio_CA": smi["EllipsoidAxisLengthZ"] / np.maximum(smi["EllipsoidAxisLengthX"], 1e-6),
        "EllipticityProlate": smi["EllipticityProlate"],
        "EllipticityOblate": smi["EllipticityOblate"],
        "AxisCX": smi["EllipsoidAxisCX"],
        "AxisCY": smi["EllipsoidAxisCY"],
        "AxisCZ": smi["EllipsoidAxisCZ"],
        "AxisAX": smi["EllipsoidAxisAX"],
        "AxisAY": smi["EllipsoidAxisAY"],
        "AxisAZ": smi["EllipsoidAxisAZ"],
        "NumberOfTriangles": smi["NumberOfTriangles"],
    })

    # Join with DemoDataset3_20190626_CD4blue_Statistics
    base_t = os.path.join(
        ROOT, "external/AlievaRios_BEHAV3D/demos/tcell_demo/data/DemoDataset3_20190626_CD4blue_Statistics"
    )
    pos = pd.read_csv(f"{base_t}/DemoDataset3_20190626_CD4blue_Position.csv", skiprows=3)
    spd = pd.read_csv(f"{base_t}/DemoDataset3_20190626_CD4blue_Speed.csv", skiprows=3)
    dist = pd.read_csv(f"{base_t}/DemoDataset3_20190626_CD4blue_Intensity_Min_Ch=9_Img=1.csv", skiprows=3)
    dye = pd.read_csv(f"{base_t}/DemoDataset3_20190626_CD4blue_Intensity_Mean_Ch=3_Img=1.csv", skiprows=3)

    df_t = (
        pos[["Time", "TrackID", "ID", "Position X", "Position Y", "Position Z"]]
        .merge(spd[["ID", "Speed"]], on="ID")
        .merge(dist[["ID", "Intensity Min"]].rename(columns={"Intensity Min": "dist_org_um"}), on="ID")
        .merge(dye[["ID", "Intensity Mean"]].rename(columns={"Intensity Mean": "dead_dye"}), on="ID")
        .merge(df_ims, on="ID")
    )
    df_t["in_contact"] = (df_t["dist_org_um"] <= 0.0015).astype(int)
    df_t["Speed_um_min"] = df_t["Speed"] * 60.0

    # Join BEHAV3D cluster assignments from classified_tcell_track_data_summary.rds
    rds_sum_path = os.path.join(
        ROOT,
        "external/AlievaRios_BEHAV3D/demos/tcell_demo/example_output/tcell_behavior/results/classified_tcell_track_data_summary.rds",
    )
    df_sum = list(pyreadr.read_r(rds_sum_path).values())[0]
    df_sum_e3 = df_sum[(df_sum["exp_nr"] == 3) & (df_sum["tcell_line"] == "CD4_TEG")].copy()
    df_sum_e3["RawTrackID"] = df_sum_e3["TrackID"].apply(lambda s: int(str(s).split("_")[1]))
    df_t = df_t.merge(df_sum_e3[["RawTrackID", "cluster2"]], left_on="TrackID", right_on="RawTrackID", how="left")
    df_t["cluster"] = pd.to_numeric(df_t["cluster2"].astype(str), errors="coerce").fillna(0).astype(int)

    # Event-level per-track summary
    def count_contact_bouts(s):
        arr = s.values
        if len(arr) == 0:
            return 0
        diffs = np.diff(np.concatenate(([0], arr, [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        # Bouts lasting at least 2 frames (4 min)
        valid = (ends - starts) >= 2
        return int(np.sum(valid))

    def mean_bout_duration_min(s):
        arr = s.values
        diffs = np.diff(np.concatenate(([0], arr, [0])))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]
        durs = (ends - starts) * 2.0
        durs = durs[durs >= 4.0]
        return float(np.mean(durs)) if len(durs) > 0 else 0.0

    track_events = []
    for tid, grp in df_t.groupby("TrackID"):
        grp = grp.sort_values("Time")
        c_bouts = count_contact_bouts(grp["in_contact"])
        mean_bout_min = mean_bout_duration_min(grp["in_contact"])
        total_contact_min = float(grp["in_contact"].sum() * 2.0)
        dur_min = float(len(grp) * 2.0)
        l_min = float(grp["AxisC_um"].min())
        l_max = float(grp["AxisC_um"].max())
        l_q05 = float(np.percentile(grp["AxisC_um"], 5))
        l_q95 = float(np.percentile(grp["AxisC_um"], 95))
        track_events.append({
            "TrackID": int(tid),
            "cluster": int(grp["cluster"].iloc[0]),
            "track_duration_min": round(dur_min, 1),
            "total_contact_min": round(total_contact_min, 1),
            "contact_fraction": round(total_contact_min / max(dur_min, 1.0), 4),
            "n_contact_bouts": c_bouts,
            "mean_bout_duration_min": round(mean_bout_min, 2),
            "mean_speed_um_min": round(float(grp["Speed_um_min"].mean()), 3),
            "mean_volume_um3": round(float(grp["Volume_um3"].mean()), 1),
            "mean_area_um2": round(float(grp["Area_um2"].mean()), 1),
            "mean_sphericity": round(float(grp["Sphericity"].mean()), 4),
            "min_sphericity": round(float(grp["Sphericity"].min()), 4),
            "min_length_um": round(l_min, 2),
            "max_length_um": round(l_max, 2),
            "length_fold_change_max_min": round(l_max / max(l_min, 1e-3), 3),
            "length_fold_change_q95_q05": round(l_q95 / max(l_q05, 1e-3), 3),
            "max_aspect_ratio_CA": round(float(grp["AspectRatio_CA"].max()), 3),
            "mean_ellipticity_prolate": round(float(grp["EllipticityProlate"].mean()), 4),
        })

    df_events = pd.DataFrame(track_events)
    df_events.to_csv(os.path.join(DATA_DIR, "tcell_event_calibration.csv"), index=False)
    print(f"Saved data/tcell_event_calibration.csv ({len(df_events)} tracks)")

    # Cluster-level summary table
    df_cl_valid = df_events[df_events["cluster"] > 0]
    cl_summary = (
        df_cl_valid.groupby("cluster")
        .agg(
            n_tracks=("TrackID", "count"),
            contact_fraction=("contact_fraction", "mean"),
            mean_bout_duration_min=("mean_bout_duration_min", "mean"),
            mean_speed_um_min=("mean_speed_um_min", "mean"),
            mean_volume_um3=("mean_volume_um3", "mean"),
            mean_area_um2=("mean_area_um2", "mean"),
            mean_sphericity=("mean_sphericity", "mean"),
            min_sphericity=("min_sphericity", "mean"),
            min_length_um=("min_length_um", "mean"),
            max_length_um=("max_length_um", "mean"),
            length_fold_change_max_min=("length_fold_change_max_min", "mean"),
            length_fold_change_q95_q05=("length_fold_change_q95_q05", "mean"),
            max_aspect_ratio_CA=("max_aspect_ratio_CA", "mean"),
        )
        .round(3)
        .reset_index()
    )
    cl_summary.to_csv(os.path.join(DATA_DIR, "tcell_3d_morphology_by_cluster.csv"), index=False)
    print("Saved data/tcell_3d_morphology_by_cluster.csv")

    # Organoid death & fragmentation time series from DemoDataset_8_organoids
    base_o = os.path.join(
        ROOT, "external/AlievaRios_BEHAV3D/demos/organoid_demo/data/DemoDataset_8_organoids_Statistics"
    )
    o_pos = pd.read_csv(f"{base_o}/DemoDataset_8_organoids_Position.csv", skiprows=3)
    o_vol = pd.read_csv(f"{base_o}/DemoDataset_8_organoids_Volume.csv", skiprows=3)
    o_area = pd.read_csv(f"{base_o}/DemoDataset_8_organoids_Area.csv", skiprows=3)
    o_dye = pd.read_csv(f"{base_o}/DemoDataset_8_organoids_Intensity_Mean_Ch=3_Img=1.csv", skiprows=3)
    o_dist = pd.read_csv(f"{base_o}/DemoDataset_8_organoids_Intensity_Min_Ch=7_Img=1.csv", skiprows=3)

    df_o_frag = (
        o_pos[["Position X", "Position Y", "Position Z", "Time", "TrackID", "ID"]]
        .merge(o_vol[["Volume", "ID"]], on="ID")
        .merge(o_area[["Area", "ID"]], on="ID")
        .merge(o_dye[["Intensity Mean", "ID"]].rename(columns={"Intensity Mean": "dead_dye"}), on="ID")
        .merge(o_dist[["Intensity Min", "ID"]].rename(columns={"Intensity Min": "tcell_dist_um"}), on="ID")
    )

    # Aggregate per TrackID and Time
    org_ts = []
    for (tid, t_idx), grp in df_o_frag.groupby(["TrackID", "Time"]):
        tot_v = float(grp["Volume"].sum())
        tot_a = float(grp["Area"].sum())
        # Volume-weighted dead dye mean
        dd_mean = float(np.sum(grp["Volume"] * grp["dead_dye"]) / max(tot_v, 1e-3))
        min_d = float(grp["tcell_dist_um"].min())
        # Volume-weighted centroid
        cx = float(np.sum(grp["Volume"] * grp["Position X"]) / max(tot_v, 1e-3))
        cy = float(np.sum(grp["Volume"] * grp["Position Y"]) / max(tot_v, 1e-3))
        cz = float(np.sum(grp["Volume"] * grp["Position Z"]) / max(tot_v, 1e-3))
        org_ts.append({
            "TrackID": int(tid),
            "Time_frame": int(t_idx),
            "Time_min": round((int(t_idx) - 1) * 2.0, 1),
            "n_fragments": int(len(grp)),
            "total_volume_um3": round(tot_v, 1),
            "total_area_um2": round(tot_a, 1),
            "dead_dye_mean": round(dd_mean, 3),
            "min_tcell_dist_um": round(min_d, 3),
            "centroid_x_um": round(cx, 2),
            "centroid_y_um": round(cy, 2),
            "centroid_z_um": round(cz, 2),
            "is_dead": int(dd_mean >= 10.0),
        })
    df_org_ts = pd.DataFrame(org_ts).sort_values(["TrackID", "Time_frame"])
    # Compute volume_fraction_remaining relative to initial 5-frame mean
    init_vols = (
        df_org_ts[df_org_ts["Time_frame"] <= 5]
        .groupby("TrackID")["total_volume_um3"]
        .mean()
        .to_dict()
    )
    df_org_ts["volume_fraction"] = (
        df_org_ts["total_volume_um3"] / df_org_ts["TrackID"].map(init_vols)
    ).round(4)
    df_org_ts.to_csv(os.path.join(DATA_DIR, "organoid_kill_and_fragmentation.csv"), index=False)
    print(f"Saved data/organoid_kill_and_fragmentation.csv ({len(df_org_ts)} rows)")

    # Extract exemplar 4D tracks (Super-engager T cell tracks + fragmenting organoid tracks)
    # Pick top Cluster 9 tracks with strong contact and elongation
    top_c9 = (
        df_events[(df_events["cluster"] == 9) & (df_events["n_contact_bouts"] >= 1)]
        .sort_values("length_fold_change_max_min", ascending=False)
        .head(5)["TrackID"]
        .tolist()
    )
    exemplar_tcells = {}
    for tid in top_c9:
        sub = df_t[df_t["TrackID"] == tid].sort_values("Time")
        # Subsample every 2 frames (4 min) up to 120 frames (240 min) for compact JSON
        sub = sub.iloc[::2].head(90)
        p0 = sub[["Position X", "Position Y", "Position Z"]].iloc[0].values
        exemplar_tcells[str(tid)] = {
            "TrackID": int(tid),
            "cluster": 9,
            "time_min": [round(float((t - sub["Time"].iloc[0]) * 2.0), 1) for t in sub["Time"]],
            "rel_x_um": [round(float(x - p0[0]), 2) for x in sub["Position X"]],
            "rel_y_um": [round(float(y - p0[1]), 2) for y in sub["Position Y"]],
            "rel_z_um": [round(float(z - p0[2]), 2) for z in sub["Position Z"]],
            "axis_a_um": [round(float(a), 2) for a in sub["AxisA_um"]],
            "axis_b_um": [round(float(b), 2) for b in sub["AxisB_um"]],
            "axis_c_um": [round(float(c), 2) for c in sub["AxisC_um"]],
            "sphericity": [round(float(s), 3) for s in sub["Sphericity"]],
            "speed_um_min": [round(float(v), 2) for v in sub["Speed_um_min"]],
            "in_contact": [int(c) for c in sub["in_contact"]],
            "axis_cx": [round(float(cx), 3) for cx in sub["AxisCX"]],
            "axis_cy": [round(float(cy), 3) for cy in sub["AxisCY"]],
            "axis_cz": [round(float(cz), 3) for cz in sub["AxisCZ"]],
        }

    # Exemplar fragmenting targets from DemoDataset_8_organoids
    exemplar_targets = {}
    for tid in [1000245316, 1000244591, 1000209032, 1000224278]:
        sub_ts = df_org_ts[df_org_ts["TrackID"] == tid].sort_values("Time_frame")
        sub_frag = df_o_frag[df_o_frag["TrackID"] == tid].sort_values(["Time", "ID"])
        c0 = [
            float(sub_ts["centroid_x_um"].iloc[0]),
            float(sub_ts["centroid_y_um"].iloc[0]),
            float(sub_ts["centroid_z_um"].iloc[0]),
        ]
        frames_list = []
        for t_frame in sub_ts["Time_frame"].iloc[::4]:  # every 8 min
            f_rows = sub_frag[sub_frag["Time"] == t_frame]
            ts_row = sub_ts[sub_ts["Time_frame"] == t_frame].iloc[0]
            frags = []
            for _, r in f_rows.iterrows():
                rad = float(((3.0 * r["Volume"]) / (4.0 * np.pi)) ** (1.0 / 3.0))
                frags.append({
                    "id": int(r["ID"]),
                    "x": round(float(r["Position X"] - c0[0]), 2),
                    "y": round(float(r["Position Y"] - c0[1]), 2),
                    "z": round(float(r["Position Z"] - c0[2]), 2),
                    "r": round(rad, 2),
                    "vol": round(float(r["Volume"]), 1),
                    "dye": round(float(r["dead_dye"]), 2),
                })
            frames_list.append({
                "t_min": float(ts_row["Time_min"]),
                "n_frag": int(ts_row["n_fragments"]),
                "vol_frac": float(ts_row["volume_fraction"]),
                "dead_dye": float(ts_row["dead_dye_mean"]),
                "frags": frags,
            })
        exemplar_targets[str(tid)] = {
            "TrackID": int(tid),
            "initial_volume_um3": float(sub_ts["total_volume_um3"].iloc[0]),
            "final_volume_um3": float(sub_ts["total_volume_um3"].iloc[-1]),
            "max_fragments": int(sub_ts["n_fragments"].max()),
            "max_dead_dye": float(sub_ts["dead_dye_mean"].max()),
            "frames": frames_list,
        }

    with open(os.path.join(DATA_DIR, "real_exemplar_4d.json"), "w") as fp:
        json.dump(
            {
                "ims_metadata": ims_meta,
                "cluster_summary": cl_summary.to_dict(orient="records"),
                "exemplar_tcells": exemplar_tcells,
                "exemplar_targets": exemplar_targets,
            },
            fp,
            indent=2,
        )
    print("Saved data/real_exemplar_4d.json")


if __name__ == "__main__":
    main()
