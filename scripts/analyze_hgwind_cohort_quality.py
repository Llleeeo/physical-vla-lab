#!/usr/bin/env python3
"""CPU-only descriptive quality screen of the rendered HGwind development cohort."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from capstone_vla.hf_archive import sha256
from capstone_vla.hgwind_cohort import CAMERAS, require, verify_files
from capstone_vla.trajectory_quality import motion_metrics, contact_metrics, segmentation_metrics, describe_groups

METRICS = {
    "tail_relative_center_jitter_p95_mm": "Final 0.5 s center jitter, p95 (mm)",
    "lift_to_final_relative_center_shift_mm": "LIFT to final center displacement (mm)",
    "minimum_tail_lift_from_first_bilateral_mm": "Minimum final-window lift (mm)",
    "tail_relative_speed_p95_mps": "Final relative speed, p95 (m/s)",
    "raw_longest_bilateral_duration_s": "Longest raw bilateral interval (s)",
    "precontact_ee_acceleration_p95_mps2": "Precontact EE acceleration, p95 (m/s2)",
}


def positive_contact_masks(f, target):
    c = f["evidence/contacts"]
    paths = c["paths"].asstr()[:].tolist()
    target_ids = {i for i, p in enumerate(paths) if p == target or p.startswith(target + "/")}
    require(bool(target_ids), "Target absent from raw contact table")
    masks = []
    headers, points = c["headers"][:], c["points"][:]
    for finger in ("panda_leftfinger", "panda_rightfinger"):
        ids = {i for i, p in enumerate(paths) if "/" + finger in p}
        require(bool(ids), "Finger absent from raw contact table")
        mask = np.zeros(len(c["step_time"]), bool)
        for h in headers:
            if h[7] <= 0 or not ((h[1] in target_ids and h[2] in ids) or (h[2] in target_ids and h[1] in ids)):
                continue
            values = points[int(h[6]):int(h[6] + h[7]), 6:9]
            if np.any(np.linalg.norm(values, axis=1) > 1e-12):
                mask[int(h[0])] = True
        masks.append(mask)
    return c["step_time"][:], c["step_dt"][:], *masks


def telemetry(ep, n):
    values = {}
    for line in (ep / "analyse.jsonl").open():
        row = json.loads(line)
        if row["type"] == "control_telemetry":
            values[row["data"]["name"]] = np.asarray(row["data"]["values"])
    for key in ("skill_phase", "close_request", "attach_grasp_active"):
        require(key in values and len(values[key]) == n, "Missing control field: " + key)
    return values


def assess(row, out):
    source, staged = Path(row["source"]), Path(row["staged"])
    verify_files(source, row["files"])
    meta = json.loads((source / "meta.json").read_text())
    require(meta["success"] and meta["runtime_safety"]["passed"], "Unexpected source outcome")
    n = row["frames"]
    control = telemetry(source, n)
    require(not control["attach_grasp_active"].any() and not meta["fix_joint"] and not meta["fixedforce"], "Assistance found")
    names = meta["skill_schedule"]["phase_names"]
    phase = control["skill_phase"].reshape(n, -1)[:, 0]
    with h5py.File(source / "replay.hdf5", "r") as f:
        time = f["timestamp"][:]
        oi = meta["object_paths"].index(meta["policy"]["roles"]["A"])
        raw = contact_metrics(*positive_contact_masks(f, meta["policy"]["roles"]["A"]), control_time=time)
        onset = raw["raw_first_bilateral_time_s"]
        first = min(n-1, int(np.searchsorted(time, onset))) if onset is not None else None
        measured = motion_metrics(time, f["obs/ee_pos"][:], f["obs/ee_quat"][:],
                                  f["obs/object_pos"][:, oi], f["obs/ee_vel"][:], phase,
                                  control["close_request"], meta["validation"]["pinch_frame"]["offset_local_m"],
                                  first, names.index("lift"), names.index("hold"), ball_velocity=f["obs/object_vel"][:, oi],
                                  first_contact=min(n-1, int(np.searchsorted(time, raw["raw_first_any_contact_time_s"]))))
    events = {"pre_contact": max(0, int(np.searchsorted(time, raw["raw_first_any_contact_time_s"])) - 3),
              "bilateral": measured["first_bilateral_state_frame"], "lift": measured["first_lift_frame"],
              "hold": measured["first_hold_frame"], "final": n-1}
    require(all(i is not None for i in events.values()), "Missing event in accepted source")
    samples = sorted(set(events.values()) | set(range(max(0, n-30), n, 6)))
    target_visibility = {camera: [] for camera in CAMERAS}
    sheet = Image.new("RGB", (1410, 1190), "#15191e")
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 8), f"{row['host']}/{row['episode']} | green = visible target segmentation | event snapshots only", fill="white")
    with h5py.File(staged / "video/rendered.hdf5", "r") as f:
        require(np.array_equal(f["source_frame_index"][:], np.arange(n)) and np.allclose(f["timestamp_s"][:], time, atol=1e-7, rtol=0), "Image alignment changed")
        contract = json.loads(f.attrs["segmentation_contract_json"])
        ids = [a["id"] for a in contract["assignments"] if a["prim_path"] == meta["policy"]["roles"]["A"]]
        require(len(ids) == 1 and int(f.attrs["fps"]) == 60, "Missing target segmentation contract")
        for y, camera in enumerate(CAMERAS):
            masks = {i: segmentation_metrics(f[camera + "_seg"][i], ids[0]) for i in samples}
            target_visibility[camera] = [dict(frame=i, **masks[i]) for i in samples]
            draw.text((8, 70 + y*220), camera, fill="white")
            for x, (event, i) in enumerate(events.items()):
                image = Image.fromarray(f[camera + "_rgb"][i])
                if masks[i]["bbox"]:
                    ImageDraw.Draw(image).rectangle(masks[i]["bbox"], outline="#49ff73", width=2)
                image = image.resize((256, 192))
                sheet.paste(image, (130+x*256, 70+y*220))
                draw.text((130+x*256, 45+y*220), f"{event} f{i} {time[i]:.2f}s", fill="white")
    sheet_path = out / "contact_sheets" / (row["host"] + "_" + row["episode"] + ".jpg")
    sheet.save(sheet_path, quality=90)
    joint_missing = meta.get("skill_telemetry_contract", {}).get("missing_attributes", [])
    flags = []
    if raw["raw_tail_bilateral_fraction"] < 1-1e-9: flags.append("not_simultaneous_bilateral_on_every_physics_tick")
    if measured["lift_entries"] > 1: flags.append("multiple_lift_entries")
    if measured["reopen_transitions_after_first_bilateral"]: flags.append("reopening_after_contact")
    sampled_zero_both = [i for i in samples if all(next(x for x in target_visibility[c] if x["frame"] == i)["pixels"] == 0 for c in ("opst_cam", "wrist_cam"))]
    if sampled_zero_both: flags.append("target_absent_in_both_policy_views_at_sample")
    result = dict(host=row["host"], episode=row["episode"], episode_seed=meta["episode_seed"],
                  base_seed=meta["base_seed"], depth_bias_m=row["depth_bias_m"], **measured, **raw,
                  event_frames=events, sampled_visibility=target_visibility, missing_telemetry_attributes=joint_missing,
                  sampled_zero_target_both_policy_views=sampled_zero_both, review_flags=flags,
                  contact_sheet=str(sheet_path), video_directory=str(staged / "video"),
                  source_sha256={x["path"]: x["sha256"] for x in row["files"]}, human_review_status="pending")
    verify_files(source, row["files"])
    return result


def plots(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    rng = np.random.default_rng(20261028)
    for ax, (key, label) in zip(axes.flat, METRICS.items()):
        for position, (host, color) in enumerate((("local", "#0072B2"), ("hongtao", "#D55E00"))):
            values = np.asarray([r[key] for r in rows if r["host"] == host])
            ax.scatter(position+rng.uniform(-.12,.12,len(values)), values, color=color, alpha=.8, s=25)
            ax.hlines(np.median(values),position-.22,position+.22,color="black",linewidth=2)
        ax.set_xticks([0,1], ["local (21)", "hongtao (19)"]);ax.set_title(label,fontsize=10);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Selected successful demonstrations: descriptive group comparison\nDifferent seeds and depth settings; no causal conclusion or single quality score")
    fig.savefig(out / "quality_comparison.png",dpi=140);plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/derived/hgwind_40_camera_v2_20261028/render_manifest.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/derived/hgwind_quality_20261028")
    args = parser.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True);(out/"contact_sheets").mkdir(exist_ok=True)
    manifest=json.loads(args.manifest.read_text());rows=[]
    for i,row in enumerate(manifest["rows"]):
        rows.append(assess(row,out));print(f"QUALITY {i+1}/{len(manifest['rows'])} {row['host']}/{row['episode']}",flush=True)
    comparison={key:describe_groups(rows,key) for key in METRICS}
    counts={host:sum(r["host"]==host for r in rows) for host in ("local","hongtao")}
    report={"schema":"physical-vla-lab.trajectory-quality.v1","scope":"Exposed development cohort; descriptive selected-success comparison, no policy training or success relabeling.",
        "manifest_sha256":sha256(args.manifest),"episodes_by_host":counts,"rows":rows,"group_comparison":comparison,
        "contact_definition":"Positive target-finger impulse on each recorded physics tick; simultaneous 180 Hz bilateral contact is distinguished from both sides contacting within a 60 Hz control bin. Neither alone certifies stable grasp.",
        "position_definition":"Ball center relative to the hand, with quaternion rotation compensation and recorded pinch offset; not material-point tangential slip or ball-spin arrest.",
        "observation_anchor":"First control-state timestamp at/after first raw bilateral contact; physics contacts are end-of-substep, observations pre-physics.",
        "visibility_definition":"Five event frames plus sparse final-window segmentation samples; zero mask can mean occlusion or outside view. Not full-video certification.",
        "inference_limits":["Local and remote have distinct base seeds and 2 mm vs 0 mm depth bias; selected outcomes cannot identify the bias's causal effect.","Bootstrap intervals are exploratory descriptions across episodes, not multiplicity-adjusted hypothesis tests.","No automatic exclusions, split assignment or sampling-weight changes."],
        "recommendation":"Retain both groups; establish uniform episode sampling first. Defer reweighting until visual review and fixed-budget closed-loop comparison.","human_review_complete":False}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    with (out/"metrics.csv").open("w",newline="") as f:
        fields=["host","episode","episode_seed","depth_bias_m",*METRICS,"reopen_transitions_after_first_bilateral","lift_entries"]
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(rows)
    plots(rows,out)
    cards=[]
    for row in rows:
        sheet="contact_sheets/"+Path(row["contact_sheet"]).name
        videos=" ".join(f'<a href="{html.escape((Path(row["video_directory"])/(c+"_rgb_vis.mp4")).as_uri())}">{c}</a>' for c in CAMERAS)
        cards.append(f'<section><h2>{row["host"]}/{row["episode"]}</h2><p>{html.escape(str(row["review_flags"]))} — {videos}</p><a href="{sheet}"><img loading="lazy" width="1050" src="{sheet}"></a></section>')
    (out/"review.html").write_text('<!doctype html><meta charset="utf-8"><title>HGwind quality review</title><style>body{background:#15191e;color:#eee;font:16px sans-serif;margin:25px}a{color:#7cc5ff}section{margin-bottom:45px}</style><h1>HGwind: 40 development demonstrations</h1><p>Event snapshots and full RGB videos. Human review pending; flags are triage, not exclusions.</p>'+"\n".join(cards))
    print("QUALITY_REPORT "+str(out/"report.json"));print("REVIEW "+str(out/"review.html"))
    for key,value in comparison.items():print("GROUP "+key+" "+json.dumps(value))


if __name__=="__main__":main()
