"""Relocalize the Fern 006 keyframes against the published fixed SfM map."""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.red_solace_multivideo import (
    extract_features, image_view, load_anchors, load_sparse_points, localize,
)

def failed_checks(pose):
    if pose["registered"]:
        return ""
    if pose["reason"] != "pose_consensus_below_quality_threshold":
        return pose["reason"]
    checks = {
        "inliers<20": pose["num_inliers"] < 20,
        "inlier_ratio<0.16": pose["inlier_ratio"] < 0.16,
        "median_error>2.8px": pose["median_error_px"] > 2.8,
        "coverage<2.5%": pose["image_coverage"] < 0.025,
        "grid_cells<6": pose["grid_cells"] < 6,
        "supporting_anchors<2": pose["anchors_with_at_least_five_inliers"] < 2,
    }
    return " | ".join(label for label, failed in checks.items() if failed)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("fern_006_diagnostics.csv"))
    args = parser.parse_args()
    start = time.perf_counter()
    cv2.setNumThreads(2)
    cv2.setRNGSeed(6121)
    outputs = ROOT / "outputs"
    final = outputs / "final" / "fern"
    experiment = json.loads((final / "experiment_metrics.json").read_text())["experiments"]["fern_video_rootsift"]
    points = load_sparse_points(final / "point_clouds" / "fern_video_rootsift_sparse.ply")
    assert len(points) == 4657
    frames = outputs / "fern_72" / "keyframes" / "Fern"
    entries = experiment["views"]
    assert len(entries) == 72 and sum(entry["registered"] for entry in entries) == 24
    anchors = load_anchors(entries, frames, points, 2400)
    assert len(anchors) == 12
    mapped_times = [entry["timestamp_sec"] for entry in entries if entry["registered"]]
    gaps = list(zip(mapped_times, mapped_times[1:]))
    gap_start, gap_end = max(gaps, key=lambda pair: pair[1] - pair[0])
    rows = []
    for index, entry in enumerate(entries):
        path = frames / entry["name"]
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode frame {path}")
        view = image_view(image, entry["name"], path)
        extract_features([view], max_features=2400)
        eligible = [anchor for anchor in anchors if anchor.view.name != view.name]
        pose = localize(view, points, eligible)
        timestamp = entry["timestamp_sec"]
        result = {
            "keyframe_index": index,
            "name": entry["name"],
            "frame_index": entry["frame_index"],
            "timestamp_sec": timestamp,
            "mapped_before": entry["registered"],
            "inside_largest_gap": gap_start < timestamp < gap_end,
            "anchors_used": len(eligible),
            "keypoints": len(view.kps),
            "localized": pose["registered"],
            "outcome": "accepted" if pose["registered"] else pose["reason"],
            "failed_checks": failed_checks(pose),
        }
        for key in (
            "num_correspondences", "num_inliers", "inlier_ratio", "median_error_px",
            "p90_error_px", "image_coverage", "grid_cells", "anchors_with_at_least_five_inliers",
            "inliers_voted_by_multiple_anchors", "focal_factor",
        ):
            result[key] = pose.get(key)
        if entry["registered"] and pose["registered"]:
            relative_rotation = np.asarray(pose["world_to_camera_R"]) @ np.asarray(entry["world_to_camera_R"]).T
            result["control_rotation_delta_deg"] = np.degrees(
                np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1))
            ).item()
        else:
            result["control_rotation_delta_deg"] = None
        rows.append(result)
        print(index, round(timestamp, 2), result["outcome"], result["num_correspondences"], flush=True)
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for label, selection in (
        ("previously unregistered", [r for r in rows if not r["mapped_before"]]),
        ("largest gap", [r for r in rows if r["inside_largest_gap"]]),
        ("registered controls", [r for r in rows if r["mapped_before"]]),
    ):
        print(label, len(selection), dict(Counter(r["outcome"] for r in selection)))
        print("correspondences min/median/max", np.quantile([r["num_correspondences"] for r in selection], [0, 0.5, 1]).tolist())
        print("failed checks", dict(Counter(check for r in selection for check in r["failed_checks"].split(" | ") if check)))
    print("Gap endpoints", gap_start, gap_end)
    print(f"Elapsed: {time.perf_counter() - start:.1f} sec; OpenCV {cv2.__version__}")

if __name__ == "__main__":
    main()

