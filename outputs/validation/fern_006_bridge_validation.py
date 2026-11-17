"""Compare two fixed-budget keyframe sets from Fern clip 006."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from run_experiments import PROJECT_ROOT, match_views, reconstruct, summarize
from sfm.data import _resize_image_max_dim, _track_motion
from tools.red_solace_multivideo import image_view

def corners(gray):
    return cv2.goodFeaturesToTrack(gray, maxCorners=250, qualityLevel=0.01, minDistance=6)

def score(gray, left, right, left_corners, right_corners):
    left_overlap, _, left_duplicate = _track_motion(left, gray, left_corners)
    right_overlap, _, right_duplicate = _track_motion(right, gray, right_corners)
    return min(left_overlap, right_overlap), left_duplicate or right_duplicate

def read_frames(video, indices):
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 4])
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    images = {}
    small = {}
    try:
        for index in range(max(indices) + 1):
            if not capture.grab():
                raise ValueError(f"Cannot advance decoder to frame {index}")
            if index not in indices:
                continue
            ok, image = capture.retrieve()
            if not ok:
                raise ValueError(f"Cannot decode frame {index}")
            image = _resize_image_max_dim(image, 1280)
            images[index] = image
            small[index] = cv2.cvtColor(_resize_image_max_dim(image, 480), cv2.COLOR_BGR2GRAY)
            if len(images) % 50 == 0:
                print(f"Decoded {len(images)}/{len(indices)} requested frames", flush=True)
    finally:
        capture.release()
    return images, small, fps

def select_replacements(baseline, windows, gray):
    replacement = baseline.copy()
    changes = []
    for slot, candidates in windows.items():
        original = baseline[slot]
        left = gray[baseline[slot - 1]]
        right = gray[baseline[slot + 1]]
        left_corners, right_corners = corners(left), corners(right)
        baseline_score, _ = score(gray[original], left, right, left_corners, right_corners)
        original_sharpness = float(cv2.Laplacian(gray[original], cv2.CV_32F).var())
        scored = []
        for index in candidates:
            overlap, duplicate = score(gray[index], left, right, left_corners, right_corners)
            sharpness = float(cv2.Laplacian(gray[index], cv2.CV_32F).var())
            if not duplicate and overlap > baseline_score and sharpness >= 0.90 * original_sharpness:
                scored.append((overlap, sharpness, index))
        if scored:
            overlap, sharpness, chosen = max(scored)
            replacement[slot] = chosen
            changes.append(dict(slot=slot, original_frame=original, replacement_frame=chosen,
                                original_min_overlap=baseline_score, replacement_min_overlap=overlap,
                                original_sharpness=original_sharpness, replacement_sharpness=sharpness))
    return replacement, changes

def evaluate(indices, images, fps, video, gap):
    views = []
    for slot, index in enumerate(indices):
        view = image_view(images[index], f"006_kf{slot:03d}_f{index:04d}.jpg", video)
        view.idx = slot
        view.frame_index = index
        view.timestamp_sec = index / fps
        view.source_type = "video"
        view.camera_group = "fern_006"
        views.append(view)
    options = SimpleNamespace(input="video", matching="auto", features=2400, ratio=0.78,
                              ransac_px=2.5, min_inliers=20, no_ba=False, fixed_focal=False, max_seeds=4)
    matches = match_views(views, "rootsift", options)
    pairs, state = reconstruct(views, matches, options)
    metrics = summarize(state, pairs, matches)
    registered = [view for view in metrics["views"] if view["registered"]]
    registered_times = [view["timestamp_sec"] for view in registered]
    metrics["largest_internal_gap_sec"] = max(
        (b - a for a, b in zip(registered_times, registered_times[1:])), default=None)
    metrics["largest_gap_including_clip_ends_sec"] = max(
        (b - a for a, b in zip([0.0, *registered_times], [*registered_times, indices[-1] / fps])))
    metrics["registrations_inside_original_gap"] = sum(gap[0] < view["frame_index"] < gap[1] for view in registered)
    return metrics

def encode_numpy(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__}")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=PROJECT_ROOT / "dataset/Fern/video/006.mp4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(6)
    source = PROJECT_ROOT / "outputs/final/fern/experiment_metrics.json"
    published = json.loads(source.read_text())["experiments"]["fern_video_rootsift"]["views"]
    baseline = [view["frame_index"] for view in published]
    assert len(baseline) == len(set(baseline)) == 72
    mapped = [view["frame_index"] for view in published if view["registered"]]
    gap = max(zip(mapped, mapped[1:]), key=lambda pair: pair[1] - pair[0])
    slots = [slot for slot, index in enumerate(baseline) if gap[0] < index < gap[1]][::2]
    windows = {
        slot: sorted(set(np.linspace(baseline[slot - 1] + 1, baseline[slot + 1] - 1, 9, dtype=int).tolist())
                     - set(baseline))
        for slot in slots
    }
    indices = set(baseline).union(*(set(window) for window in windows.values()))
    images, gray, fps = read_frames(args.video, indices)
    modified, changes = select_replacements(baseline, windows, gray)
    assert changes and len(modified) == len(set(modified)) == 72 and modified == sorted(modified)
    print(f"Selected {len(changes)} replacement frames before reconstruction", flush=True)
    print("Baseline reconstruction", flush=True)
    original = evaluate(baseline, images, fps, args.video, gap)
    print("Replacement reconstruction", flush=True)
    revised = evaluate(modified, images, fps, args.video, gap)
    result = {
        "source_video": "dataset/Fern/video/006.mp4",
        "opencv_version": cv2.__version__,
        "fps": fps,
        "original_gap_source_frame_indices": gap,
        "target_slots": slots,
        "selection": ("Nine potential frame indices linearly spaced between the two fixed neighbors, excluding "
                      "all baseline indices. Maximize the weaker adjacent optical-flow overlap, requiring an "
                      "improvement over the original, at least 90% of its sharpness and no near duplicates."),
        "settings": {"keyframes": 72, "max_dim": 1280, "descriptor": "RootSIFT", "features": 2400,
                     "ratio": 0.78, "matching": "sequential_loop", "ransac_px": 2.5, "min_inliers": 20,
                     "max_seeds": 4, "initial_focal_factor": 0.95, "bundle_adjustment": True,
                     "refine_shared_focal": True, "cv2_threads": 6},
        "replacements": changes,
        "baseline": original,
        "replacement": revised,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False, default=encode_numpy) + "\n", encoding="utf-8")
    for name, metrics in (("baseline", original), ("replacement", revised)):
        print(name, {key: metrics[key] for key in (
            "num_registered_images", "num_3d_points", "largest_internal_gap_sec",
            "largest_gap_including_clip_ends_sec", "registrations_inside_original_gap")}, flush=True)

if __name__ == "__main__":
    main()


