"""Run the AI6121 incremental SfM experiments on the original dataset."""

import argparse
import json
import time
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

from sfm.data import extract_video_keyframes, load_camera_calibrations, load_image_dataset
from sfm.densify import densify_point_cloud
from sfm.features import extract_features, match_features_pairwise
from sfm.geometry import verify_two_view_geometry
from sfm.pipeline import IncrementalSfM
from sfm.visualization import (
    build_point_cloud_render_frame,
    export_ply,
    plot_feature_matches,
    plot_scene_graph,
    render_point_cloud_views,
)


SCENES = {
    "fern": ("Fern", "006.mp4", 72),
    "red-solace": ("Red_Solace", "003.mp4", 40),
}
PROJECT_ROOT = Path(__file__).resolve().parent


def load_views(scene, options, calibrations):
    folder, clip, default_count = SCENES[scene]
    if options.input == "photos":
        views = load_image_dataset(
            str(options.dataset / folder / "image"),
            max_dim=options.max_dim,
            focal_factor=1.05 if options.focal_factor is None else options.focal_factor,
            calibrations=calibrations,
        )
        if len(views) < 2:
            raise ValueError(f"{folder}: need at least two readable photos")
        return views, {}

    path = options.dataset / folder / "video" / clip
    if not path.is_file():
        raise FileNotFoundError(f"Missing input clip: {path}")
    views, stats = extract_video_keyframes(
        [str(path)],
        target_keyframes_per_video=options.keyframes or default_count,
        max_dim=options.max_dim,
        focal_factor=0.95 if options.focal_factor is None else options.focal_factor,
        save_dir=str(options.output_dir / "keyframes" / folder),
        calibrations=calibrations,
        adaptive=not options.uniform_keyframes,
    )
    if len(views) < 2:
        raise ValueError(f"{folder}: need at least two readable frames from {path.name}")
    return views, stats


def match_views(views, method, options):
    extract_features(views, method=method, max_features=options.features)
    matching = options.matching
    if matching == "auto":
        matching = "sequential_loop" if options.input == "video" and len(views) > 50 else "exhaustive"
    return match_features_pairwise(
        views, ratio_thresh=options.ratio, min_raw_matches=options.min_inliers,
        matching_mode=matching,
    )


def reconstruct(views, raw_matches, options):
    cv2.setRNGSeed(0)
    pairs = verify_two_view_geometry(
        views, raw_matches, ransac_px_thresh=options.ransac_px, min_inliers=options.min_inliers
    )
    print(f"  {len(raw_matches)} raw pairs, {len(pairs)} verified pairs", flush=True)
    mapper = IncrementalSfM(
        views,
        pairs,
        use_ba=not options.no_ba,
        optimize_focal=not options.fixed_focal,
        max_seed_attempts=options.max_seeds,
    )
    return pairs, mapper.run()


def largest_match_component(view_count, pairs):
    neighbors = [set() for _ in range(view_count)]
    for first, second in pairs:
        neighbors[first].add(second)
        neighbors[second].add(first)
    unseen = set(range(view_count))
    largest = 0
    while unseen:
        queue = [unseen.pop()]
        size = 0
        while queue:
            view = queue.pop()
            size += 1
            found = unseen & neighbors[view]
            unseen -= found
            queue.extend(found)
        largest = max(size, largest)
    return largest


def summarize(state, pairs, raw_matches):
    metrics = state.get_summary_metrics()
    errors = state.compute_all_reprojection_errors()
    registered = set(state.registered_cams)
    grid_by_view = {view: set() for view in registered}
    tracks = [point for point in state.points3d if point.valid and len(point.observations) >= 2]
    for point in tracks:
        for index, keypoint in point.observations.items():
            if index in registered:
                view = state.dataset[index]
                x, y = view.kps[keypoint]
                grid_by_view[index].add(
                    (min(7, int(8 * x / view.width)), min(7, int(8 * y / view.height)))
                )

    metrics.update(
        num_candidate_pairs=len(raw_matches),
        num_verified_pairs=len(pairs),
        largest_match_component=largest_match_component(len(state.dataset), pairs),
        p90_reproj_error_px=float(np.percentile(errors, 90)) if len(errors) else None,
        median_image_grid_coverage=(
            float(np.median([len(cells) / 64 for cells in grid_by_view.values()]))
            if grid_by_view else None
        ),
        init_pair=[state.dataset[i].name for i in state.init_pair if i >= 0],
        seed_attempts=state.seed_attempts,
        num_seed_candidates=state.num_seed_candidates,
        num_feature_tracks=state.num_feature_tracks,
        track_conflicts=state.track_conflicts,
        registration_retries=state.registration_retries,
        bundle_adjustment=state.ba_history,
        views=[
            {
                "name": view.name,
                "frame_index": view.frame_index,
                "timestamp_sec": view.timestamp_sec,
                "registered": i in registered,
                "camera_group": view.camera_group,
                "intrinsics_source": view.intrinsics_source,
                "initial_K": view.K.tolist(),
                "final_K": state.cam_K[i].tolist() if i in registered else None,
                "world_to_camera_R": state.cam_R[i].tolist() if i in registered else None,
                "world_to_camera_t": state.cam_t[i].ravel().tolist() if i in registered else None,
            }
            for i, view in enumerate(state.dataset)
        ],
    )
    return metrics


def export_results(output_dir, name, state, pairs, raw_matches, metrics, use_densify):
    xyz, rgb, _ = state.get_valid_points()
    figure_dir = output_dir / "figures"
    cloud_dir = output_dir / "point_clouds"
    figure_dir.mkdir(parents=True, exist_ok=True)
    cloud_dir.mkdir(parents=True, exist_ok=True)
    plot_scene_graph(
        len(state.dataset), pairs, state,
        save_path=str(figure_dir / f"{name}_scene_graph.png"), title=name,
    )
    if state.init_pair in pairs and state.init_pair in raw_matches:
        first, second = state.init_pair
        pair = pairs[state.init_pair]
        plot_feature_matches(
            state.dataset[first], state.dataset[second], raw_matches[state.init_pair], pair.seed_matches,
            save_path=str(figure_dir / f"{name}_init_matches.png"),
            title=f"Initial pair: {state.dataset[first].name} / {state.dataset[second].name}",
        )

    if not len(xyz):
        metrics["semi_dense_status"] = "no_sparse_reconstruction"
        return
    render_frame = build_point_cloud_render_frame(xyz, state)
    export_ply(str(cloud_dir / f"{name}_sparse.ply"), xyz, rgb)
    render_point_cloud_views(
        xyz, rgb, state, save_path=str(figure_dir / f"{name}_sparse_3d.png"),
        title=f"{name}: sparse reconstruction", render_frame=render_frame,
    )
    if not use_densify:
        metrics["semi_dense_status"] = "disabled"
        return

    # A practical gate for the demo: the multi-view stage requires broad pose support.
    if (
        len(state.registered_cams) < 8
        or metrics["registration_rate"] < 0.60
        or metrics["track_3plus_fraction"] < 0.20
        or metrics["p90_reproj_error_px"] is None
        or metrics["p90_reproj_error_px"] > 4.0
    ):
        metrics["semi_dense_status"] = "insufficient_sparse_support"
        return
    start = time.perf_counter()
    dense_xyz, dense_rgb = densify_point_cloud(state, max_dense_features=4000)
    metrics["num_semi_dense_points"] = len(dense_xyz)
    metrics["semi_dense_sec"] = round(time.perf_counter() - start, 2)
    metrics["semi_dense_status"] = "completed" if len(dense_xyz) > len(xyz) else "no_additional_points"
    export_ply(str(cloud_dir / f"{name}_semi_dense.ply"), dense_xyz, dense_rgb)
    render_point_cloud_views(
        dense_xyz, dense_rgb, state, save_path=str(figure_dir / f"{name}_semi_dense_3d.png"),
        title=f"{name}: multi-view semi-dense reconstruction", render_frame=render_frame,
    )


def parse_options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=["both", *SCENES], default="both")
    parser.add_argument("--input", choices=["video", "photos"], default="video")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--calibration", type=Path, help="JSON camera calibration keyed by dataset-relative path")
    parser.add_argument("--focal-factor", type=float, help="Override the fallback focal length / long side ratio")
    parser.add_argument("--keyframes", type=int, help="Target frames per video")
    parser.add_argument("--uniform-keyframes", action="store_true")
    parser.add_argument("--max-dim", type=int, default=1280)
    parser.add_argument("--features", type=int, default=2400)
    parser.add_argument("--feature-method", choices=["rootsift", "sift", "orb"], default="rootsift")
    parser.add_argument("--ablation", action="store_true", help="Compare all three descriptors with shared settings")
    parser.add_argument("--ratio", type=float, default=0.78)
    parser.add_argument("--matching", choices=["auto", "exhaustive", "sequential_loop"], default="auto")
    parser.add_argument("--ransac-px", type=float, default=2.5)
    parser.add_argument("--min-inliers", type=int, default=20)
    parser.add_argument("--max-seeds", type=int, default=4)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--no-ba", action="store_true")
    parser.add_argument("--fixed-focal", action="store_true")
    parser.add_argument("--no-densify", action="store_true")
    options = parser.parse_args()
    if not 0 < options.ratio < 1:
        parser.error("--ratio must be between 0 and 1")
    if options.keyframes is not None and options.keyframes < 2:
        parser.error("--keyframes must be at least 2")
    if options.features < options.min_inliers or options.min_inliers < 8:
        parser.error("--features must be at least --min-inliers, which must be at least 8")
    if min(options.max_dim, options.max_seeds, options.threads) <= 0 or not 0 < options.ransac_px < float("inf"):
        parser.error("--max-dim, --max-seeds, --threads and --ransac-px must be positive")
    if options.focal_factor is not None and not 0 < options.focal_factor < float("inf"):
        parser.error("--focal-factor must be positive")
    return options


def main():
    options = parse_options()
    cv2.setNumThreads(options.threads)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    calibrations = (
        load_camera_calibrations(options.calibration, options.dataset) if options.calibration else None
    )
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(options).items()}
    results = {"settings": settings, "experiments": {}}
    scenes = list(SCENES) if options.scene == "both" else [options.scene]
    methods = [options.feature_method]
    if options.ablation:
        methods.extend(method for method in ("rootsift", "sift", "orb") if method not in methods)
    for scene in scenes:
        start = time.perf_counter()
        views, input_stats = load_views(scene, options, calibrations)
        print(f"{scene}: selected {len(views)} views in {time.perf_counter() - start:.1f}s", flush=True)
        for method in methods:
            name = f"{scene.replace('-', '_')}_{options.input}_{method}"
            print(f"{name}: extracting and matching features", flush=True)
            start = time.perf_counter()
            raw_matches = match_views(views, method, options)
            pairs, state = reconstruct(views, raw_matches, options)
            metrics = summarize(state, pairs, raw_matches)
            metrics["keyframe_selection"] = input_stats
            metrics["feature_method"] = method
            metrics["reconstruction_sec"] = round(time.perf_counter() - start, 2)
            print(
                f"  registered {len(state.registered_cams)}/{len(views)}, "
                f"{metrics['num_3d_points']} sparse points, "
                f"{metrics['track_3plus_fraction']:.1%} in 3+ views",
                flush=True,
            )
            if method == options.feature_method:
                export_results(options.output_dir, name, state, pairs, raw_matches, metrics, not options.no_densify)
            results["experiments"][name] = metrics
            target = options.output_dir / "experiment_metrics.json"
            temp = target.with_suffix(".json.tmp")
            temp.write_text(json.dumps(results, indent=2, allow_nan=False), encoding="utf-8")
            temp.replace(target)
    print(f"Results: {options.output_dir}", flush=True)


if __name__ == "__main__":
    main()
