"""Localize additional Red Solace clips in the saved 003.mp4 SfM frame."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sfm.data import ImageData, _read_candidate, _resize_image_max_dim
from sfm.features import extract_features, match_pair_mutual_ratio
from sfm.geometry import image_coverage
from tools.red_solace_stereo import load_sparse_points


EXPERIMENT = "red_solace_video_rootsift"
FINAL = ROOT / "outputs/final/red-solace"
EXTRA_CLIPS = ("001", "002", "004", "005", "006")
FOCAL_FACTORS = (0.62, 0.76, 0.95, 1.12, 1.35)


@dataclass
class Anchor:
    view: ImageData
    point_for_keypoint: np.ndarray


def image_view(image: np.ndarray, name: str, path: Path, max_dim: int = 1280) -> ImageData:
    image = _resize_image_max_dim(image, max_dim)
    h, w = image.shape[:2]
    return ImageData(
        idx=0, name=name, path=str(path), width=w, height=h, K=intrinsics(w, h, 0.95),
        image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
    )


def intrinsics(width: int, height: int, factor: float) -> np.ndarray:
    f = factor * max(width, height)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def load_anchors(views: list[dict], directory: Path, points: np.ndarray, features: int) -> list[Anchor]:
    anchors = []
    registered = [view for view in views if view["registered"]]
    for entry in registered[::2]:
        filename = directory / entry["name"]
        image = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode anchor keyframe {filename}")
        view = image_view(image, entry["name"], filename)
        view.K = np.asarray(entry["final_K"], dtype=np.float64)
        R, t = np.asarray(entry["world_to_camera_R"]), np.asarray(entry["world_to_camera_t"])
        camera_points = points @ R.T + t.reshape(3)
        depth = camera_points[:, 2]
        pix = camera_points @ view.K.T
        uv = pix[:, :2] / np.where(depth > 0, depth, 1)[:, None]
        visible = (
            (depth > 0) & (uv[:, 0] > 0) & (uv[:, 0] < view.width - 1)
            & (uv[:, 1] > 0) & (uv[:, 1] < view.height - 1)
        )
        ids = np.flatnonzero(visible)
        if not len(ids):
            continue
        extract_features([view], max_features=features)
        if not len(view.kps):
            continue
        projection_tree = cKDTree(uv[ids])
        distance, projected = projection_tree.query(view.kps, distance_upper_bound=2.75)
        reverse = cKDTree(view.kps).query(uv[ids])[1]
        near = np.flatnonzero(np.isfinite(distance))
        near = near[reverse[projected[near]] == near]
        point_ids = np.full(len(view.kps), -1, dtype=np.int32)
        point_ids[near] = ids[projected[near]]
        anchors.append(Anchor(view, point_ids))
    return anchors


def read_views(path: Path, count: int, features: int):
    try:
        name = str(path.relative_to(Path.cwd()))
    except ValueError:
        name = str(path)
    capture = cv2.VideoCapture(name, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 3])
    if not capture.isOpened():
        raise OSError(f"Cannot open video {path}")
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if total <= 0:
            raise OSError(f"No frames available in {path}")
        for index in np.unique(np.linspace(0, total - 1, min(count, total), dtype=int)):
            frame = _read_candidate(capture, int(index), fps)
            if frame is None:
                continue
            view = image_view(frame.image, f"{path.stem}_f{frame.index:04d}.jpg", path)
            view.frame_index, view.timestamp_sec = frame.index, frame.timestamp_sec
            extract_features([view], max_features=features)
            yield view
    finally:
        capture.release()


def correspondences(view: ImageData, anchors: list[Anchor]):
    votes = defaultdict(lambda: defaultdict(set))
    for anchor_id, anchor in enumerate(anchors):
        matches = match_pair_mutual_ratio(view.descs, anchor.view.descs, ratio_thresh=0.77)
        for key, anchor_key in matches:
            point_id = anchor.point_for_keypoint[anchor_key]
            if point_id >= 0:
                votes[int(key)][int(point_id)].add(anchor_id)
    proposals = []
    for key, alternatives in votes.items():
        ranked = sorted(alternatives.items(), key=lambda value: (-len(value[1]), value[0]))
        point_id, support = ranked[0]
        if len(ranked) > 1 and len(support) == len(ranked[1][1]):
            continue
        proposals.append((key, point_id, support))
    proposals.sort(key=lambda proposal: (-len(proposal[2]), proposal[0]))
    unique = {}
    for key, point_id, support in proposals:
        unique.setdefault(point_id, (key, support))
    keys = np.fromiter((key for key, _ in unique.values()), dtype=np.int32)
    point_ids = np.fromiter(unique, dtype=np.int32)
    sources = [support for _, support in unique.values()]
    return keys, point_ids, sources


def residuals(parameters: np.ndarray, xyz: np.ndarray, uv: np.ndarray, view: ImageData):
    K = intrinsics(view.width, view.height, float(np.exp(parameters[6])))
    projected = cv2.projectPoints(xyz, parameters[:3], parameters[3:6], K, None)[0].reshape(-1, 2)
    return (projected - uv).ravel()


def measure_pose(parameters: np.ndarray, xyz: np.ndarray, uv: np.ndarray, view: ImageData):
    R = cv2.Rodrigues(parameters[:3])[0]
    depth = (xyz @ R.T + parameters[3:6])[:, 2]
    error = np.linalg.norm(residuals(parameters, xyz, uv, view).reshape(-1, 2), axis=1)
    inlier = (depth > 0) & (error < 4.0)
    return inlier, error


def localize(view: ImageData, points: np.ndarray, anchors: list[Anchor]) -> dict:
    keys, point_ids, sources = correspondences(view, anchors)
    stats = {"num_correspondences": len(keys)}
    if len(keys) < 20:
        return {"registered": False, "reason": "insufficient_2d_3d_correspondences", **stats}
    xyz, uv = points[point_ids], view.kps[keys]
    best = None
    for factor in FOCAL_FACTORS:
        K = intrinsics(view.width, view.height, factor)
        ok, rotation, translation, indices = cv2.solvePnPRansac(
            xyz, uv, K, None, iterationsCount=1500, reprojectionError=5.0,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or indices is None or len(indices) < 12:
            continue
        indices = indices.ravel()
        initial = np.r_[rotation.ravel(), translation.ravel(), np.log(factor)]
        lower = np.r_[np.full(6, -np.inf), np.log(0.50)]
        upper = np.r_[np.full(6, np.inf), np.log(1.65)]
        solution = least_squares(
            residuals, initial, args=(xyz[indices], uv[indices], view),
            bounds=(lower, upper), loss="huber", f_scale=2.0, max_nfev=20,
        ).x
        valid, errors = measure_pose(solution, xyz, uv, view)
        count = int(valid.sum())
        if count < 12:
            continue
        score = (count, -float(np.median(errors[valid])))
        if best is None or score > best[0]:
            best = (score, solution, valid, errors)
    if best is None:
        return {"registered": False, "reason": "pose_solver_no_consensus", **stats}

    _, solution, valid, errors = best
    good_ids = np.flatnonzero(valid)
    supported_anchors = Counter(anchor for point in good_ids for anchor in sources[point])
    strong_anchors = sum(count >= 5 for count in supported_anchors.values())
    coverage = image_coverage(uv[valid], view.width, view.height)
    cell_ids = np.floor(uv[valid] * (8 / np.array([view.width, view.height]))).astype(int)
    grid_cells = len(np.unique(cell_ids, axis=0))
    stats.update(
        num_inliers=len(good_ids), inlier_ratio=len(good_ids) / len(keys),
        median_error_px=float(np.median(errors[valid])),
        p90_error_px=float(np.percentile(errors[valid], 90)),
        image_coverage=coverage, grid_cells=grid_cells,
        anchors_with_at_least_five_inliers=strong_anchors,
        inliers_voted_by_multiple_anchors=sum(len(sources[index]) >= 2 for index in good_ids),
        focal_factor=float(np.exp(solution[6])),
    )
    if (
        len(good_ids) < 20 or stats["inlier_ratio"] < 0.16 or stats["median_error_px"] > 2.8
        or coverage < 0.025 or grid_cells < 6 or strong_anchors < 2
    ):
        return {"registered": False, "reason": "pose_consensus_below_quality_threshold", **stats}
    K = intrinsics(view.width, view.height, stats["focal_factor"])
    return {
        "registered": True,
        "world_to_camera_R": cv2.Rodrigues(solution[:3])[0].tolist(),
        "world_to_camera_t": solution[3:6].tolist(),
        "final_K": K.tolist(),
        "intrinsics_source": "focal_fit_from_pnp",
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=FINAL / "experiment_metrics.json")
    parser.add_argument("--anchors", type=Path, default=ROOT / "outputs/red_solace_40/keyframes/Red_Solace")
    parser.add_argument("--sparse", type=Path, default=FINAL / "point_clouds/red_solace_video_rootsift_sparse.ply")
    parser.add_argument("--video-dir", type=Path, default=ROOT / "dataset/Red_Solace/video")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/case-study/red-solace-multi-video-poses.json")
    parser.add_argument("--clips", nargs="+", choices=EXTRA_CLIPS, default=EXTRA_CLIPS)
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--features", type=int, default=2400)
    args = parser.parse_args()
    if args.frames_per_clip < 1 or args.features < 100:
        parser.error("Use at least one query per clip and 100 features per view")
    cv2.setNumThreads(6)
    cv2.setRNGSeed(2026)
    data = json.loads(args.metrics.read_text(encoding="utf-8"))
    experiment = data["experiments"][EXPERIMENT]
    views = [dict(view, clip="003") for view in experiment["views"]]
    points = load_sparse_points(args.sparse)
    anchors = load_anchors(views, args.anchors, points, args.features)
    print(f"Prepared {len(anchors)} registered anchor images and {len(points):,} saved points", flush=True)
    summary = {"003": {"sampled": len(views), "registered": sum(view["registered"] for view in views)}}
    for clip in dict.fromkeys(args.clips):
        registered = 0
        sampled = 0
        path = args.video_dir.resolve() / f"{clip}.mp4"
        for view in read_views(path, args.frames_per_clip, args.features):
            pose = localize(view, points, anchors)
            views.append({
                "name": view.name, "clip": clip, "frame_index": view.frame_index,
                "timestamp_sec": view.timestamp_sec, **pose,
            })
            registered += int(pose["registered"])
            sampled += 1
            status = "registered" if pose["registered"] else pose["reason"]
            print(
                f"{view.name}: {status}, {pose.get('num_inliers', 0)}/"
                f"{pose['num_correspondences']} inliers", flush=True,
            )
        summary[clip] = {"sampled": sampled, "registered": registered}
    out = {
        "source": "003.mp4 sparse map with 2D–3D localization of additional videos",
        "settings": {
            "frames_per_extra_clip": args.frames_per_clip, "features": args.features,
            "anchor_count": len(anchors), "extra_clips": args.clips,
        },
        "clip_summary": summary,
        "experiments": {EXPERIMENT: {
            "views": views, "num_total_images": len(views),
            "num_registered_images": sum(view["registered"] for view in views),
        }},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(out, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
