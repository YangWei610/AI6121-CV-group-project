"""Recover pixel stereo samples in the saved SfM coordinate frame."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/final/red-solace"
SCENES = {
    "red-solace": ("red_solace_video_rootsift", "red_solace_40/keyframes/Red_Solace"),
    "fern": ("fern_video_rootsift", "fern_72/keyframes/Fern"),
}
COVERAGE_PRESETS = {
    # rectification alpha, disparity span limit, temporal search radius
    "conservative": (0.05, 224, 2),
    "expanded": (0.38, 384, 3),
}


@dataclass
class View:
    index: int
    name: str
    image: np.ndarray
    original_size: tuple[int, int]
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    camera_group: str = ""

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t


@dataclass
class RectifiedPair:
    reference: View
    target: View
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    axis: int
    min_disparity: int
    disparity_count: int
    score: float


@dataclass
class DepthMap:
    pair: RectifiedPair
    depth: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    uv: np.ndarray
    lr_error: np.ndarray


def load_views(
    metrics: Path, image_dir: Path, max_width: int,
    experiment: str = "red_solace_video_rootsift",
) -> list[View]:
    experiments = json.loads(metrics.read_text(encoding="utf-8"))["experiments"]
    entries = experiments[experiment]["views"]
    views = []
    for index, entry in enumerate(entries):
        if not entry["registered"]:
            continue
        path = image_dir / entry["name"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing source keyframe: {path}")
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to decode {path}")
        height, width = image.shape[:2]
        target_width = min(width, max_width)
        target_height = round(height * target_width / width)
        K = np.asarray(entry["final_K"], dtype=np.float64)
        K[0] *= target_width / width
        K[1] *= target_height / height
        views.append(View(
            index, entry["name"],
            cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA),
            (width, height), K, np.asarray(entry["world_to_camera_R"], dtype=np.float64),
            np.asarray(entry["world_to_camera_t"], dtype=np.float64).reshape(3),
            entry.get("camera_group") or "",
        ))
    if len(views) < 3:
        raise ValueError("Stereo depth verification requires at least three registered views")
    if len({view.image.shape for view in views}) != 1:
        raise ValueError("Expected keyframes with matching aspect ratios")
    return views


def load_sparse_points(path: Path) -> np.ndarray:
    with path.open(encoding="ascii") as stream:
        if stream.readline().strip() != "ply" or stream.readline().strip() != "format ascii 1.0":
            raise ValueError(f"Expected an ASCII SfM PLY: {path}")
        for line in stream:
            if line.strip() == "end_header":
                break
        points = np.loadtxt(stream, usecols=(0, 1, 2), dtype=np.float64, ndmin=2)
    if len(points) < 100:
        raise ValueError("Need at least 100 sparse points to bound disparity search")
    return points


def propose_pair(
    reference: View, target: View, points: np.ndarray, alpha: float, disparity_limit: int,
) -> RectifiedPair | None:
    height, width = reference.image.shape[:2]
    rotation = target.R @ reference.R.T
    translation = target.t - rotation @ reference.t
    if np.linalg.norm(translation) < 1e-5:
        return None
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        reference.K, None, target.K, None, (width, height), rotation, translation,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha,
    )
    axis = int(abs(P2[1, 3]) > abs(P2[0, 3]))
    rect_points = (points @ reference.R.T + reference.t) @ R1.T
    rect_points = rect_points[rect_points[:, 2] > 1e-4]
    left = (rect_points @ P1[:2, :3].T + P1[:2, 3]) / rect_points[:, 2, None]
    right = (rect_points @ P2[:2, :3].T + P2[:2, 3]) / rect_points[:, 2, None]
    usable = np.all(
        (left > 8) & (right > 8) & (left < (width - 8, height - 8))
        & (right < (width - 8, height - 8)), axis=1,
    )
    disparities = left[usable, axis] - right[usable, axis]
    if len(disparities) < 100:
        return None

    low, median, high = np.percentile(disparities, [1, 50, 99])
    if abs(median) < 6:
        return None
    margin = max(8, (high - low) * 0.12)
    match_width = (width, height)[axis]
    minimum = max(-match_width + 24, int(np.floor(low - margin)))
    maximum = min(match_width - 24, int(np.ceil(high + margin)))
    count = int(np.ceil((maximum - minimum + 1) / 16)) * 16
    if count > disparity_limit or maximum <= minimum or count + abs(minimum) >= match_width - 8:
        return None
    # Favor substantial common image content and enough depth precision for
    # accurate triangulation, without large changes in the viewed surface.
    angle = float(np.linalg.norm(cv2.Rodrigues(rotation)[0]))
    score = len(disparities) * min(abs(median) / 20, 1) / (1 + angle * 2 + count / 256)
    if target.index > reference.index:
        score *= 1.03
    return RectifiedPair(reference, target, R1, R2, P1, P2, Q, axis, minimum, count, score)


def rectified_image(view: View, rotation: np.ndarray, projection: np.ndarray):
    height, width = view.image.shape[:2]
    x, y = cv2.initUndistortRectifyMap(
        view.K, None, rotation, projection[:, :3], (width, height), cv2.CV_32FC1,
    )
    valid = (x >= 3) & (x < width - 4) & (y >= 3) & (y < height - 4)
    return cv2.remap(view.image, x, y, cv2.INTER_LINEAR), valid, (x, y)


def disparity(left: np.ndarray, right: np.ndarray, minimum: int, count: int) -> np.ndarray:
    block = 5
    matcher = cv2.StereoSGBM_create(
        minDisparity=minimum, numDisparities=count, blockSize=block,
        P1=8 * block ** 2, P2=32 * block ** 2, disp12MaxDiff=2,
        uniquenessRatio=9, speckleWindowSize=48, speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    return matcher.compute(np.ascontiguousarray(left), np.ascontiguousarray(right)).astype(np.float32) / 16.0


def compute_depth(pair: RectifiedPair, stride: int) -> DepthMap:
    left, left_bounds, source_maps = rectified_image(pair.reference, pair.R1, pair.P1)
    right, right_bounds, _ = rectified_image(pair.target, pair.R2, pair.P2)
    contrast = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    left_gray = contrast.apply(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
    right_gray = contrast.apply(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
    right_min = -pair.min_disparity - pair.disparity_count + 1
    if pair.axis == 1:
        forward = disparity(left_gray.T, right_gray.T, pair.min_disparity, pair.disparity_count).T
        reverse = disparity(right_gray.T, left_gray.T, right_min, pair.disparity_count).T
    else:
        forward = disparity(left_gray, right_gray, pair.min_disparity, pair.disparity_count)
        reverse = disparity(right_gray, left_gray, right_min, pair.disparity_count)

    y, x = np.indices(forward.shape, dtype=np.float32)
    mapped_x = x - forward if pair.axis == 0 else x
    mapped_y = y - forward if pair.axis == 1 else y
    right_d = cv2.remap(reverse, mapped_x, mapped_y, cv2.INTER_NEAREST, borderValue=right_min - 1)
    right_g = cv2.remap(right_gray, mapped_x, mapped_y, cv2.INTER_LINEAR)
    right_valid = cv2.remap(right_bounds.astype(np.uint8), mapped_x, mapped_y, cv2.INTER_NEAREST) > 0
    lr_error = np.abs(forward + right_d)
    photo_error = np.abs(left_gray.astype(np.float32) - right_g)
    gray_float = left_gray.astype(np.float32)
    local_var = cv2.blur(gray_float ** 2, (5, 5)) - cv2.blur(gray_float, (5, 5)) ** 2
    valid = (
        left_bounds & right_valid & (forward >= pair.min_disparity) & (right_d >= right_min)
        & (lr_error <= 1.5) & (photo_error <= 32) & (local_var >= 7)
    )
    xyz_rect = cv2.reprojectImageTo3D(forward, pair.Q)
    valid &= np.isfinite(xyz_rect).all(axis=2) & (xyz_rect[:, :, 2] > 0)
    depth = np.where(valid, xyz_rect[:, :, 2], np.nan)
    sample = valid[::stride, ::stride]
    rect_points = xyz_rect[::stride, ::stride][sample]
    world = (rect_points @ pair.R1 - pair.reference.t) @ pair.reference.R
    rgb = left[::stride, ::stride][sample, ::-1]
    width, height = pair.reference.original_size
    image_height, image_width = left.shape[:2]
    uv = np.column_stack((
        source_maps[0][::stride, ::stride][sample] * (width / image_width),
        source_maps[1][::stride, ::stride][sample] * (height / image_height),
    ))
    return DepthMap(pair, depth, world, rgb, uv, lr_error[::stride, ::stride][sample])


def count_depth_agreement(source: DepthMap, maps: list[DepthMap]) -> np.ndarray:
    xyz = source.xyz
    support = np.zeros(len(xyz), dtype=np.uint8)
    original_views = {source.pair.reference.index, source.pair.target.index}
    neighbors = sorted(
        (other for other in maps
         if other is not source
         and {other.pair.reference.index, other.pair.target.index} != original_views),
        key=lambda other: np.linalg.norm(other.pair.reference.center - source.pair.reference.center),
    )[:7]
    for other in neighbors:
        view = other.pair.reference
        rect = (xyz @ view.R.T + view.t) @ other.pair.R1.T
        z = rect[:, 2]
        xy = (rect @ other.pair.P1[:2, :3].T + other.pair.P1[:2, 3]) / np.maximum(z[:, None], 1e-6)
        height, width = other.depth.shape
        valid = (z > 0) & np.all((xy >= 0.5) & (xy < (width - 1.5, height - 1.5)), axis=1)
        px, py = np.round(xy[valid]).astype(np.int32).T
        observed = other.depth[py, px]
        # Relative tolerance handles depth-dependent disparity quantization
        # and the residual uncertainty in the saved, uncalibrated poses.
        support[valid] += np.abs(observed - z[valid]) <= np.maximum(0.035, 0.025 * z[valid])
    return support


def run(
    metrics: Path, keyframes: Path, sparse: Path, output: Path, width: int, stride: int, coverage: str,
    experiment: str = "red_solace_video_rootsift",
) -> None:
    start = time.perf_counter()
    alpha, disparity_limit, radius = COVERAGE_PRESETS[coverage]
    views = load_views(metrics, keyframes, width, experiment)
    sparse_points = load_sparse_points(sparse)
    maps = []
    for i, reference in enumerate(views):
        choices = [
            pair for target in views[max(0, i - radius):i + radius + 1]
            if target.index != reference.index and target.camera_group == reference.camera_group
            if (pair := propose_pair(reference, target, sparse_points, alpha, disparity_limit)) is not None
        ]
        if not choices:
            continue
        pair = max(choices, key=lambda entry: entry.score)
        depth = compute_depth(pair, stride)
        if len(depth.xyz):
            maps.append(depth)
            print(f"Pair {reference.index:02d}/{pair.target.index:02d}: {len(depth.xyz):,} LR samples", flush=True)
    if len(maps) < 2:
        raise RuntimeError("Fewer than two stereo maps supported reliable disparity")

    records = []
    for pair_id, source in enumerate(maps):
        support = count_depth_agreement(source, maps)
        keep = support > 0
        records.append((
            source.xyz[keep].astype(np.float32), source.rgb[keep],
            np.full(np.count_nonzero(keep), source.pair.reference.index, dtype=np.int16),
            np.full(np.count_nonzero(keep), pair_id, dtype=np.int16),
            source.uv[keep], source.lr_error[keep], support[keep],
        ))
    xyz, rgb, view_ids, pair_ids, source_uv, lr_error, support_count = (
        np.concatenate(items) for items in zip(*records)
    )
    if not len(xyz):
        raise RuntimeError("No depth samples agreed with an independent stereo pair")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "method": "Rectified bidirectional SGBM, photometric and cross-view depth checks",
        "coordinates": "Original SfM PLY world coordinates",
        "experiment": experiment,
        "keyframe_names": {view.index: view.name for view in views},
        "matching_width": width,
        "coverage_preset": coverage,
        "rectification_alpha": alpha,
        "max_disparity_span": disparity_limit,
        "pair_radius": radius,
        "pairing_constraint": "Within each saved camera group",
        "sample_stride": stride,
        "pair_count": len(maps),
        "lr_sample_count": sum(len(source.xyz) for source in maps),
        "consistent_sample_count": len(xyz),
        "elapsed_sec": round(time.perf_counter() - start, 1),
    }
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary, xyz=xyz, rgb=rgb, view_ids=view_ids, pair_ids=pair_ids,
        source_uv=source_uv, lr_error=lr_error, support_count=support_count,
        pairs=np.array([(depth.pair.reference.index, depth.pair.target.index) for depth in maps], dtype=np.int16),
        metadata_json=np.array(json.dumps(metadata)),
    )
    temporary.replace(output)
    print(f"{len(xyz):,} consistent pixel stereo samples in {metadata['elapsed_sec']:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=SCENES, default="red-solace")
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--keyframes", type=Path)
    parser.add_argument("--sparse", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--coverage", choices=COVERAGE_PRESETS, default="conservative")
    parser.add_argument("--max-width", type=int, default=960)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    if args.max_width < 320 or args.sample_stride < 1 or args.threads < 1:
        parser.error("Use width >= 320 and positive sample stride and thread count")
    cv2.setNumThreads(args.threads)
    experiment, cache = SCENES[args.scene]
    final = ROOT / "outputs/final" / args.scene
    filename = "stereo_samples_expanded.npz" if args.coverage == "expanded" else "stereo_samples.npz"
    run(
        args.metrics or final / "experiment_metrics.json",
        args.keyframes or ROOT / "outputs" / cache,
        args.sparse or final / "point_clouds" / f"{experiment}_sparse.ply",
        args.output or final / "surface" / filename,
        args.max_width, args.sample_stride, args.coverage, experiment,
    )


if __name__ == "__main__":
    main()
