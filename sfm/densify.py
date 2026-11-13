"""Geometry-checked semi-dense triangulation with fixed sparse camera poses."""

from collections import Counter, defaultdict
from itertools import combinations

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .pipeline import ReconstructionState


def _covisible_pairs(state, registered, neighbor_span, min_covisible_points):
    counts = Counter()
    for point in state.points3d:
        if point.valid:
            cameras = sorted(registered.intersection(point.observations))
            counts.update(combinations(cameras, 2))

    neighbors = defaultdict(list)
    for (i, j), count in counts.items():
        if count >= min_covisible_points:
            neighbors[i].append((count, j))
            neighbors[j].append((count, i))
    pairs = {
        tuple(sorted((i, j)))
        for i, candidates in neighbors.items()
        for _, j in sorted(candidates, key=lambda candidate: (-candidate[0], candidate[1]))[:neighbor_span]
    }
    return sorted(pairs)


def _ratio_neighbors(query, matcher, ratio=0.80):
    indices = np.full(len(query), -1, dtype=np.int32)
    distances = np.full(len(query), np.inf)
    for neighbors in matcher.knnMatch(query, k=2):
        if len(neighbors) == 2:
            best, second = neighbors
            if best.distance < ratio * second.distance:
                indices[best.queryIdx] = best.trainIdx
                distances[best.queryIdx] = best.distance
    return indices, distances


def _verified_matches(i, j, features, inverse_intrinsics, state, reproj_limit, color_limit):
    pts_i, desc_i, rgb_i, matcher_i = features[i]
    pts_j, desc_j, rgb_j, matcher_j = features[j]
    forward, distances = _ratio_neighbors(desc_i, matcher_j)
    backward, _ = _ratio_neighbors(desc_j, matcher_i)
    ki = np.flatnonzero(forward >= 0)
    kj = forward[ki]
    mutual = backward[kj] == ki
    ki, kj = ki[mutual], kj[mutual]
    if not len(ki):
        return []

    relative_rotation = state.cam_R[j] @ state.cam_R[i].T
    tx, ty, tz = (state.cam_t[j].reshape(3) - relative_rotation @ state.cam_t[i].reshape(3))
    baseline = np.linalg.norm([tx, ty, tz])
    if baseline <= np.finfo(float).eps:
        return []
    skew = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]]) / baseline
    fundamental = inverse_intrinsics[j].T @ skew @ relative_rotation @ inverse_intrinsics[i]
    xi = np.column_stack([pts_i[ki], np.ones(len(ki))])
    xj = np.column_stack([pts_j[kj], np.ones(len(kj))])
    lines_j = xi @ fundamental.T
    lines_i = xj @ fundamental
    denominator = np.sum(lines_i[:, :2] ** 2 + lines_j[:, :2] ** 2, axis=1)
    residual = np.sum(xj * lines_j, axis=1)
    color_error = np.linalg.norm(rgb_i[ki] - rgb_j[kj], axis=1)
    good = (residual ** 2 < reproj_limit ** 2 * denominator) & (color_error < color_limit)
    return [
        (float(distances[a]), (i, int(a)), (j, int(b)))
        for a, b in zip(ki[good], kj[good])
    ]


def _build_tracks(edges, min_track_views):
    """Merge the strongest edges first and forbid duplicate observations in a view."""
    parent = {}
    observations = {}

    def find(node):
        root = parent.setdefault(node, node)
        while root != parent[root]:
            root = parent[root]
        while node != root:
            following = parent[node]
            parent[node] = root
            node = following
        return root

    for _, a, b in sorted(edges):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        views_a = observations.setdefault(ra, {ra[0]: ra[1]})
        views_b = observations.setdefault(rb, {rb[0]: rb[1]})
        if not views_a.keys().isdisjoint(views_b):
            continue
        if len(views_a) < len(views_b):
            ra, rb, views_a, views_b = rb, ra, views_b, views_a
        parent[rb] = ra
        views_a.update(views_b)
        del observations[rb]

    return [tuple(sorted(track.items())) for track in observations.values() if len(track) >= min_track_views]


def _triangulate_track(track, features, poses, projections, centers, max_error, min_angle, min_views):
    camera_ids = [camera for camera, _ in track]
    pixels = np.array([features[camera][0][kp] for camera, kp in track])
    colors = np.array([features[camera][2][kp] for camera, kp in track])
    cameras = np.array([projections[camera] for camera in camera_ids])
    extrinsics = np.array([poses[camera] for camera in camera_ids])
    camera_centers = np.array([centers[camera] for camera in camera_ids])
    rows = pixels[:, :, None] * cameras[:, 2:3] - cameras[:, :2]
    rows /= np.maximum(np.linalg.norm(rows, axis=2, keepdims=True), 1e-12)
    keep = np.ones(len(track), dtype=bool)
    for _ in range(3):
        _, _, vectors = np.linalg.svd(rows[keep].reshape(-1, 4), full_matrices=False)
        point = vectors[-1]
        if abs(point[3]) < 1e-10:
            return None
        point = point / point[3]
        image_points = cameras @ point
        depth = extrinsics[:, 2] @ point
        positive = depth > 0.0
        uv = image_points[:, :2] / np.where(positive, depth, 1.0)[:, None]
        errors = np.linalg.norm(uv - pixels, axis=1)
        good = keep & positive & (errors <= max_error)
        if np.count_nonzero(good) < min_views:
            return None
        if np.array_equal(good, keep):
            break
        keep = good
    else:
        return None

    rays = point[:3] - camera_centers[keep]
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
    angle = np.degrees(np.arccos(np.clip(np.min(np.abs(rays @ rays.T)), 0.0, 1.0)))
    if angle < min_angle or not np.all(np.isfinite(point)):
        return None
    return point[:3], np.median(colors[keep], axis=0), np.count_nonzero(keep), np.mean(errors[keep])


def densify_point_cloud(
    state: ReconstructionState,
    max_dense_features: int = 5000,
    neighbor_span: int = 4,
    max_reproj_err: float = 2.5,
    min_angle_deg: float = 1.5,
    max_color_diff: float = 45.0,
    *,
    min_track_views: int = 3,
    min_covisible_points: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Add unique points with consistent feature tracks in at least three cameras."""
    if min_track_views < 3:
        raise ValueError("Semi-dense tracks must span at least three cameras")
    sparse_xyz, sparse_rgb, _ = state.get_valid_points()
    registered = set(state.registered_cams)
    if len(registered) < min_track_views or len(sparse_xyz) < min_covisible_points:
        return sparse_xyz, sparse_rgb

    pairs = _covisible_pairs(state, registered, neighbor_span, min_covisible_points)
    features = {}
    sift = cv2.SIFT_create(nfeatures=max_dense_features, contrastThreshold=0.012, edgeThreshold=12)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for camera in sorted({camera for pair in pairs for camera in pair}):
        image = state.dataset[camera]
        keypoints, desc = sift.detectAndCompute(clahe.apply(image.image_gray), None)
        if desc is None or len(desc) < 2:
            continue
        # SIFT can exceed nfeatures if responses tie at the cutoff.
        keypoints = keypoints[:max_dense_features]
        desc = desc[:max_dense_features]
        if len(desc) < 2:
            continue
        points = np.array([keypoint.pt for keypoint in keypoints])
        desc = np.sqrt(desc / np.maximum(np.sum(desc, axis=1, keepdims=True), 1e-8))
        x = np.clip(np.rint(points[:, 0]).astype(int), 0, image.width - 1)
        y = np.clip(np.rint(points[:, 1]).astype(int), 0, image.height - 1)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=32))
        matcher.add([desc])
        matcher.train()
        features[camera] = (points, desc, image.image_rgb[y, x].astype(np.float32), matcher)

    inverse_intrinsics = {camera: np.linalg.inv(state.cam_K[camera]) for camera in features}
    poses = {
        camera: np.column_stack((state.cam_R[camera], state.cam_t[camera].reshape(3)))
        for camera in features
    }
    projections = {camera: state.cam_K[camera] @ pose for camera, pose in poses.items()}
    centers = {camera: -pose[:, :3].T @ pose[:, 3] for camera, pose in poses.items()}
    edges = []
    for i, j in pairs:
        if i in features and j in features:
            edges.extend(_verified_matches(
                i, j, features, inverse_intrinsics, state, max_reproj_err, max_color_diff
            ))

    accepted = []
    for track in _build_tracks(edges, min_track_views):
        result = _triangulate_track(
            track, features, poses, projections, centers,
            max_reproj_err, min_angle_deg, min_track_views,
        )
        if result is not None:
            accepted.append(result)
    if not accepted:
        return sparse_xyz, sparse_rgb

    origin = np.median(sparse_xyz, axis=0)
    scene_radius = np.percentile(np.linalg.norm(sparse_xyz - origin, axis=1), 95)
    if scene_radius <= np.finfo(float).eps:
        return sparse_xyz, sparse_rgb
    voxel_size = scene_radius / 600.0
    accepted.sort(key=lambda candidate: (-candidate[2], candidate[3]))
    new_xyz = np.array([candidate[0] for candidate in accepted])
    new_rgb = np.array([candidate[1] for candidate in accepted])
    nearest_sparse = cKDTree(sparse_xyz).query(new_xyz, distance_upper_bound=voxel_size)[0]
    # Guard against extreme depths before converting positions to integer voxel coordinates.
    inside = np.linalg.norm(new_xyz - origin, axis=1) <= 3.0 * scene_radius
    keep = inside & (nearest_sparse >= voxel_size)
    new_xyz, new_rgb = new_xyz[keep], new_rgb[keep]
    voxels = np.floor((new_xyz - origin) / voxel_size).astype(np.int64)
    _, unique = np.unique(voxels, axis=0, return_index=True)
    return (
        np.vstack((sparse_xyz, new_xyz[unique])),
        np.vstack((sparse_rgb, np.rint(new_rgb[unique]).astype(np.uint8))),
    )
