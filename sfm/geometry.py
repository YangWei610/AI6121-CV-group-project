"""
Two-View Epipolar Geometry, Normalized Essential Matrix Estimation,
Linear DLT Triangulation, Cheirality, and PnP Pose Estimation.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import cv2
import numpy as np
from .data import ImageData


@dataclass
class VerifiedPair:
    """Stores geometrically verified two-view matches and relative pose."""
    i: int
    j: int
    matches: np.ndarray  # Geometrically verified 2D track edges
    E: Optional[np.ndarray]  # Essential model, if the pair supports a seed
    R: Optional[np.ndarray]  # Relative pose of a valid seed
    t: Optional[np.ndarray]
    median_angle_deg: float  # Median triangulation parallax angle in degrees
    inlier_ratio: float
    # None preserves construction of already-verified seed pairs by callers.
    seed_matches: Optional[np.ndarray] = None
    homography_ratio: float = 0.0
    coverage: float = 0.0


def image_coverage(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return float(cv2.contourArea(hull) / max(width * height, 1))


def normalize_keypoints(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Maps 2D pixel coordinates (u, v) to normalized camera coordinates:
        x_norm = (u - cx) / fx,  y_norm = (v - cy) / fy
    This enables exact Essential matrix estimation even when two cameras have
    completely different image resolutions or portrait/landscape aspect ratios.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    norm_pts = np.empty_like(pts, dtype=np.float64)
    norm_pts[:, 0] = (pts[:, 0] - cx) / fx
    norm_pts[:, 1] = (pts[:, 1] - cy) / fy
    return norm_pts


def triangulate_dlt(
    P1: np.ndarray,
    P2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
) -> np.ndarray:
    """
    Explicit Linear Triangulation via Direct Linear Transformation (DLT).
    For each correspondence (u1, v1) <-> (u2, v2), constructs A X = 0 where:
        A = [ u1 * P1[2,:] - P1[0,:] ]
            [ v1 * P1[2,:] - P1[1,:] ]
            [ u2 * P2[2,:] - P2[0,:] ]
            [ v2 * P2[2,:] - P2[1,:] ]
    and solves via SVD (vectorized over all N points).
    """
    n = len(pts1)
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Construct batch of 4x4 matrices A for all N points simultaneously
    u1, v1 = pts1[:, 0:1], pts1[:, 1:2]
    u2, v2 = pts2[:, 0:1], pts2[:, 1:2]

    row0 = u1 * P1[2:3, :] - P1[0:1, :]
    row1 = v1 * P1[2:3, :] - P1[1:2, :]
    row2 = u2 * P2[2:3, :] - P2[0:1, :]
    row3 = v2 * P2[2:3, :] - P2[1:2, :]

    A = np.stack([row0, row1, row2, row3], axis=1)  # Nx4x4
    # Normalize rows of A for numerical stability
    row_norms = np.linalg.norm(A, axis=2, keepdims=True) + 1e-12
    A = A / row_norms

    _, _, Vt = np.linalg.svd(A)
    X_hom = Vt[:, -1, :]  # Nx4
    X_3d = X_hom[:, :3] / (X_hom[:, 3:4] + 1e-12)
    return X_3d


def compute_triangulation_angles(
    pts_3d: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
) -> np.ndarray:
    """
    Computes the ray parallax angle (in degrees) between optical centers C1 and C2
    for each 3D point X:
        cos(theta) = (X - C1) . (X - C2) / (||X - C1|| ||X - C2||)
    """
    if len(pts_3d) == 0:
        return np.zeros((0,), dtype=np.float64)
    ray1 = pts_3d - C1.reshape(1, 3)
    ray2 = pts_3d - C2.reshape(1, 3)
    n1 = np.linalg.norm(ray1, axis=1) + 1e-12
    n2 = np.linalg.norm(ray2, axis=1) + 1e-12
    cos_theta = np.sum(ray1 * ray2, axis=1) / (n1 * n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def project_points(
    pts_3d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D world points X into camera frame (R, t) with intrinsics K:
        X_cam = R @ X + t
        u = fx * (X_cam_x / X_cam_z) + cx,  v = fy * (X_cam_y / X_cam_z) + cy
    Returns (pts_2d [Nx2], depths_z [N]).
    """
    X_cam = (R @ pts_3d.T).T + t.reshape(1, 3)
    depths = X_cam[:, 2]
    z_safe = np.where(np.abs(depths) < 1e-8, 1e-8, depths)
    x_norm = X_cam[:, 0] / z_safe
    y_norm = X_cam[:, 1] / z_safe
    u = K[0, 0] * x_norm + K[0, 2]
    v = K[1, 1] * y_norm + K[1, 2]
    return np.stack([u, v], axis=1), depths


def compute_reprojection_errors(
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (pixel_errors [N], depths [N]) for 3D points observed at pts_2d."""
    proj_2d, depths = project_points(pts_3d, K, R, t)
    errs = np.linalg.norm(proj_2d - pts_2d, axis=1)
    return errs, depths


def triangulate_two_views(
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    pts1: np.ndarray,
    pts2: np.ndarray,
    max_reproj_err: float = 4.0,
    min_angle_deg: float = 1.2,
    max_depth_ratio: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangulates 2D correspondences between two calibrated cameras using DLT,
    and filters by cheirality (positive depth in both cameras), parallax angle,
    and reprojection error.
    Returns (pts_3d, valid_mask, angles_deg).
    """
    # Use normalized coordinates for numerically well-conditioned DLT
    P1_norm = np.hstack([R1, t1.reshape(3, 1)])
    P2_norm = np.hstack([R2, t2.reshape(3, 1)])
    pts1_norm = normalize_keypoints(pts1, K1)
    pts2_norm = normalize_keypoints(pts2, K2)

    pts_3d = triangulate_dlt(P1_norm, P2_norm, pts1_norm, pts2_norm)

    err1, z1 = compute_reprojection_errors(pts_3d, pts1, K1, R1, t1)
    err2, z2 = compute_reprojection_errors(pts_3d, pts2, K2, R2, t2)

    C1 = -R1.T @ t1.reshape(3, 1)
    C2 = -R2.T @ t2.reshape(3, 1)
    angles = compute_triangulation_angles(pts_3d, C1.ravel(), C2.ravel())

    cheirality = (z1 > 0.05) & (z2 > 0.05)
    reproj_ok = (err1 < max_reproj_err) & (err2 < max_reproj_err)
    angle_ok = (angles >= min_angle_deg) & (angles <= 75.0)
    finite_ok = np.all(np.isfinite(pts_3d), axis=1)

    valid = cheirality & reproj_ok & angle_ok & finite_ok
    if np.sum(valid) > 10:
        med_d1 = np.median(z1[valid])
        valid = valid & (z1 < max_depth_ratio * med_d1) & (z2 < max_depth_ratio * med_d1)

    return pts_3d, valid, angles


def verify_two_view_geometry(
    dataset: List[ImageData],
    raw_matches: Dict[Tuple[int, int], np.ndarray],
    ransac_px_thresh: float = 2.5,
    min_inliers: int = 24,
) -> Dict[Tuple[int, int], VerifiedPair]:
    """Keep reliable 2D edges even when a pair cannot initialize depth."""
    verified_pairs: Dict[Tuple[int, int], VerifiedPair] = {}
    eye_K = np.eye(3, dtype=np.float64)

    for (i, j), matches in raw_matches.items():
        if len(matches) < max(8, min_inliers):
            continue

        img_i, img_j = dataset[i], dataset[j]
        pts_i = img_i.kps[matches[:, 0]]
        pts_j = img_j.kps[matches[:, 1]]

        _, mask_f = cv2.findFundamentalMat(
            pts_i, pts_j, cv2.USAC_MAGSAC, ransac_px_thresh, 0.999, 2000
        )
        _, mask_h = cv2.findHomography(
            pts_i, pts_j, cv2.USAC_MAGSAC, ransac_px_thresh,
            maxIters=1000, confidence=0.999,
        )
        f_mask = mask_f.ravel() != 0 if mask_f is not None else np.zeros(len(matches), dtype=bool)
        h_mask = mask_h.ravel() != 0 if mask_h is not None else np.zeros(len(matches), dtype=bool)
        n_f, n_h = int(f_mask.sum()), int(h_mask.sum())
        track_mask = h_mask if n_h > n_f else f_mask
        n_track = int(track_mask.sum())
        if n_track < min_inliers or n_track < 0.15 * len(matches):
            continue

        pair = VerifiedPair(
            i=i,
            j=j,
            matches=matches[track_mask],
            E=None,
            R=None,
            t=None,
            median_angle_deg=0.0,
            inlier_ratio=n_track / len(matches),
            homography_ratio=n_h / max(n_f, n_h, 1),
        )
        verified_pairs[(i, j)] = pair
        if n_f < min_inliers or pair.homography_ratio >= 0.88:
            continue

        # A narrow or homography-dominated pair can be useful for PnP and tracks,
        # but is a poor choice to define the scale and depth of the reconstruction.
        f_matches = matches[f_mask]
        x_i = normalize_keypoints(pts_i[f_mask], img_i.K)
        x_j = normalize_keypoints(pts_j[f_mask], img_j.K)
        mean_focal = np.sqrt(img_i.K[0, 0] * img_j.K[0, 0])
        E, mask_e = cv2.findEssentialMat(
            x_i, x_j, cameraMatrix=eye_K, method=cv2.RANSAC,
            prob=0.999, threshold=ransac_px_thresh / mean_focal,
        )
        if E is None or mask_e is None or E.shape != (3, 3):
            continue
        inliers = mask_e.ravel() != 0
        if np.count_nonzero(inliers) < min_inliers:
            continue
        _, R, t, pose_mask, _ = cv2.recoverPose(
            E, x_i[inliers], x_j[inliers], eye_K, distanceThresh=100.0,
        )
        pose_matches = f_matches[inliers][pose_mask.ravel() != 0]
        if len(pose_matches) < min_inliers:
            continue
        _, valid, angles = triangulate_two_views(
            img_i.K, eye_K, np.zeros((3, 1)), img_j.K, R, t,
            img_i.kps[pose_matches[:, 0]], img_j.kps[pose_matches[:, 1]],
            max_reproj_err=ransac_px_thresh * 1.5,
            min_angle_deg=1.25,
        )
        if np.count_nonzero(valid) < min_inliers:
            continue
        seed_matches = pose_matches[valid]
        coverage = min(
            image_coverage(img_i.kps[seed_matches[:, 0]], img_i.width, img_i.height),
            image_coverage(img_j.kps[seed_matches[:, 1]], img_j.width, img_j.height),
        )
        angle = float(np.median(angles[valid]))
        if coverage < 0.008 or angle < 1.75 or angle > 50:
            continue
        pair.E, pair.R, pair.t = E, R, t.reshape(3, 1)
        pair.seed_matches = seed_matches
        pair.median_angle_deg = angle
        pair.coverage = coverage

    return verified_pairs
