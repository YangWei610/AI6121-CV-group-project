"""
Sparse Bundle Adjustment (BA) via Trust-Region Reflective Non-Linear Least Squares
with Explicit Sparse Jacobian Structure and Huber Loss.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import least_squares
import cv2


def _rodrigues_rotate_vectorized(rvecs: np.ndarray, pts_3d: np.ndarray) -> np.ndarray:
    """
    Vectorized Rodrigues rotation of 3D points `pts_3d` (Nx3) by axis-angle vectors `rvecs` (Nx3).
    """
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    small_mask = (theta.ravel() < 1e-8)
    safe_theta = np.where(theta < 1e-8, 1.0, theta)
    k = rvecs / safe_theta

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    k_cross_x = np.cross(k, pts_3d)
    k_dot_x = np.sum(k * pts_3d, axis=1, keepdims=True)

    rotated = pts_3d * cos_t + k_cross_x * sin_t + k * k_dot_x * (1.0 - cos_t)
    if np.any(small_mask):
        rotated[small_mask] = pts_3d[small_mask] + np.cross(rvecs[small_mask], pts_3d[small_mask])
    return rotated



def _jacobian_pattern(pose_columns, point_start, n_pts, group_columns, obs_cam_idx, obs_pt_idx):
    n_obs = len(obs_cam_idx)
    obs_rows = np.arange(n_obs)
    rows, columns = [], []

    def add_obs(column):
        keep = column >= 0
        rows.extend((2 * obs_rows[keep], 2 * obs_rows[keep] + 1))
        columns.extend((column[keep], column[keep]))

    for dimension in range(6):
        add_obs(pose_columns[obs_cam_idx, dimension])
    for dimension in range(3):
        add_obs(point_start + 3 * obs_pt_idx + dimension)

    group_start = point_start + 3 * n_pts
    group_ids = np.unique(group_columns[group_columns >= 0])
    add_obs(group_columns[obs_cam_idx])
    rows.append(2 * n_obs + np.arange(len(group_ids)))
    columns.append(group_ids)
    r = np.concatenate(rows)
    c = np.concatenate(columns)
    return csr_matrix(
        (np.ones(len(r), dtype=np.int8), (r, c)),
        shape=(2 * n_obs + len(group_ids), group_start + len(group_ids)),
    )


def run_bundle_adjustment(
    cam_R_list: List[np.ndarray],
    cam_t_list: List[np.ndarray],
    cam_K_list: List[np.ndarray],
    pts_3d: np.ndarray,
    obs_cam_idx: np.ndarray,
    obs_pt_idx: np.ndarray,
    obs_pts_2d: np.ndarray,
    optimize_focal: bool = False,
    max_nfev: int = 35,
    f_scale: float = 2.0,
    intrinsics_groups: Optional[List[str]] = None,
    fixed_intrinsics: Optional[List[bool]] = None,
    reference_K: Optional[List[np.ndarray]] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, Dict[str, float]]:
    """Refine poses and points with fixed scale and bounded shared focal factors."""
    n_cams = len(cam_R_list)
    n_pts = len(pts_3d)
    unchanged = cam_R_list, cam_t_list, cam_K_list, pts_3d
    if n_cams < 2 or n_pts < 5 or len(obs_cam_idx) < 15:
        return (*unchanged, {})

    poses = np.array([
        np.r_[cv2.Rodrigues(R)[0].ravel(), t.ravel()]
        for R, t in zip(cam_R_list, cam_t_list)
    ])
    observed = np.bincount(obs_cam_idx, minlength=n_cams) > 0
    free_pose = np.broadcast_to(observed[:, None], poses.shape).copy()
    free_pose[0] = False
    # The seed translation has unit length. Pin its largest component so BA
    # cannot reduce its residuals by drifting in the similarity-scale gauge.
    scale_coordinate = 3 + int(np.argmax(np.abs(poses[1, 3:6])))
    free_pose[1, scale_coordinate] = False
    pose_columns = np.full(poses.shape, -1, dtype=np.int64)
    n_pose = int(free_pose.sum())
    pose_columns[free_pose] = np.arange(n_pose)

    current_K = np.asarray(cam_K_list)
    base_K = current_K.copy()
    focal_groups = intrinsics_groups if intrinsics_groups is not None else [str(i) for i in range(n_cams)]
    locked = np.asarray(fixed_intrinsics if fixed_intrinsics is not None else [False] * n_cams)
    focal_members = {}
    for cid, group in enumerate(focal_groups):
        focal_members.setdefault(group, []).append(cid)
    group_columns = np.full(n_cams, -1, dtype=np.int64)
    group_ids = np.full(n_cams, -1, dtype=np.int64)
    initial_scales = []
    priors = []
    focal_start = n_pose + n_pts * 3
    for members in focal_members.values():
        if not optimize_focal or np.any(locked[members]) or np.count_nonzero(observed[members]) < 3:
            continue
        if reference_K is not None:
            base_K[members] = np.asarray(reference_K)[members]
        group_idx = len(initial_scales)
        group_ids[members] = group_idx
        group_columns[members] = focal_start + group_idx
        relative = current_K[members, 0, 0] / base_K[members, 0, 0]
        initial_scales.append(float(np.clip(np.median(relative), 0.70, 1.40)))
        # Regularization makes gradual shared refinement preferable to focal drift.
        n_group_obs = np.count_nonzero(np.isin(obs_cam_idx, members))
        priors.append(max(10.0, 0.6 * np.sqrt(n_group_obs)))

    x0 = np.r_[poses[free_pose], pts_3d.ravel(), initial_scales]
    prior_weight = np.asarray(priors)
    obs_K = base_K[obs_cam_idx]
    obs_groups = group_ids[obs_cam_idx]
    refine_mask = obs_groups >= 0

    def unpack(parameters):
        adjusted = poses.copy()
        adjusted[free_pose] = parameters[:n_pose]
        points = parameters[n_pose:focal_start].reshape(-1, 3)
        scales = parameters[focal_start:]
        return adjusted, points, scales

    def project(parameters):
        adjusted, points, scales = unpack(parameters)
        camera_pose = adjusted[obs_cam_idx]
        x = _rodrigues_rotate_vectorized(camera_pose[:, :3], points[obs_pt_idx]) + camera_pose[:, 3:]
        z = np.maximum(x[:, 2], 1e-6)
        focal = np.ones(len(obs_cam_idx))
        focal[refine_mask] = scales[obs_groups[refine_mask]]
        projected = (
            x[:, :2] / z[:, None] * obs_K[:, (0, 1), (0, 1)] * focal[:, None]
            + obs_K[:, :2, 2]
        )
        return projected - obs_pts_2d, x[:, 2], scales

    def residual(parameters):
        errors, _, scales = project(parameters)
        return np.r_[errors.ravel(), prior_weight * (scales - 1.0)]

    def robust_cost(errors):
        scaled = np.abs(errors)
        return float(np.sum(np.where(scaled <= f_scale, scaled ** 2, 2 * f_scale * scaled - f_scale ** 2)))

    lower = np.full(len(x0), -np.inf)
    upper = np.full(len(x0), np.inf)
    lower[focal_start:], upper[focal_start:] = 0.70, 1.40
    before, old_depth, _ = project(x0)
    result = least_squares(
        residual, x0,
        jac_sparsity=_jacobian_pattern(pose_columns, n_pose, n_pts, group_columns, obs_cam_idx, obs_pt_idx),
        bounds=(lower, upper),
        x_scale="jac",
        method="trf",
        loss="huber",
        f_scale=f_scale,
        ftol=1e-4,
        xtol=1e-5,
        tr_options={"maxiter": 90},
        max_nfev=max_nfev,
    )
    after, depth, scales = project(result.x)
    before_error = np.linalg.norm(before, axis=1)
    after_error = np.linalg.norm(after, axis=1)
    accepted = (
        np.all(np.isfinite(result.x))
        and robust_cost(residual(result.x)) < robust_cost(residual(x0)) - 1e-6
        and np.count_nonzero(depth <= 0) <= np.count_nonzero(old_depth <= 0)
    )
    stats = {
        "mean_reproj_before": float(before_error.mean()),
        "mean_reproj_after": float(after_error.mean()),
        "median_reproj_before": float(np.median(before_error)),
        "median_reproj_after": float(np.median(after_error)),
        "rmse_before": float(np.sqrt(np.mean(before_error ** 2))),
        "rmse_after": float(np.sqrt(np.mean(after_error ** 2))),
        "nfev": int(result.nfev),
        "status": int(result.status),
        "accepted": bool(accepted),
        "num_optimized_points": n_pts,
        "num_focal_groups": len(scales),
    }
    if not accepted:
        return (*unchanged, stats)

    final_poses, points, scales = unpack(result.x)
    rotations, translations, intrinsics = [], [], []
    for cid, pose in enumerate(final_poses):
        rotations.append(cv2.Rodrigues(pose[:3])[0])
        translations.append(pose[3:].reshape(3, 1))
        K = base_K[cid].copy()
        group = group_ids[cid]
        if group >= 0:
            K[0, 0] *= scales[group]
            K[1, 1] *= scales[group]
        intrinsics.append(K)
    return rotations, translations, intrinsics, points, stats
