"""
Visualization, PLY Point Cloud Export, and Publication-Quality Figure Generation.
"""

from collections import defaultdict
from dataclasses import dataclass
import os
from typing import Dict, Tuple, List, Optional
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .data import ImageData
from .geometry import VerifiedPair
from .pipeline import ReconstructionState


def export_ply(filepath: str, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Exports a colored 3D point cloud (xyz [Nx3], rgb [Nx3 uint8]) to ASCII PLY format."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    n = len(xyz)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            x, y, z = xyz[i]
            r, g, b = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


@dataclass(frozen=True)
class PointCloudRenderFrame:
    center: np.ndarray
    rotation: np.ndarray
    close_extent: np.ndarray
    overview_extent: np.ndarray

    def transform(self, xyz: np.ndarray) -> np.ndarray:
        return (xyz - self.center) @ self.rotation.T


def _camera_centers(state: ReconstructionState) -> Dict[int, np.ndarray]:
    return {
        camera: -state.cam_R[camera].T @ state.cam_t[camera].reshape(3)
        for camera in sorted(state.registered_cams)
    }


def build_point_cloud_render_frame(xyz: np.ndarray, state: ReconstructionState) -> PointCloudRenderFrame:
    """Use sparse geometry to set one orientation and set of limits for both plots."""
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    center = np.median(xyz, axis=0) if len(xyz) else np.zeros(3)
    centered = xyz - center
    cameras = np.array(list(_camera_centers(state).values())).reshape(-1, 3) - center
    rotation = np.eye(3)
    if len(centered) >= 10:
        radii = np.linalg.norm(centered, axis=1)
        core = centered[radii <= np.percentile(radii, 90)]
        _, vectors = np.linalg.eigh(np.cov(core.T))
        rotation = vectors[:, ::-1].T.copy()
        if np.linalg.det(rotation) < 0:
            rotation[2] *= -1
        if len(cameras) and np.median(cameras @ rotation[2]) < 0:
            rotation[1:] *= -1

    aligned = centered @ rotation.T
    if len(aligned):
        close_extent, overview_extent = 1.15 * np.percentile(np.abs(aligned), [90, 99], axis=0)
    else:
        close_extent, overview_extent = np.ones((2, 3))
    close_extent = np.maximum(close_extent, max(np.max(close_extent) * 0.12, 1e-6))
    overview_extent = np.maximum(overview_extent, close_extent)
    if len(cameras):
        camera_extent = 1.08 * np.max(np.abs(cameras @ rotation.T), axis=0)
        overview_extent = np.maximum(overview_extent, camera_extent)
    return PointCloudRenderFrame(center, rotation, close_extent, overview_extent)


def _camera_paths(cameras: Dict[int, np.ndarray], state: ReconstructionState) -> List[np.ndarray]:
    clips = defaultdict(list)
    for camera, center in cameras.items():
        image = state.dataset[camera]
        path = getattr(image, "source_path", None)
        frame_index = getattr(image, "frame_index", None)
        if image.source_type == "video" and path is not None and frame_index is not None:
            clips[path].append((frame_index, camera, center))
    return [
        np.array([center for _, _, center in sorted(clip, key=lambda item: item[:2])])
        for clip in clips.values()
        if len(clip) >= 2
    ]


def _sample_cloud(xyz, colors, extent, max_points=35000):
    indices = np.flatnonzero(np.all(np.abs(xyz) <= extent, axis=1))
    if len(indices) > max_points:
        indices = np.random.default_rng(42).choice(indices, max_points, replace=False)
    return xyz[indices], colors[indices]


def _set_3d_limits(axis, extent):
    axis.set_xlim(-extent[0], extent[0])
    axis.set_ylim(-extent[1], extent[1])
    axis.set_zlim(-extent[2], extent[2])
    # Matplotlib can normalize an array in place. Keep the shared frame intact.
    axis.set_box_aspect(tuple(extent))


def render_point_cloud_views(
    xyz: np.ndarray,
    rgb: np.ndarray,
    state: ReconstructionState,
    save_path: str,
    title: str = "3D Reconstruction & Camera Poses",
    point_size: float = 1.8,
    *,
    render_frame: Optional[PointCloudRenderFrame] = None,
) -> None:
    """Render a four-panel view, tracing camera motion only within individual clips."""
    if len(xyz) == 0:
        return

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    if render_frame is None:
        sparse_xyz, _, _ = state.get_valid_points()
        render_frame = build_point_cloud_render_frame(sparse_xyz if len(sparse_xyz) else xyz, state)
    camera_centers = {
        camera: render_frame.transform(center)
        for camera, center in _camera_centers(state).items()
    }
    cams_al = np.array(list(camera_centers.values())).reshape(-1, 3)
    camera_paths = _camera_paths(camera_centers, state)
    aligned_xyz = render_frame.transform(xyz)
    colors = rgb.astype(np.float32) / 255.0
    overview_extent = render_frame.overview_extent
    close_extent = render_frame.close_extent
    pts, cols = _sample_cloud(aligned_xyz, colors, overview_extent)
    close_pts, close_cols = _sample_cloud(aligned_xyz, colors, close_extent)

    fig = plt.figure(figsize=(15, 11), facecolor="white")
    fig.suptitle(
        f"{title}  ({len(xyz):,} 3D Points, {len(cams_al)}/{len(state.dataset)} Registered Views)",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=cols, s=point_size, alpha=0.85, edgecolors="none")
    if len(cams_al) > 0:
        for path_idx, path in enumerate(camera_paths):
            label = "Video Clip Paths" if path_idx == 0 else None
            ax1.plot(*path.T, "r-", linewidth=1.2, alpha=0.7, label=label)
        ax1.scatter(cams_al[:, 0], cams_al[:, 1], cams_al[:, 2], c="red", marker="^", s=28, edgecolors="black", linewidths=0.4, label="Camera Centers")
        ax1.legend(loc="upper right", fontsize=8)
    _set_3d_limits(ax1, overview_extent)
    ax1.view_init(elev=28, azim=-50)
    ax1.set_title("(a) 3D Reconstruction & Camera Centers", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Panel 2: Oblique 3D Close-Up of Reconstructed Structure
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    ax2.scatter(*close_pts.T, c=close_cols, s=point_size * 1.35, alpha=0.9, edgecolors="none")
    _set_3d_limits(ax2, close_extent)
    ax2.view_init(elev=22, azim=130)
    ax2.set_title("(b) Point Cloud Close-Up (Opposite Viewpoint)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")

    # Panel 3: Top-Down Plan View (XY) with Camera Trajectory
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.scatter(pts[:, 0], pts[:, 1], c=cols, s=point_size * 1.1, alpha=0.85, edgecolors="none")
    if len(cams_al) > 0:
        for path in camera_paths:
            ax3.plot(*path[:, :2].T, "r--", linewidth=1.1, alpha=0.75)
        ax3.scatter(cams_al[:, 0], cams_al[:, 1], c="red", marker="^", s=32, edgecolors="black", linewidths=0.5, zorder=5, label="Registered Cameras")
        ax3.legend(loc="upper right", fontsize=8)
    ax3.set_xlim(-overview_extent[0], overview_extent[0])
    ax3.set_ylim(-overview_extent[1], overview_extent[1])
    ax3.set_aspect("equal", adjustable="box")
    ax3.grid(True, linestyle=":", alpha=0.4)
    ax3.set_title("(c) Top-Down Plan View (X-Y Plane)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X (Principal Axis 1)")
    ax3.set_ylabel("Y (Principal Axis 2)")

    # Panel 4: Side Elevation View (X-Z Plane)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(close_pts[:, 0], close_pts[:, 2], c=close_cols, s=point_size * 1.2, alpha=0.85, edgecolors="none")
    ax4.set_xlim(-close_extent[0], close_extent[0])
    ax4.set_ylim(-close_extent[2], close_extent[2])
    ax4.set_aspect("equal", adjustable="box")
    ax4.grid(True, linestyle=":", alpha=0.4)
    ax4.set_title("(d) Side Elevation Close-Up (X-Z Plane)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("X (Principal Axis 1)")
    ax4.set_ylabel("Z (Principal Axis 3)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=220)
    plt.close(fig)


def plot_feature_matches(
    img1: ImageData,
    img2: ImageData,
    raw_matches: np.ndarray,
    inlier_matches: np.ndarray,
    save_path: str,
    title: str = "Two-View Feature Matching & Epipolar RANSAC Verification",
    max_draw: int = 120,
) -> None:
    """Plots raw ratio-test matches vs. RANSAC epipolar-verified inliers side-by-side."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    def _make_canvas(im_a: np.ndarray, im_b: np.ndarray) -> Tuple[np.ndarray, int]:
        h1, w1 = im_a.shape[:2]
        h2, w2 = im_b.shape[:2]
        target_h = 600
        s1 = target_h / float(h1)
        s2 = target_h / float(h2)
        im_a_r = cv2.resize(im_a, (int(w1 * s1), target_h))
        im_b_r = cv2.resize(im_b, (int(w2 * s2), target_h))
        canvas = np.hstack([im_a_r, im_b_r])
        return canvas, im_a_r.shape[1], s1, s2

    canvas_raw, offset_w, s1, s2 = _make_canvas(img1.image_rgb, img2.image_rgb)
    canvas_inlier = canvas_raw.copy()

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Top: Raw Matches (with outliers highlighted in red, inliers in green)
    inlier_set = set(map(tuple, inlier_matches.tolist()))
    ax = axes[0]
    ax.imshow(canvas_raw)
    step_r = max(1, len(raw_matches) // max_draw)
    for idx in range(0, len(raw_matches), step_r):
        i1, i2 = int(raw_matches[idx, 0]), int(raw_matches[idx, 1])
        u1, v1 = img1.kps[i1] * s1
        u2, v2 = img2.kps[i2] * s2
        is_in = (i1, i2) in inlier_set
        col = "#00e676" if is_in else "#ff1744"
        alpha = 0.65 if is_in else 0.85
        ax.plot([u1, u2 + offset_w], [v1, v2], color=col, linewidth=0.9, alpha=alpha)
    ax.set_title(
        f"(a) Raw Mutual Ratio-Test Matches ({len(raw_matches)} matches: green = epipolar inliers, red = outliers)",
        fontsize=11,
        fontweight="bold",
    )
    ax.axis("off")

    # Bottom: Verified Epipolar Inliers
    ax2 = axes[1]
    ax2.imshow(canvas_inlier)
    step_in = max(1, len(inlier_matches) // max_draw)
    rng = np.random.default_rng(123)
    for idx in range(0, len(inlier_matches), step_in):
        i1, i2 = int(inlier_matches[idx, 0]), int(inlier_matches[idx, 1])
        u1, v1 = img1.kps[i1] * s1
        u2, v2 = img2.kps[i2] * s2
        col = rng.uniform(0.2, 1.0, size=3)
        ax2.plot([u1, u2 + offset_w], [v1, v2], color=col, linewidth=1.0, alpha=0.8)
        ax2.scatter([u1, u2 + offset_w], [v1, v2], color=col, s=10)
    inlier_pct = 100.0 * len(inlier_matches) / max(1, len(raw_matches))
    ax2.set_title(
        f"(b) 5-Point RANSAC Essential Matrix Verified Inliers ({len(inlier_matches)} inliers, {inlier_pct:.1f}% inlier ratio)",
        fontsize=11,
        fontweight="bold",
    )
    ax2.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_reprojection_error_hist(
    errors_before: np.ndarray,
    errors_after: np.ndarray,
    save_path: str,
    title: str = "Reprojection Error Distribution: Before vs. After Bundle Adjustment",
) -> None:
    """Plots overlapping histograms and cumulative distributions of reprojection errors."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    if len(errors_before) == 0 or len(errors_after) == 0:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    max_x = min(5.0, max(np.percentile(errors_before, 99), np.percentile(errors_after, 99)) * 1.1)
    bins = np.linspace(0, max_x, 45)

    ax1.hist(errors_before, bins=bins, density=True, alpha=0.55, color="#ef5350", edgecolor="black", linewidth=0.4,
             label=f"Before Final BA (Mean: {np.mean(errors_before):.2f}px, Med: {np.median(errors_before):.2f}px)")
    ax1.hist(errors_after, bins=bins, density=True, alpha=0.65, color="#1e88e5", edgecolor="black", linewidth=0.4,
             label=f"After Final BA (Mean: {np.mean(errors_after):.2f}px, Med: {np.median(errors_after):.2f}px)")
    ax1.axvline(np.mean(errors_before), color="#c62828", linestyle="--", linewidth=1.5)
    ax1.axvline(np.mean(errors_after), color="#0d47a1", linestyle="-", linewidth=1.5)
    ax1.set_xlabel("Reprojection Error (pixels)")
    ax1.set_ylabel("Probability Density")
    ax1.set_title("(a) Error Histogram", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Empirical CDF
    sb = np.sort(errors_before)
    sa = np.sort(errors_after)
    ax2.plot(sb, np.linspace(0, 100, len(sb)), color="#e53935", linestyle="--", linewidth=2.0, label="Without / Pre-BA")
    ax2.plot(sa, np.linspace(0, 100, len(sa)), color="#1e88e5", linestyle="-", linewidth=2.0, label="With Sparse BA")
    ax2.set_xlim(0, max_x)
    ax2.set_ylim(0, 102)
    ax2.set_xlabel("Reprojection Error Threshold (pixels)")
    ax2.set_ylabel("Cumulative Percentage of Observations (%)")
    ax2.set_title("(b) Cumulative Error Distribution (CDF)", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_scene_graph(
    n_images: int,
    verified_pairs: Dict[Tuple[int, int], VerifiedPair],
    state: ReconstructionState,
    save_path: str,
    title: str = "Scene Connectivity Matrix & Track Length Distribution",
) -> None:
    """Plots the pairwise verified inlier match matrix and 3D point track length histogram."""
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    mat = np.zeros((n_images, n_images), dtype=np.float32)
    for (i, j), vp in verified_pairs.items():
        mat[i, j] = len(vp.matches)
        mat[j, i] = len(vp.matches)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    im = ax1.imshow(np.log10(mat + 1.0), cmap="magma", interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("log10(Verified Inliers + 1)")
    ax1.set_title(f"(a) Verified Epipolar Adjacency Matrix ({len(verified_pairs)} edges)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("View Index j")
    ax1.set_ylabel("View Index i")

    _, _, valid_ids = state.get_valid_points()
    track_lens = [len(state.points3d[pid].observations) for pid in valid_ids]
    if track_lens:
        max_t = min(25, max(track_lens) + 1)
        bins = np.arange(2, max_t + 2) - 0.5
        ax2.hist(np.clip(track_lens, 2, max_t), bins=bins, color="#26a69a", edgecolor="black", alpha=0.85)
        ax2.axvline(np.mean(track_lens), color="#d81b60", linestyle="--", linewidth=1.6,
                    label=f"Mean Track Length: {np.mean(track_lens):.2f} views")
        ax2.legend(fontsize=9)
    ax2.set_title("(b) Multi-View 3D Point Track Length Distribution", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Number of Observing Cameras per 3D Point")
    ax2.set_ylabel("Number of 3D Points")
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
