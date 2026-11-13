"""Draw white-background case study figures from saved SfM poses and stereo samples."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import trimesh

from sfm.visualization import build_point_cloud_render_frame
from tools.fern_surface_patch import subject_samples
from tools.red_solace_surface import foreground_mask, read_cameras

ROOT = Path("outputs")
SPECS = {
    "fern": ("Fern", "fern_video_rootsift", ROOT / "validation/fern_archive",
             "fern_same_50_frames_sparse.ply", (0, 24)),
    "red-solace": ("Red Solace", "red_solace_video_rootsift", ROOT / "final/red-solace",
                   "point_clouds/red_solace_video_rootsift_sparse.ply", (-18, 25)),
}


def projection(xyz, azimuth, elevation):
    a, e = np.deg2rad([azimuth, elevation])
    x, y, z = xyz.T
    u = np.cos(a) * x - np.sin(a) * y
    v = np.sin(a) * x + np.cos(a) * y
    return np.column_stack((u, np.cos(e) * z + np.sin(e) * v,
                            np.sin(e) * z - np.cos(e) * v))


def frusta(state, span):
    centers = []
    edges = []
    corner_links = [(0, 1), (0, 2), (0, 3), (0, 4),
                    (1, 2), (2, 3), (3, 4), (4, 1)]
    for index in state.registered_cams:
        R, t, K = state.cam_R[index], state.cam_t[index].ravel(), state.cam_K[index]
        center = -R.T @ t
        width, height = 2 * K[:2, 2]
        pixels = np.array([[0, 0, 1], [width, 0, 1],
                           [width, height, 1], [0, height, 1]])
        rays = pixels @ np.linalg.inv(K).T
        corners = center + span * (rays @ R)
        nodes = np.vstack((center, corners))
        centers.append(center)
        edges.extend(nodes[[i, j]] for i, j in corner_links)
    return np.asarray(centers), np.asarray(edges)


def paint(points, colors, transform, azimuth, elevation, output, *, subject=None,
          cameras=None, camera_clips=None):
    fig = plt.figure(figsize=(9.6, 6.15), facecolor="white", dpi=240)
    axis = fig.add_axes((0.01, 0.115, 0.98, 0.875) if camera_clips is not None
                        else (0.01, 0.01, 0.98, 0.98))
    coords = projection(transform(points), azimuth, elevation)
    if subject is None:
        # Keep a deterministic sample of supported pixels for a readable raster.
        if len(coords) > 250_000:
            selection = np.random.default_rng(2026).choice(len(coords), 250_000, replace=False)
            coords, colors = coords[selection], colors[selection]
        order = np.argsort(coords[:, 2])
        size = 2.45 if len(coords) < 25_000 else 0.63
        axis.scatter(*coords[order, :2].T, s=size, c=colors[order] / 255.0,
                     alpha=0.91, linewidths=0)
        limits = np.percentile(coords[:, :2], [0.15, 99.85], axis=0)
    else:
        order = np.argsort(coords[:, 2])
        rgba = np.column_stack((colors / 255.0, np.where(subject, 0.88, 0.15)))
        axis.scatter(*coords[order, :2].T, s=np.where(subject[order], 1.65, 0.52),
                     c=rgba[order], linewidths=0)
        limits = np.percentile(coords[subject, :2], [0.15, 99.85], axis=0)
        centers, edges = cameras
        lines = projection(transform(edges.reshape(-1, 3)), azimuth, elevation).reshape(-1, 2, 3)
        center_xy = projection(transform(centers), azimuth, elevation)[:, :2]
        if camera_clips is None:
            axis.add_collection(LineCollection(lines[:, :, :2], colors="#e53c2d", linewidths=0.96,
                                               alpha=0.88, zorder=3, capstyle="round"))
            axis.scatter(*center_xy.T, c="#dc3023", s=1.9, linewidths=0, zorder=4)
        else:
            palette = {
                "001": "#1761ab", "002": "#008064", "003": "#d43225",
                "004": "#7b419e", "005": "#b66c12", "006": "#40596c",
            }
            handles = []
            clips = np.asarray(camera_clips)
            per_camera = lines.reshape(len(centers), -1, 2, 3)
            for clip in sorted(set(camera_clips)):
                keep = clips == clip
                color = palette[clip]
                segments = per_camera[keep, :, :, :2].reshape(-1, 2, 2)
                axis.add_collection(LineCollection(segments, colors=color, linewidths=0.95,
                                                   alpha=0.93, zorder=3, capstyle="round"))
                axis.scatter(*center_xy[keep].T, c=color, s=1.8, linewidths=0, zorder=4)
                handles.append(Line2D([], [], color=color, lw=1.7,
                                      label=f"{clip}  ({int(keep.sum())})"))
            fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.035),
                       ncol=min(len(handles), 6), frameon=False, fontsize=9,
                       handlelength=1.5, columnspacing=1.3)
            fig.text(0.5, 0.026, "Video clip and registered frames",
                     ha="center", va="top", fontsize=8, color="#42556b")
        limits[0] = np.minimum(limits[0], lines[:, :, :2].min(axis=(0, 1)))
        limits[1] = np.maximum(limits[1], lines[:, :, :2].max(axis=(0, 1)))

    # Fit the sculpture and all registered camera outlines with equal scale.
    mid = limits.mean(axis=0)
    span_x, span_y = (limits[1] - limits[0]) * 1.11
    ratio = (9.6 * 0.98) / (6.15 * (0.875 if camera_clips is not None else 0.98))
    half_y = max(span_y, span_x / ratio) * 0.5
    half_x = half_y * ratio
    axis.set(xlim=(mid[0] - half_x, mid[0] + half_x),
             ylim=(mid[1] - half_y, mid[1] + half_y), aspect="equal")
    axis.set_axis_off()
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


def render_scene(scene, output, azimuth=None, elevation=None):
    name, experiment, base, ply, dense_view = SPECS[scene]
    views, state = read_cameras(base / "experiment_metrics.json", experiment)
    cloud = trimesh.load(base / ply, process=False)
    sparse, colors = cloud.vertices, cloud.colors[:, :3]
    frame = build_point_cloud_render_frame(sparse, state)
    surface = ROOT / "final" / scene / "surface"
    info = json.loads((surface / "surface_metrics.json").read_text(encoding="utf-8"))
    display = info["display_frame"]
    center, axes = np.asarray(display["center"]), np.asarray(display["rotation"])
    def transform(xyz):
        return (xyz - center) @ axes.T

    if scene == "fern":
        masks = json.loads((base / "silhouettes.json").read_text(encoding="utf-8"))
        def foreground(xyz):
            return subject_samples(xyz, frame, views, state, masks)
        sample_path = base / "stereo_samples_expanded.npz"
        sparse_view = (15, 35)
    else:
        def foreground(xyz):
            return foreground_mask(frame.transform(xyz), frame, views, state, True)
        sample_path = surface / "stereo_samples_expanded.npz"
        sparse_view = (-58, 55)

    subject = foreground(sparse)
    aligned = transform(sparse[subject])
    object_span = np.linalg.norm(np.percentile(aligned, 95, axis=0) -
                                 np.percentile(aligned, 5, axis=0))
    cameras = frusta(state, 0.088 * object_span)
    a, e = sparse_view
    paint(sparse, colors, transform, a if azimuth is None else azimuth,
          e if elevation is None else elevation, output / f"{scene}-sparse-cameras.png",
          subject=subject, cameras=cameras)

    with np.load(sample_path, allow_pickle=False) as data:
        samples, sample_rgb = data["xyz"], data["rgb"]
        if scene == "fern":
            observed = data["view_ids"] == info["reference_view"]
            samples, sample_rgb = samples[observed], sample_rgb[observed]
        selected = foreground(samples)
    paint(samples[selected], sample_rgb[selected], transform, *dense_view,
          output / f"{scene}-dense-cloud.png")
    print(f"{name}: {len(sparse):,} sparse points, {len(state.registered_cams)} camera poses, "
          f"{np.count_nonzero(selected):,} foreground stereo samples")


def render_red_solace_clips(poses, output, azimuth, elevation):
    _, experiment, base, ply, _ = SPECS["red-solace"]
    base_views, base_state = read_cameras(base / "experiment_metrics.json", experiment)
    views, state = read_cameras(poses, experiment)
    cloud = trimesh.load(base / ply, process=False)
    sparse = cloud.vertices
    frame = build_point_cloud_render_frame(sparse, base_state)
    subject = foreground_mask(frame.transform(sparse), frame, base_views, base_state, True)
    display = json.loads((base / "surface/surface_metrics.json").read_text(encoding="utf-8"))
    center = np.asarray(display["display_frame"]["center"])
    axes = np.asarray(display["display_frame"]["rotation"])

    def transform(xyz):
        return (xyz - center) @ axes.T

    clips = [Path(str(views[i]["clip"])).stem.zfill(3) for i in state.registered_cams]
    if len(set(clips)) < 2:
        raise ValueError("At least two registered video clips are needed")
    aligned = transform(sparse[subject])
    extent = np.percentile(aligned, [5, 95], axis=0)
    span = 0.081 * np.linalg.norm(extent[1] - extent[0])
    paint(sparse, cloud.colors[:, :3], transform, azimuth, elevation,
          output / "red-solace-multi-video-cameras.png", subject=subject,
          cameras=frusta(state, span), camera_clips=clips)
    print(f"Red Solace: {len(state.registered_cams)} camera poses across {len(set(clips))} video clips")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(SPECS))
    parser.add_argument("--output", type=Path, default=ROOT / "case-study")
    parser.add_argument("--azimuth", type=float)
    parser.add_argument("--elevation", type=float)
    parser.add_argument("--multi-video-poses", type=Path,
                        help="Render the registered Red Solace clips together")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.multi_video_poses:
        render_red_solace_clips(args.multi_video_poses, args.output,
                               -58 if args.azimuth is None else args.azimuth,
                               55 if args.elevation is None else args.elevation)
        return
    for scene in (args.scene,) if args.scene else SPECS:
        render_scene(scene, args.output, args.azimuth, args.elevation)


if __name__ == "__main__":
    main()
