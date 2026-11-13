"""Fuse Red Solace stereo depth into an observed, open triangle surface."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes
import trimesh

from sfm.pipeline import ReconstructionState
from sfm.visualization import build_point_cloud_render_frame


FINAL = Path("outputs/final/red-solace")
EXPERIMENT = "red_solace_video_rootsift"
ROI_MIN = np.array([-3.65, -1.02, -1.10])
ROI_MAX = np.array([3.98, 1.68, 0.43])
THUMBNAIL_SIZE = (432, 243)
# Hand-traced object silhouettes in thumbnail pixels, keyed by source frame number.
SILHOUETTES = {
    0: [
        (51, 113), (95, 110), (350, 108), (362, 120), (352, 143), (333, 157),
        (304, 164), (251, 170), (190, 169), (142, 163), (94, 151), (61, 132),
    ],
    168: [
        (14, 95), (100, 91), (236, 92), (380, 88), (409, 98), (415, 123),
        (404, 148), (383, 162), (345, 172), (297, 177), (171, 178), (94, 169),
        (48, 148), (24, 127), (11, 108),
    ],
    464: [
        (91, 169), (125, 136), (185, 97), (250, 50), (296, 27), (333, 21),
        (351, 30), (365, 53), (367, 80), (357, 115), (338, 147), (307, 172),
        (265, 190), (212, 201), (152, 195), (100, 186),
    ],
    988: [
        (0, 115), (4, 97), (27, 87), (66, 85), (174, 90), (260, 94), (343, 100),
        (421, 104), (432, 111), (431, 143), (399, 162), (348, 177), (260, 187),
        (169, 185), (100, 178), (48, 162), (8, 136),
    ],
    1448: [
        (31, 131), (56, 110), (105, 89), (179, 63), (254, 44), (335, 29),
        (382, 21), (410, 35), (425, 58), (420, 85), (402, 115), (374, 145),
        (336, 172), (282, 196), (230, 209), (177, 213), (130, 211), (86, 193),
        (52, 171), (33, 154),
    ],
    1611: [
        (78, 189), (114, 147), (148, 102), (185, 57), (220, 20), (242, 9),
        (260, 8), (279, 14), (291, 29), (301, 52), (298, 91), (287, 132),
        (265, 173), (243, 204), (216, 213), (177, 211), (133, 204), (83, 194),
    ],
    1775: [
        (132, 28), (156, 11), (177, 9), (201, 19), (230, 41), (269, 84),
        (330, 138), (376, 188), (403, 223), (387, 236), (316, 242), (245, 242),
        (221, 233), (183, 203), (158, 169), (137, 127), (125, 86), (125, 57),
    ],
    1939: [
        (69, 56), (75, 47), (99, 45), (150, 48), (223, 62), (302, 81),
        (375, 89), (418, 101), (422, 117), (410, 147), (378, 165), (346, 176),
        (293, 182), (245, 174), (175, 154), (125, 126), (92, 96), (72, 74),
    ],
    2103: [
        (71, 82), (100, 71), (150, 71), (280, 72), (351, 72), (377, 80),
        (382, 98), (367, 115), (327, 130), (267, 136), (194, 135), (145, 128),
        (100, 114), (75, 98),
    ],
}


def read_cameras(path, experiment=EXPERIMENT):
    views = json.loads(path.read_text(encoding="utf-8"))["experiments"][experiment]["views"]
    state = ReconstructionState([])
    for index, view in enumerate(views):
        if view["registered"]:
            state.registered_cams.append(index)
            state.cam_R[index] = np.array(view["world_to_camera_R"])
            state.cam_t[index] = np.array(view["world_to_camera_t"])
            state.cam_K[index] = np.array(view["final_K"])
    return views, state


def project(world, camera, state):
    camera_xyz = world @ state.cam_R[camera].T + state.cam_t[camera].reshape(3)
    depth = camera_xyz[:, 2]
    homogeneous = camera_xyz @ state.cam_K[camera].T
    pixels = homogeneous[:, :2] / np.where(depth > 0.0, depth, 1.0)[:, None]
    return pixels, depth


def silhouette_mask(world, views, state, *, polygons, thumbnail_size, minimum_votes):
    """Keep measured samples whose projections agree with the foreground masks."""
    inside = np.zeros(len(world), dtype=np.uint8)
    visible = np.zeros_like(inside)
    used_frames = set()
    width, height = thumbnail_size
    for camera in state.registered_cams:
        view = views[camera]
        frame_key = (view["name"].split("_", 1)[0], view["frame_index"])
        if frame_key not in polygons:
            continue
        used_frames.add(frame_key)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(polygons[frame_key], dtype=np.int32)], 1)
        mask = cv2.dilate(mask, np.ones((9, 9), dtype=np.uint8))
        pixels, depth = project(world, camera, state)
        center = state.cam_K[camera][:2, 2]
        x, y = np.rint(pixels * (np.array([width, height]) / (2.0 * center))).astype(int).T
        valid = (depth > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
        visible[valid] += 1
        inside[valid] += mask[y[valid], x[valid]]
    if used_frames != polygons.keys():
        raise ValueError("The silhouette masks require the saved camera poses of their source frames")
    return (inside >= minimum_votes) & (visible - inside <= 1)


def foreground_mask(aligned, frame, views, state, silhouettes):
    floor = 0.00139 * aligned[:, 0] + 0.13295 * aligned[:, 1] - 1.438
    height = aligned[:, 2] - floor
    keep = (
        np.all((aligned >= ROI_MIN) & (aligned <= ROI_MAX), axis=1)
        & (height >= 0.50) & (height <= 1.84)
    )
    if silhouettes:
        world = aligned[keep] @ frame.rotation + frame.center
        keep[keep] = silhouette_mask(
            world, views, state, polygons={("003", frame): points for frame, points in SILHOUETTES.items()},
            thumbnail_size=THUMBNAIL_SIZE, minimum_votes=7,
        )
    return keep


def binned_depth(world, camera, state, pixel_bin):
    pixels, depth = project(world, camera, state)
    width, height = (np.ceil(2.0 * state.cam_K[camera][:2, 2] / pixel_bin).astype(int) + 1)
    x, y = np.rint(pixels / pixel_bin).astype(int).T
    valid = (depth > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    cells = y[valid] * width + x[valid]
    depths = depth[valid]
    result = np.full((height, width), np.nan, dtype=np.float32)
    if not len(cells):
        return result
    order = np.lexsort((depths, cells))
    bins, first, counts = np.unique(cells[order], return_index=True, return_counts=True)
    values = depths[order]
    result.flat[bins] = (values[first + counts // 2] + values[first + (counts - 1) // 2]) / 2.0
    distance, nearest = ndimage.distance_transform_edt(np.isnan(result), return_indices=True)
    fill = (distance > 0) & (distance <= 1.5)
    result[fill] = result[tuple(nearest[:, fill])]
    return result


def fuse_depth(aligned, view_ids, frame, state, voxel_size, min_views, pixel_bin, bounds=(ROI_MIN, ROI_MAX)):
    truncation = 3.0 * voxel_size
    lower, upper = bounds
    origin = lower - truncation
    dimensions = np.ceil((upper + truncation - origin) / voxel_size).astype(int) + 1
    if np.prod(dimensions) > 25_000_000:
        raise ValueError("Use a larger voxel size to keep fusion within the memory budget")
    occupied = np.zeros(tuple(dimensions), dtype=bool)
    occupied[tuple(np.rint((aligned - origin) / voxel_size).astype(int).T)] = True
    candidates = np.argwhere(ndimage.binary_dilation(
        occupied, structure=ndimage.generate_binary_structure(3, 2), iterations=4
    ))
    sample_world = aligned @ frame.rotation + frame.center
    voxel_world = (origin + candidates * voxel_size) @ frame.rotation + frame.center
    signed_sum = np.zeros(len(candidates), dtype=np.float32)
    signed_sq = np.zeros_like(signed_sum)
    count = np.zeros(len(candidates), dtype=np.uint8)
    fused_views = 0

    for camera in np.unique(view_ids):
        selection = view_ids == camera
        if camera not in state.cam_R or np.count_nonzero(selection) < 100:
            continue
        depth_map = binned_depth(sample_world[selection], camera, state, pixel_bin)
        pixels, z = project(voxel_world, camera, state)
        x, y = np.rint(pixels / pixel_bin).astype(int).T
        height, width = depth_map.shape
        eligible = np.flatnonzero((z > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height))
        signed = depth_map[y[eligible], x[eligible]] - z[eligible]
        valid = np.isfinite(signed) & (np.abs(signed) <= truncation)
        indices = eligible[valid]
        signed_sum[indices] += signed[valid]
        signed_sq[indices] += signed[valid] ** 2
        count[indices] += 1
        fused_views += bool(len(indices))

    mean = signed_sum / np.maximum(count, 1)
    variance = signed_sq / np.maximum(count, 1) - mean ** 2
    supported = (count >= min_views) & (variance <= (0.60 * truncation) ** 2)
    known = np.zeros_like(occupied)
    signed_field = np.full(tuple(dimensions), truncation, dtype=np.float32)
    index = tuple(candidates[supported].T)
    signed_field[index] = mean[supported]
    known[index] = True
    if not np.any(mean[supported] < 0) or not np.any(mean[supported] > 0):
        raise ValueError("Stereo depth does not support a zero crossing in the chosen volume")
    return signed_field, known, origin, fused_views


def extract_mesh(signed_field, known, origin, voxel_size, aligned, colors, min_component):
    # This winding gives faces oriented toward increasing (camera-facing) distance.
    vertices, faces, _, _ = marching_cubes(
        signed_field, 0.0, gradient_direction="descent",
        mask=ndimage.binary_dilation(known), allow_degenerate=False,
    )
    # Require measured endpoints for every interpolated crossing. This discards
    # artificial faces where the volume transitions from measured to unknown.
    lower = np.floor(vertices + 1e-3).astype(int)
    upper = np.ceil(vertices - 1e-3).astype(int)
    endpoints_known = known[tuple(lower.T)] & known[tuple(upper.T)]
    low_value, high_value = signed_field[tuple(lower.T)], signed_field[tuple(upper.T)]
    crossing = (low_value * high_value <= 0) | (np.abs(low_value) <= voxel_size * 0.005)
    vertices = vertices * voxel_size + origin
    distances, neighbors = cKDTree(aligned).query(vertices, k=4, workers=4)
    measured = distances[:, 0] <= 1.5 * voxel_size
    valid_vertices = endpoints_known & crossing & measured
    faces = faces[np.all(valid_vertices[faces], axis=1)]
    if not len(faces):
        raise ValueError("No supported faces survived extraction")
    weights = np.exp(-((distances / voxel_size) ** 2))
    rgb = np.rint(np.sum(colors[neighbors] * weights[:, :, None], axis=1)
                  / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)).astype(np.uint8)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=rgb, process=False)
    parts = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=min_component,
    )
    if not len(parts):
        raise ValueError("No connected surface patches met the minimum face count")
    mesh.update_faces(np.concatenate(parts))
    mesh.remove_unreferenced_vertices()
    return mesh, sorted((len(part) for part in parts), reverse=True)


def export_mesh(aligned_mesh, frame, output, stats, object_name="red_solace"):
    world_mesh = aligned_mesh.copy()
    world_mesh.vertices = aligned_mesh.vertices @ frame.rotation + frame.center
    glb_mesh = aligned_mesh.copy()
    glb_mesh.vertices = aligned_mesh.vertices[:, [0, 2, 1]] * np.array([1, 1, -1])
    rgba = glb_mesh.visual.vertex_colors.copy()
    srgb = rgba[:, :3] / 255.0
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    rgba[:, :3] = np.rint(linear * 255).astype(np.uint8)
    glb_mesh.visual.vertex_colors = rgba
    normals = world_mesh.vertex_normals
    if not np.all(np.isfinite(normals)) or not np.allclose(np.linalg.norm(normals, axis=1), 1):
        raise ValueError("The exported surface contains invalid vertex normals")
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f"{object_name}_surface_", dir=output) as tmp:
        temp = Path(tmp)
        ply = temp / f"{object_name}_surface_mesh.ply"
        glb = temp / f"{object_name}_surface.glb"
        metrics = temp / "surface_metrics.json"
        ply.write_bytes(trimesh.exchange.ply.export_ply(world_mesh, vertex_normal=True))
        glb.write_bytes(trimesh.exchange.gltf.export_glb(glb_mesh, include_normals=True))
        for path in (ply, glb):
            readback = trimesh.load_mesh(path, process=False)
            if len(readback.faces) != len(aligned_mesh.faces) or len(readback.vertices) != len(aligned_mesh.vertices):
                raise ValueError(f"Surface export lost geometry in {path.name}")
        metrics.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
        for path in (ply, glb, metrics):
            path.replace(output / path.name)


def parse_surface_args(description, final, experiment, voxel_size, *, default_output=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", type=Path, default=final / "surface/stereo_samples_expanded.npz")
    parser.add_argument("--metrics", type=Path, default=final / "experiment_metrics.json")
    parser.add_argument("--sparse", type=Path, default=final / f"point_clouds/{experiment}_sparse.ply")
    parser.add_argument("--output-dir", "--outdir", type=Path, default=default_output or final / "surface")
    parser.add_argument("--voxel-size", type=float, default=voxel_size)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--pixel-bin", type=float, default=2.0)
    parser.add_argument("--min-component-faces", type=int, default=48)
    parser.add_argument("--without-silhouettes", action="store_true")
    args = parser.parse_args()
    if args.voxel_size <= 0 or args.min_views < 2 or args.pixel_bin <= 0 or args.min_component_faces < 1:
        parser.error("Use positive sizes and at least two depth views")
    return args


def reconstruct_surface(args, *, experiment, object_name, bounds, mask_points, silhouette_count, align_for_display=None):
    views, state = read_cameras(args.metrics, experiment)
    sparse = trimesh.load(args.sparse, process=False)
    frame = build_point_cloud_render_frame(sparse.vertices, state)
    with np.load(args.input, allow_pickle=False) as inputs:
        xyz = inputs["xyz"]
        rgb = inputs["rgb"]
        view_ids = inputs["view_ids"]
    if xyz.shape != rgb.shape or xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(view_ids):
        raise ValueError("The stereo NPZ needs xyz and rgb of shape N×3 and one view ID per point")
    finite = np.all(np.isfinite(xyz), axis=1)
    aligned = frame.transform(xyz[finite])
    rgb, view_ids = rgb[finite], view_ids[finite]
    mask = mask_points(aligned, frame, views, state, not args.without_silhouettes)
    aligned, rgb, view_ids = aligned[mask], rgb[mask], view_ids[mask]
    if len(aligned) < 100 or len(np.unique(view_ids)) < args.min_views:
        raise ValueError("Insufficient foreground stereo samples")
    print(f"Fusing {len(aligned):,} foreground stereo samples from {len(np.unique(view_ids))} reference views", flush=True)
    signed_field, known, origin, fused_views = fuse_depth(
        aligned, view_ids, frame, state, args.voxel_size, args.min_views, args.pixel_bin, bounds,
    )
    mesh, parts = extract_mesh(
        signed_field, known, origin, args.voxel_size, aligned, rgb, args.min_component_faces,
    )
    edge_lengths = mesh.edges_unique_length
    if not np.all(np.isfinite(edge_lengths)) or np.max(edge_lengths) > np.sqrt(3) * args.voxel_size * 1.001:
        raise ValueError("Mesh contains invalid or nonlocal triangles")
    if align_for_display is not None:
        display_frame = align_for_display(frame, mesh.vertices, state)
        mesh.vertices = display_frame.transform(mesh.vertices @ frame.rotation + frame.center)
        frame = display_frame
    stats = {
        "method": "multi-view projective truncated signed-distance fusion",
        "input_npz": args.input.name,
        "input_stereo_samples": int(len(xyz)),
        "foreground_stereo_samples": int(len(aligned)),
        "silhouette_views": 0 if args.without_silhouettes else silhouette_count,
        "fused_depth_views": int(fused_views),
        "minimum_supporting_depth_views": args.min_views,
        "voxel_size_in_sfm_units": args.voxel_size,
        "supported_grid_vertices": int(np.count_nonzero(known)),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "connected_components": len(parts),
        "largest_component_triangle_count": int(parts[0]),
        "largest_component_triangle_fraction": float(parts[0] / len(mesh.faces)),
        "maximum_edge_in_sfm_units": float(np.max(edge_lengths)),
        "watertight": bool(mesh.is_watertight),
        "ply_coordinates": "original SfM world coordinates",
        "glb_coordinates": "centered and upright, Y-up",
        "display_frame": {"center": frame.center.tolist(), "rotation": frame.rotation.tolist()},
    }
    export_mesh(mesh, frame, args.output_dir, stats, object_name)
    print(json.dumps(stats, indent=2), flush=True)


def main():
    args = parse_surface_args(__doc__, FINAL, EXPERIMENT, 0.030)
    reconstruct_surface(
        args, experiment=EXPERIMENT, object_name="red_solace",
        bounds=(ROI_MIN, ROI_MAX), mask_points=foreground_mask,
        silhouette_count=len(SILHOUETTES),
    )


if __name__ == "__main__":
    main()
