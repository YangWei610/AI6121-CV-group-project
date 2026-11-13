"""Reproduce the selected Fern surface patch from the saved 002/004 stereo replay."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay
import trimesh

from sfm.visualization import build_point_cloud_render_frame
from tools.red_solace_surface import export_mesh, project, read_cameras, silhouette_mask


REPLAY = Path("outputs/validation/fern_archive")
EXPERIMENT = "fern_video_rootsift"
REFERENCE = 34
REFERENCE_KEYFRAME = "004_kf009_f0662.jpg"
MAX_IMAGE_EDGE = 7.5


def subject_samples(xyz, frame, views, state, masks):
    aligned = frame.transform(xyz)
    floor_height = aligned @ np.array([0.071, 0.026, 0.9971]) + 1.2569
    keep = (
        np.all(aligned >= [-7.2, -8.5, -1.6], axis=1)
        & np.all(aligned <= [3.2, 1.8, 3.2], axis=1)
        & (floor_height >= 0.12)
    )
    polygons = {}
    for camera_id, outline in masks["polygons"].items():
        view = views[int(camera_id)]
        clip = view["name"].split("_", 1)[0]
        polygons[clip, view["frame_index"]] = outline
    keep[keep] = silhouette_mask(
        xyz[keep], views, state, polygons=polygons,
        thumbnail_size=masks["reference_size"], minimum_votes=4,
    )
    return keep


def triangulate_patch(xyz, pixels, rgb, state):
    _, unique = np.unique(np.round(pixels, 2), axis=0, return_index=True)
    xyz, pixels, rgb = xyz[unique], pixels[unique], rgb[unique]
    faces = Delaunay(pixels).simplices.copy()
    triangles, image_triangles = xyz[faces], pixels[faces]
    image_edges = np.linalg.norm(image_triangles - np.roll(image_triangles, 1, axis=1), axis=2)
    world_edges = np.linalg.norm(triangles - np.roll(triangles, 1, axis=1), axis=2)
    _, depth = project(xyz, REFERENCE, state)
    local_pixel_size = depth[faces].mean(axis=1) / state.cam_K[REFERENCE][0, 0]
    world_limit = np.minimum(0.16, 3.3 * local_pixel_size[:, None] * image_edges)
    supported = np.all((image_edges <= MAX_IMAGE_EDGE) & (world_edges <= world_limit), axis=1)
    faces, triangles = faces[supported], triangles[supported]
    camera = -state.cam_R[REFERENCE].T @ state.cam_t[REFERENCE].reshape(3)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flip = np.sum(normals * (camera - triangles.mean(axis=1)), axis=1) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    mesh = trimesh.Trimesh(vertices=xyz, faces=faces, vertex_colors=rgb, process=False)
    components = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(faces)), min_len=20,
    )
    if not len(components):
        raise ValueError("No supported components in the selected Fern reference")
    mesh.update_faces(np.concatenate(components))
    mesh.remove_unreferenced_vertices()
    return mesh, camera, len(components)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=REPLAY)
    parser.add_argument("--output-dir", type=Path, default=REPLAY / "reproduced")
    args = parser.parse_args()
    replay = args.replay_dir
    views, state = read_cameras(replay / "experiment_metrics.json", EXPERIMENT)
    if REFERENCE not in state.cam_R or views[REFERENCE]["name"] != REFERENCE_KEYFRAME:
        raise ValueError("Expected the saved Fern view 34 camera pose for frame 662 of clip 004")
    sparse = trimesh.load(replay / "fern_same_50_frames_sparse.ply", process=False)
    frame = build_point_cloud_render_frame(sparse.vertices, state)
    masks = json.loads((replay / "silhouettes.json").read_text(encoding="utf-8"))
    with np.load(replay / "stereo_samples_expanded.npz", allow_pickle=False) as samples:
        xyz, rgb = samples["xyz"], samples["rgb"]
        view_ids = samples["view_ids"]
        pixels, support = samples["source_uv"], samples["support_count"]
        pairs = samples["pairs"]
    reference = view_ids == REFERENCE
    selected = reference & (support >= 1) & subject_samples(xyz, frame, views, state, masks)
    if np.count_nonzero(selected) < 100:
        raise ValueError("Insufficient measured stereo depth in the Fern reference")
    mesh, camera, components = triangulate_patch(xyz[selected], pixels[selected], rgb[selected], state)

    # Retain the replay's up axis and rotate only the horizontal plane to face the camera.
    relative_camera = (camera - np.median(mesh.vertices, axis=0)) @ frame.rotation.T
    yaw = -np.pi / 2 - np.arctan2(relative_camera[1], relative_camera[0])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    yaw_rotation = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    display = replace(frame, rotation=yaw_rotation @ frame.rotation)
    elevation = np.degrees(np.arctan2(relative_camera[2], np.linalg.norm(relative_camera[:2])))
    stats = {
        "method": "Local triangulation in the original image of cross-view-checked stereo depth",
        "scope": "Observed surface patch of the visible ribs and base; sides outside the selected view and unsupported gaps remain open",
        "archived_pose_source": "Fern 002 and 004 archived replay",
        "reference_view": REFERENCE,
        "reference_keyframe": REFERENCE_KEYFRAME,
        "stereo_pair_view_indices": pairs[pairs[:, 0] == REFERENCE].tolist(),
        "candidate_stereo_reference_count": int(len(np.unique(view_ids))),
        "reference_depth_samples_before_subject_crop": int(np.count_nonzero(reference)),
        "maximum_image_edge_px": MAX_IMAGE_EDGE,
        "maximum_3d_edge_in_sfm_units": float(np.max(mesh.edges_unique_length)),
        "vertices": int(len(mesh.vertices)),
        "triangles": int(len(mesh.faces)),
        "connected_components": int(components),
        "watertight": bool(mesh.is_watertight),
        "preview": {"azimuth": 0, "elevation": float(elevation)},
        "display_frame": {"center": display.center.tolist(), "rotation": display.rotation.tolist()},
        "ply_coordinates": "archived replay SfM world coordinates",
        "glb_coordinates": "upright Y-up, turned toward the source camera",
    }
    mesh.vertices = display.transform(mesh.vertices)
    # Retain the selected patch's original GLB scene and mesh names.
    mesh.metadata.update(name="best_view_patch.ply", node="best_view_patch.ply")
    export_mesh(mesh, display, args.output_dir, stats, "fern")
    print(f"Fern reference {REFERENCE_KEYFRAME}: {stats['vertices']:,} vertices, {stats['triangles']:,} triangles")


if __name__ == "__main__":
    main()
