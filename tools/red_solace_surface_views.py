"""Render the same reconstructed surface with color, edges and face normals."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from matplotlib.font_manager import findfont
from PIL import Image, ImageDraw, ImageFont


def display_axes(sparse_path: Path, metrics_path: Path,
                 experiment: str = "red_solace_video_rootsift") -> np.ndarray:
    sparse = trimesh.load(sparse_path, process=False).vertices
    centered = sparse - np.median(sparse, axis=0)
    radii = np.linalg.norm(centered, axis=1)
    core = centered[radii <= np.percentile(radii, 90)]
    _, vectors = np.linalg.eigh(np.cov(core.T))
    axes = vectors[:, ::-1]
    if np.linalg.det(axes) < 0:
        axes[:, 2] *= -1
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    views = report["experiments"][experiment]["views"]
    cameras = np.array([
        -np.asarray(view["world_to_camera_R"]).T @ view["world_to_camera_t"]
        for view in views if view["registered"]
    ])
    if np.median((cameras - np.median(sparse, axis=0)) @ axes[:, 2]) < 0:
        axes[:, 1:] *= -1
    return axes


def project(points: np.ndarray, width: int, height: int, azimuth: float, elevation: float):
    a, e = np.deg2rad([azimuth, elevation])
    u = np.cos(a) * points[:, 0] - np.sin(a) * points[:, 1]
    v = np.sin(a) * points[:, 0] + np.cos(a) * points[:, 1]
    h = np.cos(e) * points[:, 2] + np.sin(e) * v
    depth = np.sin(e) * points[:, 2] - np.cos(e) * v
    xy = np.column_stack((u, -h))
    minimum = xy.min(axis=0)
    maximum = xy.max(axis=0)
    scale = np.min(np.array([width * 0.94, height * 0.91]) / np.maximum(maximum - minimum, 1e-8))
    pixels = (xy - (minimum + maximum) / 2) * scale + [width / 2, height / 2]
    return pixels, depth


def render(mesh: trimesh.Trimesh, axes: np.ndarray, width: int, height: int,
           azimuth: float, elevation: float) -> tuple[np.ndarray, ...]:
    if mesh.is_empty or not len(mesh.faces):
        raise ValueError("The surface must contain triangles.")
    vertices = np.asarray(mesh.vertices)
    pixels, depth = project((vertices - np.median(vertices, axis=0)) @ axes,
                            width, height, azimuth, elevation)
    normals = mesh.face_normals @ axes
    faces = mesh.faces
    face_depth = depth[faces].mean(axis=1)
    face_rgb = mesh.visual.vertex_colors[faces, :3].mean(axis=1)
    light = np.array([-0.38, -0.46, 0.8])
    shading = 0.74 + 0.26 * np.maximum(normals @ light, 0)
    color_shaded = np.rint(face_rgb * shading[:, None]).astype(np.uint8)
    normal_color = np.rint(127.5 * (np.clip(normals, -1, 1) + 1)).astype(np.uint8)

    background = (248, 250, 252)
    colored = np.full((height, width, 3), background, dtype=np.uint8)
    normal_view = colored.copy()
    # Rasterize edges at twice the output resolution so dense triangles
    # retain visible interiors when the comparison is scaled down.
    wire = np.full((height * 2, width * 2, 3), background, dtype=np.uint8)
    triangles = np.rint(pixels[faces]).astype(np.int32)
    wire_triangles = np.rint(2 * pixels[faces]).astype(np.int32)
    for index in np.argsort(face_depth):
        triangle = triangles[index]
        cv2.fillConvexPoly(colored, triangle, tuple(map(int, color_shaded[index])), cv2.LINE_8)
        cv2.fillConvexPoly(normal_view, triangle, tuple(map(int, normal_color[index])), cv2.LINE_8)
        edge_triangle = wire_triangles[index]
        cv2.fillConvexPoly(wire, edge_triangle, (239, 242, 246), cv2.LINE_8)
        cv2.polylines(wire, [edge_triangle], True, (56, 77, 104), 1, cv2.LINE_AA)
    return colored, cv2.resize(wire, (width, height), interpolation=cv2.INTER_AREA), normal_view


def save_comparison(panels, path: Path):
    gap, label_height, caption_height = 22, 63, 44
    height, width = panels[0].shape[:2]
    page = Image.new("RGB", (3 * width + 2 * gap, height + label_height + caption_height), "white")
    draw = ImageDraw.Draw(page)
    typeface = findfont("DejaVu Sans")
    font = ImageFont.truetype(typeface, 31)
    caption = ImageFont.truetype(typeface, 20)
    for index, (title, panel) in enumerate(zip(("Colored surface", "Wireframe", "Face normals"), panels)):
        left = index * (width + gap)
        draw.text((left + 16, 16), title, fill=(25, 42, 64), font=font)
        page.paste(Image.fromarray(panel), (left, label_height))
    draw.text((2 * (width + gap) + 16, height + label_height + 10),
              "RGB encodes the X, Y, Z components of each normal.",
              fill=(69, 84, 102), font=caption)
    path.parent.mkdir(parents=True, exist_ok=True)
    page.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scene", choices=("red-solace", "fern"), default="red-solace")
    parser.add_argument("--surface-metrics", type=Path,
                        help="Use the display frame saved with the surface mesh")
    parser.add_argument("--azimuth", type=float, default=-19.0)
    parser.add_argument("--elevation", type=float, default=26.0)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1] / "outputs/final" / args.scene
    experiment = args.scene.replace("-", "_") + "_video_rootsift"
    if args.surface_metrics:
        frame = json.loads(args.surface_metrics.read_text(encoding="utf-8"))["display_frame"]
        axes = np.asarray(frame["rotation"], dtype=float).T
    else:
        axes = display_axes(base / "point_clouds" / f"{experiment}_sparse.ply",
                            base / "experiment_metrics.json", experiment)
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    panels = render(mesh, axes, 920, 600, args.azimuth, args.elevation)
    save_comparison(panels, args.output)


if __name__ == "__main__":
    main()
