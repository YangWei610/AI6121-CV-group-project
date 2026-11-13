"""
AI6121 Structure from Motion (SfM) Package
==========================================
A modular, step-by-step Incremental Structure-from-Motion and Epipolar Semi-Dense
Reconstruction pipeline developed for the NTU AI6121 Computer Vision Group Project.
"""

from .data import ImageData, load_camera_calibrations, load_image_dataset, extract_video_keyframes
from .features import extract_features, match_features_pairwise
from .geometry import (
    verify_two_view_geometry,
    triangulate_two_views,
    triangulate_dlt,
    compute_reprojection_errors,
    compute_triangulation_angles,
)
from .bundle_adjustment import run_bundle_adjustment
from .pipeline import IncrementalSfM, ReconstructionState
from .densify import densify_point_cloud
from .visualization import (
    build_point_cloud_render_frame,
    export_ply,
    render_point_cloud_views,
    plot_feature_matches,
    plot_reprojection_error_hist,
    plot_scene_graph,
)
