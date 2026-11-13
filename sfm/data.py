"""Load photos and select usable keyframes with explicit camera metadata."""

import json
import os
import shutil
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List, Mapping, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class ImageData:
    """A view and its intrinsics in the pixel coordinates used by SfM."""

    idx: int
    name: str
    path: str
    width: int
    height: int
    K: np.ndarray
    image_rgb: np.ndarray
    image_gray: np.ndarray
    kps: Optional[np.ndarray] = None
    descs: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    source_type: str = "image"
    source_path: Optional[str] = None
    camera_group: str = ""
    intrinsics_source: str = "focal_prior"
    intrinsics_fixed: bool = False
    frame_index: Optional[int] = None
    timestamp_sec: Optional[float] = None


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole calibration in upright, original-resolution source pixels."""

    width: int
    height: int
    K: np.ndarray
    distortion: np.ndarray
    group: Optional[str] = None


def _source_key(path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def load_camera_calibrations(
    path, dataset_root=None,
) -> Dict[str, CameraCalibration]:
    """Read per-source width, height, K, optional OpenCV distortion and group.

    JSON keys are file paths, relative to dataset_root (or the JSON file's
    parent when omitted). Each entry calibrates one photo or one entire clip.
    """
    config_path = Path(path)
    with config_path.open(encoding="utf-8-sig") as stream:
        entries = json.load(stream)
    if not isinstance(entries, dict):
        raise ValueError("Calibration JSON must map source paths to camera intrinsics.")

    root = Path(dataset_root) if dataset_root is not None else config_path.parent
    calibrations = {}
    for source, values in entries.items():
        try:
            width, height = values["width"], values["height"]
            matrix = np.asarray(values["K"], dtype=np.float64)
            distortion = np.asarray(values.get("distortion", []), dtype=np.float64)
            group = values.get("group")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid calibration for {source}: {exc}") from exc
        if type(width) is not int or type(height) is not int or min(width, height) <= 0:
            raise ValueError(f"Calibration dimensions must be positive integers for {source}.")
        if (
            matrix.shape != (3, 3)
            or not np.isfinite(matrix).all()
            or min(matrix[0, 0], matrix[1, 1]) <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.allclose([matrix[0, 1], matrix[1, 0]], 0)
        ):
            raise ValueError(f"Expected a finite, zero-skew pinhole K matrix for {source}.")
        if (
            distortion.ndim != 1
            or distortion.size not in (0, 4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all()
        ):
            raise ValueError(f"Invalid OpenCV distortion coefficients for {source}.")
        if group is not None and (not isinstance(group, str) or not group.strip()):
            raise ValueError(f"Calibration group must be a nonempty string for {source}.")
        key = _source_key(root / source)
        if key in calibrations:
            raise ValueError(f"Duplicate calibration for {source}.")
        calibrations[key] = CameraCalibration(width, height, matrix, distortion, group)
    return calibrations


def _estimate_initial_intrinsics(
    width: int,
    height: int,
    exif_focal_35mm: Optional[float] = None,
    focal_factor: float = 1.15,
) -> np.ndarray:
    focal = (
        exif_focal_35mm / 36.0 if exif_focal_35mm is not None else focal_factor
    ) * max(width, height)
    return np.array(
        [[focal, 0.0, width / 2.0],
         [0.0, focal, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _read_exif_focal_35mm(path) -> Optional[float]:
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            focal = exif.get(41989) or exif.get_ifd(34665).get(41989)
            if focal is not None and np.isfinite(float(focal)) and float(focal) > 0:
                return float(focal)
    except (OSError, ValueError, TypeError, ZeroDivisionError):
        pass
    return None


def _resize_image_max_dim(image: np.ndarray, max_dim: int) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) <= max_dim:
        return image
    scale = max_dim / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _prepare_view(
    image: np.ndarray,
    source: Path,
    max_dim: int,
    focal_factor: float,
    calibration: Optional[CameraCalibration],
    exif_focal: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
    source_height, source_width = image.shape[:2]
    image = _resize_image_max_dim(image, max_dim)
    height, width = image.shape[:2]
    if calibration is None:
        K = _estimate_initial_intrinsics(width, height, exif_focal, focal_factor)
        return image, K, "exif_prior" if exif_focal is not None else "focal_prior"

    if (source_width, source_height) != (calibration.width, calibration.height):
        raise ValueError(
            f"Calibration for {source} expects {calibration.width}x{calibration.height} "
            f"upright pixels, but decoded {source_width}x{source_height}."
        )
    scale_x, scale_y = width / source_width, height / source_height
    K = np.diag([scale_x, scale_y, 1.0]) @ calibration.K
    # OpenCV resize maps pixel centers, so an off-center calibration needs
    # the same half-pixel transformation as the image.
    K[0, 2] += (scale_x - 1.0) / 2.0
    K[1, 2] += (scale_y - 1.0) / 2.0
    if calibration.distortion.any():
        size = (width, height)
        rectified_K, _ = cv2.getOptimalNewCameraMatrix(
            K, calibration.distortion, size, 0.0, size
        )
        image = cv2.undistort(image, K, calibration.distortion, None, rectified_K)
        K = rectified_K
    return image, K, "calibrated"


def _make_view(
    idx: int,
    name: str,
    path: Path,
    source: Path,
    image: np.ndarray,
    K: np.ndarray,
    intrinsics_source: str,
    calibration: Optional[CameraCalibration],
    frame_index: Optional[int] = None,
    timestamp_sec: Optional[float] = None,
) -> ImageData:
    height, width = image.shape[:2]
    group = calibration.group if calibration is not None else None
    return ImageData(
        idx=idx,
        name=name,
        path=str(path),
        width=width,
        height=height,
        K=K,
        image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
        source_type="video" if frame_index is not None else "image",
        source_path=str(source),
        camera_group=group or f"{_source_key(source)}:{width}x{height}",
        intrinsics_source=intrinsics_source,
        intrinsics_fixed=calibration is not None,
        frame_index=frame_index,
        timestamp_sec=timestamp_sec,
    )


def load_image_dataset(
    image_dir: str,
    max_dim: int = 1600,
    focal_factor: float = 1.05,
    max_images: Optional[int] = None,
    calibrations: Optional[Mapping[str, CameraCalibration]] = None,
) -> List[ImageData]:
    """Read upright images with calibrated intrinsics or an initial focal prior."""
    if max_dim <= 0 or not np.isfinite(focal_factor) or focal_factor <= 0:
        raise ValueError("max_dim and focal_factor must be positive.")
    root = Path(image_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = sorted(
        path for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    calibrations = calibrations or {}
    dataset = []
    for path in paths[:max_images]:
        try:
            encoded = np.fromfile(path, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
        except (OSError, cv2.error):
            continue
        if image is None:
            continue
        calibration = calibrations.get(_source_key(path))
        exif_focal = _read_exif_focal_35mm(path) if calibration is None else None
        image, K, source = _prepare_view(
            image, path, max_dim, focal_factor, calibration, exif_focal
        )
        dataset.append(
            _make_view(len(dataset), path.name, path, path, image, K, source, calibration)
        )
    return dataset


@contextmanager
def _open_video(path: Path):
    """Prefer direct decoding; stage the file if a backend rejects its path."""
    if not path.is_file():
        raise FileNotFoundError(f"Video file not found: {path}")
    names = [str(path)]
    try:
        relative_name = os.path.relpath(path)
    except ValueError:  # Windows paths can be on different drives.
        relative_name = None
    if relative_name is not None and relative_name.isascii() and relative_name not in names:
        names.insert(0, relative_name)
    for name in names:
        capture = cv2.VideoCapture(name)
        if capture.isOpened():
            try:
                yield capture
            finally:
                capture.release()
            return
        capture.release()

    with TemporaryDirectory(prefix="sfm-video-") as temp_dir:
        staged = Path(temp_dir) / f"clip{path.suffix.lower()}"
        try:
            os.link(path, staged)
        except OSError:  # Hard links require the same filesystem.
            shutil.copyfile(path, staged)
        capture = cv2.VideoCapture(str(staged))
        try:
            if not capture.isOpened():
                raise OSError(f"OpenCV could not decode video: {path}")
            yield capture
        finally:
            capture.release()


@dataclass
class _FrameCandidate:
    index: int
    timestamp_sec: Optional[float]
    image: np.ndarray
    gray: np.ndarray
    sharpness: float


def _read_candidate(
    capture, requested_index: int, fps: float,
) -> Optional[_FrameCandidate]:
    next_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    distance = requested_index - next_index
    if 0 <= distance <= 12:
        # Grabbing a few consecutive frames avoids seeking back to a codec's
        # previous keyframe and decoding the same group of pictures repeatedly.
        for _ in range(distance):
            if not capture.grab():
                return None
    elif not capture.set(cv2.CAP_PROP_POS_FRAMES, requested_index):
        return None
    ok, image = capture.read()
    if not ok or image is None:
        return None
    next_index = capture.get(cv2.CAP_PROP_POS_FRAMES)
    index = int(round(next_index)) - 1 if next_index >= 1 else requested_index
    msec = capture.get(cv2.CAP_PROP_POS_MSEC)
    timestamp = (
        msec / 1000.0 if np.isfinite(msec) and (msec > 0 or (msec == 0 and index == 0))
        else index / fps if np.isfinite(fps) and fps > 0 else None
    )
    small = _resize_image_max_dim(image, 480)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    # Keep the decoded pixels for calibration. Only a few frames are sampled
    # in each window and rejected candidates are released immediately.
    return _FrameCandidate(index, timestamp, image, gray, sharpness)


def _track_motion(
    previous: np.ndarray, candidate: np.ndarray, corners: Optional[np.ndarray],
) -> Tuple[float, float, bool]:
    """Return tracked fraction, image displacement at 480 px, and duplicate flag."""
    if candidate.shape != previous.shape:
        candidate = cv2.resize(candidate, (previous.shape[1], previous.shape[0]))
    same_appearance = float(cv2.absdiff(previous, candidate).mean()) < 1.5
    if corners is None or len(corners) < 8:
        return 0.0, 0.0, same_appearance

    forward, forward_ok, _ = cv2.calcOpticalFlowPyrLK(previous, candidate, corners, None)
    if forward is None:
        return 0.0, 0.0, same_appearance
    backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(candidate, previous, forward, None)
    if backward is None:
        return 0.0, 0.0, same_appearance
    xy = forward.reshape(-1, 2)
    h, w = previous.shape
    consistent = (
        forward_ok.ravel().astype(bool)
        & backward_ok.ravel().astype(bool)
        & (np.linalg.norm((corners - backward).reshape(-1, 2), axis=1) < 1.5)
        & (xy[:, 0] >= 0) & (xy[:, 0] < w)
        & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    )
    overlap = float(consistent.mean())
    motion = (
        float(np.median(np.linalg.norm((forward - corners).reshape(-1, 2)[consistent], axis=1)))
        if consistent.any() else 0.0
    )
    duplicate = same_appearance or (overlap >= 0.45 and motion < 1.2)
    return overlap, motion, duplicate


def _candidate_indices(
    anchors: np.ndarray, anchor_idx: int, total: int, blur_window: int, adaptive: bool,
) -> List[int]:
    anchor = int(anchors[anchor_idx])
    # Disjoint temporal windows preserve the order even when a sharper frame
    # is chosen away from an anchor.
    lo = (int(anchors[anchor_idx - 1]) + anchor) // 2 + 1 if anchor_idx else 0
    hi = (anchor + int(anchors[anchor_idx + 1])) // 2 if anchor_idx + 1 < len(anchors) else total - 1
    if not adaptive:
        return list(range(max(lo, anchor - blur_window), min(hi, anchor + blur_window) + 1))
    return sorted({lo, max(lo, anchor - blur_window), anchor,
                   min(hi, anchor + blur_window), hi})


def _choose_frame(
    candidates: List[_FrameCandidate], previous: Optional[_FrameCandidate], adaptive: bool,
) -> Tuple[Optional[_FrameCandidate], float, float]:
    if previous is None or not adaptive:
        return max(candidates, key=lambda frame: frame.sharpness), 0.0, 0.0

    corners = cv2.goodFeaturesToTrack(
        previous.gray, maxCorners=250, qualityLevel=0.01, minDistance=6
    )
    sharpest = max(frame.sharpness for frame in candidates)
    best = None
    best_score = -np.inf
    best_overlap = best_motion = 0.0
    for frame in candidates:
        if frame.index <= previous.index:
            continue
        overlap, motion, duplicate = _track_motion(previous.gray, frame.gray, corners)
        if duplicate:
            continue
        # Prefer a visible connection to the previous frame, moderate image
        # displacement and a sharp image. A low-overlap choice is retained
        # when it is the only way to cover this interval and is recorded.
        score = (
            2.0 * overlap
            + 0.5 * min(motion / 15.0, 1.0)
            + 0.45 * np.log1p(frame.sharpness) / max(1.0, np.log1p(sharpest))
        )
        if score > best_score:
            best, best_score = frame, score
            best_overlap, best_motion = overlap, motion
    return best, best_overlap, best_motion


def extract_video_keyframes(
    video_paths: List[str],
    target_keyframes_per_video: int = 35,
    blur_window: int = 5,
    max_dim: int = 1280,
    focal_factor: float = 0.95,
    save_dir: Optional[str] = None,
    start_idx: int = 0,
    calibrations: Optional[Mapping[str, CameraCalibration]] = None,
    adaptive: bool = True,
) -> Tuple[List[ImageData], Dict[str, List[float]]]:
    """Cover each clip with sharp, distinct keyframes and usable local overlap.

    The target is an upper bound; windows with no motion can be skipped.
    Motion is a selection proxy in image pixels, not calibrated 3D parallax.
    Set adaptive=False to select by sharpness near uniform time anchors.
    """
    if target_keyframes_per_video <= 0 or blur_window < 0:
        raise ValueError("Keyframe target must be positive and blur_window nonnegative.")
    if max_dim <= 0 or not np.isfinite(focal_factor) or focal_factor <= 0:
        raise ValueError("max_dim and focal_factor must be positive.")
    output_dir = Path(save_dir).resolve() if save_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    paths = [Path(path).resolve() for path in video_paths]
    stem_counts = Counter(path.stem.casefold() for path in paths)
    calibrations = calibrations or {}
    dataset = []
    stats: Dict[str, List[float]] = {
        "selected_sharpness": [],
        "naive_sharpness": [],
        "selected_overlap": [],
        "selected_motion_px": [],
    }

    for clip_idx, path in enumerate(paths):
        calibration = calibrations.get(_source_key(path))
        stem = path.stem if stem_counts[path.stem.casefold()] == 1 else f"{clip_idx}_{path.stem}"
        with _open_video(path) as capture:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if total <= 0:
                raise OSError(f"OpenCV could not determine the frame count for {path}.")
            count = min(total, target_keyframes_per_video)
            anchors = np.linspace(0, total - 1, count, dtype=int)
            previous = None
            clip_keyframes = 0
            for anchor_idx, anchor in enumerate(anchors):
                indices = _candidate_indices(anchors, anchor_idx, total, blur_window, adaptive)
                candidates = [
                    frame for index in indices
                    if (frame := _read_candidate(capture, index, fps)) is not None
                ]
                if not candidates:
                    continue
                chosen, overlap, motion = _choose_frame(candidates, previous, adaptive)
                if chosen is None:
                    continue

                image, K, source = _prepare_view(
                    chosen.image, path, max_dim, focal_factor, calibration
                )
                name = f"{stem}_kf{clip_keyframes:03d}_f{chosen.index:04d}.jpg"
                keyframe_path = output_dir / name if output_dir is not None else path
                if output_dir is not None:
                    ok, encoded = cv2.imencode(".jpg", image)
                    if not ok:
                        raise OSError(f"Could not encode keyframe: {keyframe_path}")
                    encoded.tofile(keyframe_path)
                dataset.append(
                    _make_view(
                        start_idx + len(dataset), name, keyframe_path, path, image, K, source, calibration,
                        frame_index=chosen.index, timestamp_sec=chosen.timestamp_sec,
                    )
                )
                stats["selected_sharpness"].append(chosen.sharpness)
                naive = min(candidates, key=lambda frame: abs(frame.index - anchor))
                stats["naive_sharpness"].append(naive.sharpness)
                if previous is not None and adaptive:
                    stats["selected_overlap"].append(overlap)
                    stats["selected_motion_px"].append(motion)
                previous = chosen
                clip_keyframes += 1

    return dataset, stats
