"""Feature extraction and mutual ratio matching for float and binary descriptors."""

import cv2
import numpy as np

from .data import ImageData


def extract_features(
    dataset: list[ImageData],
    method: str = "rootsift",
    max_features: int = 4000,
    contrast_threshold: float = 0.02,
) -> None:
    method = method.lower()
    if method in {"sift", "rootsift"}:
        detector = cv2.SIFT_create(
            nfeatures=max_features, contrastThreshold=contrast_threshold, edgeThreshold=10
        )
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=max_features, scaleFactor=1.2, nlevels=8)
    else:
        raise ValueError(f"Unsupported feature method: {method}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for view in dataset:
        keypoints, descriptors = detector.detectAndCompute(clahe.apply(view.image_gray), None)
        if not keypoints or descriptors is None:
            dtype = np.uint8 if method == "orb" else np.float32
            view.kps = np.empty((0, 2), dtype=np.float64)
            view.descs = np.empty((0, detector.descriptorSize()), dtype=dtype)
            view.colors = np.empty((0, 3), dtype=np.uint8)
            continue

        if method == "rootsift":
            descriptors = np.sqrt(descriptors / np.maximum(descriptors.sum(axis=1, keepdims=True), 1e-7))
        elif method == "sift":
            descriptors /= np.maximum(np.linalg.norm(descriptors, axis=1, keepdims=True), 1e-7)

        points = np.array([point.pt for point in keypoints], dtype=np.float64)
        xs = np.clip(np.round(points[:, 0]).astype(int), 0, view.width - 1)
        ys = np.clip(np.round(points[:, 1]).astype(int), 0, view.height - 1)
        view.kps, view.descs, view.colors = points, descriptors, view.image_rgb[ys, xs]


def match_pair_mutual_ratio(
    desc1: np.ndarray, desc2: np.ndarray, ratio_thresh: float = 0.78
) -> np.ndarray:
    """Apply Lowe's ratio test and a mutual nearest-neighbor check."""
    if len(desc1) < 2 or len(desc2) < 2:
        return np.empty((0, 2), dtype=np.int32)
    if desc1.dtype != desc2.dtype or desc1.shape[1] != desc2.shape[1]:
        raise ValueError("Both descriptor arrays must come from the same feature method")

    norm = cv2.NORM_HAMMING if desc1.dtype == np.uint8 else cv2.NORM_L2
    matcher = cv2.BFMatcher(norm)
    good = [
        first for first, second in matcher.knnMatch(desc1, desc2, k=2)
        if first.distance < ratio_thresh * second.distance
    ]
    if not good:
        return np.empty((0, 2), dtype=np.int32)

    # Reverse-check only descriptors that survived the forward ratio test.
    train_ids = np.unique([match.trainIdx for match in good])
    reverse = {
        int(train_ids[match.queryIdx]): match.trainIdx
        for match in matcher.match(desc2[train_ids], desc1)
    }
    return np.asarray(
        [(match.queryIdx, match.trainIdx) for match in good
         if reverse.get(match.trainIdx) == match.queryIdx],
        dtype=np.int32,
    ).reshape(-1, 2)


def match_features_pairwise(
    dataset: list[ImageData],
    ratio_thresh: float = 0.78,
    min_raw_matches: int = 24,
    matching_mode: str = "exhaustive",
    seq_window: int = 14,
    loop_stride: int = 4,
) -> dict[tuple[int, int], np.ndarray]:
    """Match all pairs, or video neighbors with sampled loop candidates."""
    if matching_mode not in {"exhaustive", "sequential_loop"}:
        raise ValueError(f"Unsupported matching mode: {matching_mode}")
    raw_matches = {}
    for i, first in enumerate(dataset):
        if first.descs is None or len(first.descs) < min_raw_matches:
            continue
        for j in range(i + 1, len(dataset)):
            second = dataset[j]
            if second.descs is None or len(second.descs) < min_raw_matches:
                continue
            if matching_mode == "sequential_loop":
                same_clip = getattr(first, "source_path", "") == getattr(second, "source_path", "")
                is_neighbor = same_clip and j - i <= seq_window
                is_loop = i % loop_stride == 0 and j % loop_stride == 0
                if not (is_neighbor or is_loop):
                    continue
            matches = match_pair_mutual_ratio(first.descs, second.descs, ratio_thresh)
            if len(matches) >= min_raw_matches:
                raw_matches[i, j] = matches
    return raw_matches
