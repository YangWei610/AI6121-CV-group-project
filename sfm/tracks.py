"""Conflict-free feature tracks built from verified image-pair matches."""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .geometry import VerifiedPair


@dataclass
class FeatureTracks:
    observations: List[Dict[int, int]]
    by_image: List[Dict[int, int]]  # track id -> keypoint index
    lookup: Dict[Tuple[int, int], int]
    conflicts: int


def build_feature_tracks(
    image_count: int, pairs: Dict[Tuple[int, int], VerifiedPair]
) -> FeatureTracks:
    parent = {}
    components = {}
    conflicts = 0

    def root(key):
        if key not in parent:
            parent[key] = key
            components[key] = {key[0]: key[1]}
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    # Give edges from pairs with stronger geometric consensus first choice of
    # assignments, and never join tracks with two keypoints in the same image.
    ranked_pairs = sorted(
        pairs.values(),
        key=lambda p: (-p.inlier_ratio * len(p.matches) ** 0.5, p.i, p.j),
    )
    for pair in ranked_pairs:
        for ki, kj in pair.matches:
            a = root((pair.i, int(ki)))
            b = root((pair.j, int(kj)))
            if a == b:
                continue
            if components[a].keys() & components[b].keys():
                conflicts += 1
                continue
            if len(components[a]) < len(components[b]):
                a, b = b, a
            parent[b] = a
            components[a].update(components.pop(b))

    observations = [obs for obs in components.values() if len(obs) > 1]
    by_image = [{} for _ in range(image_count)]
    lookup = {}
    for track_id, obs in enumerate(observations):
        for image_id, keypoint_id in obs.items():
            by_image[image_id][track_id] = keypoint_id
            lookup[(image_id, keypoint_id)] = track_id
    return FeatureTracks(observations, by_image, lookup, conflicts)
