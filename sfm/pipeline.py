"""
Incremental Structure-from-Motion (SfM) Pipeline.
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import cv2
import numpy as np

from .data import ImageData
from .geometry import (
    VerifiedPair,
    triangulate_two_views,
    compute_reprojection_errors,
    image_coverage,
    normalize_keypoints,
)
from .bundle_adjustment import run_bundle_adjustment
from .tracks import build_feature_tracks


@dataclass
class Point3D:
    pt_id: int
    xyz: np.ndarray  # (3,) float64
    rgb: np.ndarray  # (3,) uint8
    observations: Dict[int, int] = field(default_factory=dict)  # img_idx -> kp_idx
    valid: bool = True


@dataclass
class ReconstructionState:
    dataset: List[ImageData]
    registered_cams: List[int] = field(default_factory=list)
    cam_R: Dict[int, np.ndarray] = field(default_factory=dict)
    cam_t: Dict[int, np.ndarray] = field(default_factory=dict)
    cam_K: Dict[int, np.ndarray] = field(default_factory=dict)
    points3d: List[Point3D] = field(default_factory=list)
    kp_to_pt3d: Dict[Tuple[int, int], int] = field(default_factory=dict)
    ba_history: List[Dict[str, float]] = field(default_factory=list)
    pre_ba_errors: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    post_ba_errors: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    init_pair: Tuple[int, int] = (-1, -1)
    elapsed_sec: float = 0.0
    track_to_pt3d: Dict[int, int] = field(default_factory=dict)
    seed_attempts: List[dict] = field(default_factory=list)
    track_conflicts: int = 0
    num_feature_tracks: int = 0
    num_seed_candidates: int = 0
    registration_retries: int = 0

    def get_valid_points(self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        xyz_list, rgb_list, ids = [], [], []
        for p in self.points3d:
            if p.valid and len(p.observations) >= 2 and np.all(np.isfinite(p.xyz)):
                xyz_list.append(p.xyz)
                rgb_list.append(p.rgb)
                ids.append(p.pt_id)
        if not xyz_list:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8), []
        return np.array(xyz_list, dtype=np.float64), np.array(rgb_list, dtype=np.uint8), ids

    def compute_all_reprojection_errors(self) -> np.ndarray:
        by_camera = {cid: [] for cid in self.registered_cams}
        for point in self.points3d:
            if point.valid and len(point.observations) >= 2:
                for cid, kp in point.observations.items():
                    if cid in by_camera:
                        by_camera[cid].append((point.xyz, kp))
        errs = []
        for cid, observations in by_camera.items():
            if not observations:
                continue
            xyz, kp = zip(*observations)
            errors, _ = compute_reprojection_errors(
                np.asarray(xyz), self.dataset[cid].kps[list(kp)],
                self.cam_K[cid], self.cam_R[cid], self.cam_t[cid],
            )
            errs.append(errors)
        return np.concatenate(errs) if errs else np.zeros(0)

    def get_summary_metrics(self) -> Dict[str, Optional[float]]:
        xyz, _, ids = self.get_valid_points()
        errs = self.compute_all_reprojection_errors()
        track_lengths = [len(self.points3d[pid].observations) for pid in ids]
        return {
            "num_total_images": len(self.dataset),
            "num_registered_images": len(self.registered_cams),
            "registration_rate": len(self.registered_cams) / max(1, len(self.dataset)),
            "num_3d_points": len(xyz),
            "num_observations": len(errs),
            "track_3plus_fraction": float(np.mean(np.asarray(track_lengths) >= 3)) if track_lengths else 0.0,
            "mean_track_length": float(np.mean(track_lengths)) if track_lengths else 0.0,
            "max_track_length": int(np.max(track_lengths)) if track_lengths else 0,
            "mean_reproj_error_px": float(np.mean(errs)) if len(errs) else None,
            "median_reproj_error_px": float(np.median(errs)) if len(errs) else None,
            "rmse_reproj_error_px": float(np.sqrt(np.mean(errs ** 2))) if len(errs) else None,
            "pre_ba_mean_error_px": float(np.mean(self.pre_ba_errors)) if len(self.pre_ba_errors) else None,
            "pre_ba_median_error_px": float(np.median(self.pre_ba_errors)) if len(self.pre_ba_errors) else None,
            "elapsed_sec": round(self.elapsed_sec, 2),
        }



class IncrementalSfM:
    """Grow a reconstruction from consistent tracks and tested seed pairs."""

    def __init__(
        self,
        dataset: List[ImageData],
        verified_pairs: Dict[Tuple[int, int], VerifiedPair],
        use_ba: bool = True,
        optimize_focal: bool = True,
        ba_interval: int = 10,
        pnp_ransac_thresh: float = 4.5,
        min_pnp_inliers: int = 12,
        max_reproj_err: float = 4.5,
        min_tri_angle_deg: float = 1.0,
        max_seed_attempts: int = 4,
        seed_probe_views: int = 10,
    ):
        self.dataset = dataset
        self.verified_pairs = verified_pairs
        self.use_ba = use_ba
        self.optimize_focal = optimize_focal
        self.ba_interval = ba_interval
        self.pnp_ransac_thresh = pnp_ransac_thresh
        self.min_pnp_inliers = min_pnp_inliers
        self.max_reproj_err = max_reproj_err
        self.min_tri_angle_deg = min_tri_angle_deg
        self.max_seed_attempts = max(1, max_seed_attempts)
        self.seed_probe_views = max(3, seed_probe_views)
        self.tracks = build_feature_tracks(len(dataset), verified_pairs)
        self.state = self._new_state()

    def _new_state(self) -> ReconstructionState:
        return ReconstructionState(
            dataset=self.dataset,
            cam_K={i: img.K.copy() for i, img in enumerate(self.dataset)},
            track_conflicts=self.tracks.conflicts,
            num_feature_tracks=len(self.tracks.observations),
        )

    def _seed_track_matches(self, pair: VerifiedPair) -> List[Tuple[int, int, int]]:
        matches = pair.seed_matches if pair.seed_matches is not None else pair.matches
        lookup = self.tracks.lookup
        usable = []
        for ki, kj in matches:
            track = lookup.get((pair.i, int(ki)))
            if track is not None and track == lookup.get((pair.j, int(kj))):
                usable.append((track, int(ki), int(kj)))
        return usable

    def rank_initial_pairs(self) -> List[VerifiedPair]:
        ranked = []
        for pair in self.verified_pairs.values():
            if pair.R is None or pair.t is None or pair.median_angle_deg < 1.75:
                continue
            usable = self._seed_track_matches(pair)
            if len(usable) < self.min_pnp_inliers:
                continue

            third_view_counts: Dict[int, int] = {}
            for track, _, _ in usable:
                for image_id in self.tracks.observations[track]:
                    if image_id != pair.i and image_id != pair.j:
                        third_view_counts[image_id] = third_view_counts.get(image_id, 0) + 1
            bridge_views = sum(count >= self.min_pnp_inliers for count in third_view_counts.values())
            bridge_points = max(third_view_counts.values(), default=0)
            # Broad image support, moderate parallax, and third views with enough
            # shared tracks are more informative than the raw match count.
            score = (
                len(usable) ** 0.5
                * min(pair.median_angle_deg / 4.0, 1.0)
                * (0.1 + pair.coverage) ** 0.5
                * (1.0 + np.log1p(bridge_points) + np.log1p(bridge_views))
            )
            ranked.append((score, pair))
        ranked.sort(key=lambda entry: (-entry[0], entry[1].i, entry[1].j))
        return [pair for _, pair in ranked]

    def select_initial_pair(self) -> Optional[VerifiedPair]:
        ranked = self.rank_initial_pairs()
        return ranked[0] if ranked else None

    def _probe_seeds(self, ranked: List[VerifiedPair]) -> List[VerifiedPair]:
        selected = []
        used_views = set()
        # A continuous clip can have several locally strong sections. Probe
        # different parts of its orbit before spending every trial in one.
        clip_sources = {getattr(view, "source_path", None) for view in self.dataset}
        frame_indices = [getattr(view, "frame_index", None) for view in self.dataset]
        if len(clip_sources) == 1 and all(index is not None for index in frame_indices):
            first, last = min(frame_indices), max(frame_indices)
            sections = set()
            for pair in ranked:
                midpoint = (frame_indices[pair.i] + frame_indices[pair.j]) / 2
                section = int((midpoint - first) * self.max_seed_attempts / (last - first + 1))
                if section in sections or pair.i in used_views or pair.j in used_views:
                    continue
                selected.append(pair)
                used_views.update((pair.i, pair.j))
                sections.add(section)
                if len(selected) == self.max_seed_attempts:
                    return selected
        for pair in ranked:
            if pair.i not in used_views and pair.j not in used_views:
                selected.append(pair)
                used_views.update((pair.i, pair.j))
                if len(selected) == self.max_seed_attempts:
                    return selected
        selected_ids = {(pair.i, pair.j) for pair in selected}
        remaining = [pair for pair in ranked if (pair.i, pair.j) not in selected_ids]
        return (selected + remaining)[:self.max_seed_attempts]

    def initialize_two_views(self, pair: VerifiedPair) -> bool:
        i, j = pair.i, pair.j
        if pair.R is None or pair.t is None:
            return False
        usable = self._seed_track_matches(pair)
        if len(usable) < self.min_pnp_inliers:
            return False
        track_ids, kp_i, kp_j = np.asarray(usable, dtype=np.int64).T
        r0, t0 = np.eye(3), np.zeros((3, 1))
        xyz, valid, _ = triangulate_two_views(
            self.state.cam_K[i], r0, t0,
            self.state.cam_K[j], pair.R, pair.t,
            self.dataset[i].kps[kp_i], self.dataset[j].kps[kp_j],
            max_reproj_err=self.max_reproj_err,
            min_angle_deg=max(1.25, self.min_tri_angle_deg),
        )
        if np.count_nonzero(valid) < self.min_pnp_inliers:
            return False
        self.state.init_pair = (i, j)
        self.state.registered_cams = [i, j]
        self.state.cam_R[i], self.state.cam_t[i] = r0, t0
        self.state.cam_R[j], self.state.cam_t[j] = pair.R.copy(), pair.t.copy()
        for index in np.flatnonzero(valid):
            self._add_point(int(track_ids[index]), xyz[index], {i: int(kp_i[index]), j: int(kp_j[index])})
        return True

    def _add_point(self, track: int, xyz: np.ndarray, observations: Dict[int, int]) -> None:
        colors = [self.dataset[cid].colors[kp].astype(np.float32) for cid, kp in observations.items()]
        color = np.round(np.mean(colors, axis=0)).astype(np.uint8)
        pid = len(self.state.points3d)
        self.state.points3d.append(Point3D(pid, xyz, color, observations))
        self.state.track_to_pt3d[track] = pid
        for item in observations.items():
            self.state.kp_to_pt3d[item] = pid

    def find_2d_3d_correspondences(self, cand_img: int) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
        by_image = self.tracks.by_image[cand_img]
        active = self.state.track_to_pt3d
        shared = sorted(by_image.keys() & active.keys())
        kps = [by_image[track] for track in shared]
        pids = [active[track] for track in shared]
        xyz = np.array([self.state.points3d[pid].xyz for pid in pids], dtype=np.float64).reshape(-1, 3)
        uv = self.dataset[cand_img].kps[kps] if kps else np.zeros((0, 2))
        return xyz, uv, kps, pids

    def register_next_view(
        self, img_idx: int, pts_3d: np.ndarray, pts_2d: np.ndarray,
        kp_indices: List[int], pid_list: List[int],
    ) -> bool:
        if len(pts_3d) < self.min_pnp_inliers:
            return False
        K = self.state.cam_K[img_idx]
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, None,
            iterationsCount=1200, reprojectionError=self.pnp_ransac_thresh,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
        )
        min_support = max(self.min_pnp_inliers, int(np.ceil(0.15 * len(pts_3d))))
        if not success or inliers is None or len(inliers) < min_support:
            return False
        inliers = inliers.ravel()
        try:
            rvec, tvec = cv2.solvePnPRefineLM(pts_3d[inliers], pts_2d[inliers], K, None, rvec, tvec)
        except cv2.error:
            pass
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        errors, depth = compute_reprojection_errors(pts_3d, pts_2d, K, R, t)
        valid = (depth > 1e-6) & (errors < self.pnp_ransac_thresh)
        image = self.dataset[img_idx]
        if valid.sum() < min_support or image_coverage(pts_2d[valid], image.width, image.height) < 0.004:
            return False

        self.state.registered_cams.append(img_idx)
        self.state.cam_R[img_idx], self.state.cam_t[img_idx] = R, t
        for index in np.flatnonzero(valid):
            pid, kp = pid_list[index], kp_indices[index]
            self.state.points3d[pid].observations[img_idx] = kp
            self.state.kp_to_pt3d[(img_idx, kp)] = pid
        self._triangulate_new_points(img_idx)
        return True

    def _triangulate_new_points(self, new_img: int) -> int:
        state = self.state
        by_image = self.tracks.by_image
        unresolved = by_image[new_img].keys() - state.track_to_pt3d.keys()
        best = {}
        for other in state.registered_cams:
            if other == new_img:
                continue
            shared = sorted(unresolved & by_image[other].keys())
            if not shared:
                continue
            kp_new = [by_image[new_img][track] for track in shared]
            kp_old = [by_image[other][track] for track in shared]
            xyz, valid, angles = triangulate_two_views(
                state.cam_K[new_img], state.cam_R[new_img], state.cam_t[new_img],
                state.cam_K[other], state.cam_R[other], state.cam_t[other],
                self.dataset[new_img].kps[kp_new], self.dataset[other].kps[kp_old],
                max_reproj_err=self.max_reproj_err,
                min_angle_deg=self.min_tri_angle_deg,
            )
            for index in np.flatnonzero(valid):
                track = shared[index]
                if track not in best or angles[index] > best[track][0]:
                    best[track] = (angles[index], xyz[index], {new_img: kp_new[index], other: kp_old[index]})

        for track, (_, xyz, observations) in best.items():
            self._add_point(track, xyz, observations)
        self._refresh_observations(best.keys())
        return len(best)

    def _refresh_observations(self, track_ids, max_error: Optional[float] = None) -> None:
        state = self.state
        selected = sorted(set(track_ids) & state.track_to_pt3d.keys())
        if not selected:
            return
        error_limit = self.max_reproj_err if max_error is None else max_error
        for track in selected:
            point = state.points3d[state.track_to_pt3d[track]]
            for item in point.observations.items():
                state.kp_to_pt3d.pop(item, None)
            point.observations.clear()

        selected = set(selected)
        for cid in state.registered_cams:
            by_image = self.tracks.by_image[cid]
            shared = sorted(selected & by_image.keys())
            if not shared:
                continue
            pids = [state.track_to_pt3d[track] for track in shared]
            keys = [by_image[track] for track in shared]
            xyz = np.asarray([state.points3d[pid].xyz for pid in pids])
            errors, depth = compute_reprojection_errors(
                xyz, self.dataset[cid].kps[keys], state.cam_K[cid], state.cam_R[cid], state.cam_t[cid],
            )
            for idx in np.flatnonzero((depth > 1e-6) & (errors < error_limit)):
                pid, kp = pids[idx], keys[idx]
                state.points3d[pid].observations[cid] = kp
                state.kp_to_pt3d[(cid, kp)] = pid

        for track in selected:
            point = state.points3d[state.track_to_pt3d[track]]
            if len(point.observations) < 2:
                point.valid = False
                for item in point.observations.items():
                    state.kp_to_pt3d.pop(item, None)
                point.observations.clear()
                del state.track_to_pt3d[track]

    def _retriangulate(self, track_ids, min_views: int = 2, keep_valid_position: bool = False) -> None:
        state = self.state
        cameras = {
            cid: np.column_stack([state.cam_R[cid], state.cam_t[cid]])
            for cid in state.registered_cams
        }
        normalized = {
            cid: normalize_keypoints(self.dataset[cid].kps, state.cam_K[cid])
            for cid in state.registered_cams
        }
        for track in set(track_ids) & state.track_to_pt3d.keys():
            point = state.points3d[state.track_to_pt3d[track]]
            if len(point.observations) < min_views:
                continue
            A = []
            for cid, kp in point.observations.items():
                P = cameras[cid]
                u, v = normalized[cid][kp]
                A.extend((u * P[2] - P[0], v * P[2] - P[1]))
            rows = np.asarray(A)
            rows /= np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-12)
            _, _, vh = np.linalg.svd(rows, full_matrices=False)
            candidate = vh[-1, :3] / vh[-1, 3] if abs(vh[-1, 3]) > 1e-10 else np.full(3, np.nan)
            if keep_valid_position:
                # DLT minimizes a linear approximation. Accept a replacement
                # only when it also improves the observed pixel fit.
                before, after = 0.0, 0.0
                for cid, kp in point.observations.items():
                    xy = np.repeat(self.dataset[cid].kps[kp][None, :], 2, axis=0)
                    err, depth = compute_reprojection_errors(
                        np.stack([point.xyz, candidate]), xy,
                        state.cam_K[cid], state.cam_R[cid], state.cam_t[cid],
                    )
                    if depth[1] <= 1e-6 or not np.isfinite(err[1]) or err[1] >= self.max_reproj_err:
                        after = np.inf
                        break
                    before += err[0] ** 2
                    after += err[1] ** 2
                if not after < before:
                    continue
            point.xyz = candidate

    def perform_bundle_adjustment(self, max_nfev: int = 24, max_active_pts: int = 1800) -> bool:
        state = self.state
        cameras = state.registered_cams
        if len(cameras) < 2 or len(state.track_to_pt3d) < 10:
            return False
        ranked = sorted(state.track_to_pt3d, key=lambda tid: -len(state.points3d[state.track_to_pt3d[tid]].observations))
        per_camera = {cid: [] for cid in cameras}
        minimum = max(1, min(24, max_active_pts // len(cameras)))
        for track in ranked:
            point = state.points3d[state.track_to_pt3d[track]]
            for cid in point.observations:
                if len(per_camera[cid]) < minimum:
                    per_camera[cid].append(track)
        active = list(dict.fromkeys([track for group in per_camera.values() for track in group] + ranked))[:max_active_pts]
        points = [state.points3d[state.track_to_pt3d[tid]] for tid in active]
        indices = {cid: i for i, cid in enumerate(cameras)}
        obs = [(indices[cid], p, self.dataset[cid].kps[kp])
               for p, point in enumerate(points) for cid, kp in point.observations.items()]
        if not obs:
            return False
        obs_cam, obs_pt, obs_xy = zip(*obs)
        groups = [getattr(self.dataset[cid], "camera_group", "") or f"view_{cid}" for cid in cameras]
        fixed = [
            getattr(self.dataset[cid], "intrinsics_fixed", False)
            or getattr(self.dataset[cid], "intrinsics_source", "") == "calibrated"
            for cid in cameras
        ]
        rotations, translations, intrinsics, xyz, stats = run_bundle_adjustment(
            [state.cam_R[cid] for cid in cameras],
            [state.cam_t[cid] for cid in cameras],
            [state.cam_K[cid] for cid in cameras],
            np.asarray([point.xyz for point in points]),
            np.asarray(obs_cam, dtype=np.int32),
            np.asarray(obs_pt, dtype=np.int32),
            np.asarray(obs_xy),
            optimize_focal=self.optimize_focal,
            max_nfev=max_nfev,
            intrinsics_groups=groups,
            fixed_intrinsics=fixed,
            reference_K=[self.dataset[cid].K for cid in cameras],
        )
        if stats:
            state.ba_history.append(stats)
        if not stats or not stats.get("accepted", False):
            return False
        shared_scales = {}
        for i, cid in enumerate(cameras):
            if not fixed[i] and not np.isclose(intrinsics[i][0, 0], state.cam_K[cid][0, 0]):
                shared_scales[groups[i]] = intrinsics[i][0, 0] / self.dataset[cid].K[0, 0]
            state.cam_R[cid], state.cam_t[cid], state.cam_K[cid] = rotations[i], translations[i], intrinsics[i]
        for cid, image in enumerate(self.dataset):
            scale = shared_scales.get(getattr(image, "camera_group", ""))
            calibrated = getattr(image, "intrinsics_fixed", False) or getattr(image, "intrinsics_source", "") == "calibrated"
            if cid not in state.cam_R and scale is not None and not calibrated:
                K = image.K.copy()
                K[0, 0] *= scale
                K[1, 1] *= scale
                state.cam_K[cid] = K
        for point, position in zip(points, xyz):
            point.xyz = position
        # Every exported point must refer to the updated camera geometry.
        self._retriangulate(state.track_to_pt3d.keys() - set(active))
        self._refresh_observations(list(state.track_to_pt3d), max_error=self.max_reproj_err)
        return True

    def filter_outlier_points(self, max_err: float = 5.0) -> int:
        count = len(self.state.track_to_pt3d)
        self._refresh_observations(list(self.state.track_to_pt3d), max_error=max_err)
        return count - len(self.state.track_to_pt3d)

    def _grow(self, limit: Optional[int] = None, refine: bool = False) -> None:
        failed: Set[int] = set()
        unregistered = set(range(len(self.dataset))) - set(self.state.registered_cams)
        refined_stall = False
        while unregistered and (limit is None or len(self.state.registered_cams) < limit):
            active = self.state.track_to_pt3d
            remaining = unregistered - failed
            if remaining:
                cand = max(
                    remaining,
                    key=lambda cid: (len(self.tracks.by_image[cid].keys() & active.keys()), -cid),
                )
                count = len(self.tracks.by_image[cand].keys() & active.keys())
            else:
                count = 0
            if count < self.min_pnp_inliers:
                if refine and not refined_stall and self.perform_bundle_adjustment(max_nfev=16):
                    refined_stall = True
                    self.state.registration_retries += 1
                    failed.clear()
                    continue
                break

            args = self.find_2d_3d_correspondences(cand)
            if self.register_next_view(cand, *args):
                unregistered.remove(cand)
                failed.clear()
                refined_stall = False
                if refine and self.ba_interval > 0 and len(self.state.registered_cams) % self.ba_interval == 0:
                    self.perform_bundle_adjustment(max_nfev=12, max_active_pts=1000)
            else:
                failed.add(cand)

    def run(self) -> ReconstructionState:
        t0 = time.perf_counter()
        attempts = []
        best = self._new_state()
        trials = []
        ranked = self.rank_initial_pairs()
        for pair in self._probe_seeds(ranked):
            self.state = self._new_state()
            success = self.initialize_two_views(pair)
            if success:
                self._grow(limit=self.seed_probe_views)
            n_views = len(self.state.registered_cams)
            supported = sum(len(point.observations) >= 3 for point in self.state.points3d if point.valid)
            attempts.append({
                "seed": [pair.i, pair.j],
                "probe_registered": n_views,
                "probe_tracks_3plus": supported,
                "median_angle_deg": pair.median_angle_deg,
                "coverage": pair.coverage,
            })
            score = (n_views, supported, len(self.state.track_to_pt3d))
            if success:
                trials.append((score, self.state, attempts[-1]))

        # Similar short probes can have very different final coverage. Expand
        # the strongest few before doing the final optimization on the winner.
        best_score = (-1, -1, -1)
        for _, trial, diagnostic in sorted(trials, key=lambda item: item[0], reverse=True)[:3]:
            self.state = trial
            self._grow(refine=self.use_ba)
            diagnostic["expanded_registered"] = len(self.state.registered_cams)
            supported = sum(len(point.observations) >= 3 for point in self.state.points3d if point.valid)
            score = (len(self.state.registered_cams), supported, len(self.state.track_to_pt3d))
            if score > best_score:
                best, best_score = self.state, score
            if len(self.state.registered_cams) == len(self.dataset):
                break
        self.state = best
        self.state.seed_attempts = attempts
        self.state.num_seed_candidates = len(ranked)
        if self.state.registered_cams:
            self._retriangulate(self.state.track_to_pt3d, min_views=3, keep_valid_position=True)
            self.state.pre_ba_errors = self.state.compute_all_reprojection_errors()
            if self.use_ba:
                self.perform_bundle_adjustment()
            else:
                self.filter_outlier_points(max_err=self.max_reproj_err)
            self.state.post_ba_errors = self.state.compute_all_reprojection_errors()
        self.state.elapsed_sec = time.perf_counter() - t0
        return self.state
