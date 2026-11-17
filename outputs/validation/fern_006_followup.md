# Fern 006 follow-up experiments

These experiments use the Fern 006 input and reference reconstruction published at
commit `90784e31fa0cfe480fed130ce199ca8fb2c47624`.

## Fixed-map registration diagnostic

The diagnostic keeps the 4,657 published landmarks and camera poses fixed.
Every second registered frame in chronological order gives 12 reference images.
The existing Red Solace localization code associates RootSIFT keypoints with
projected landmarks, matches the query to the references, and tests the
resulting 2D-3D correspondences with robust PnP and pose refinement. Each
originally registered query excludes itself from the reference set. Those
queries serve as a pipeline check on the same map.

Of the 48 originally unregistered frames, 3 had fewer than 20 candidate
correspondences and 45 reached the solver without obtaining a qualifying
pose consensus. All 48 fall inside the original gap between 11.31 and 59.94
seconds. They have a median of 52 candidate 2D-3D correspondences, compared
with 791 for the 24 original registered frames. The diagnostic recovered 23
of those 24 poses under its acceptance criteria. This tests correspondence
support against the completed reference map using the localization code.
The underlying capture and matching factors are not individually isolated.

- [Per-frame observations](fern_006_diagnostics.csv)
- [Correspondence and inlier coverage plot](fern_006_diagnostics.png)

## Fixed 72-frame replacement test

Both groups use full-quality decoded frames from the same clip, the same
coordinate preparation, and the same RootSIFT, matching, verification,
initialization and bundle-adjustment settings. The control uses the original
72 source frame indices. The treatment considers alternate frames at every
other slot inside the published gap so neighboring reference indices stay
fixed. The selector maximizes the weaker of the two adjacent bidirectional
Lucas-Kanade overlap scores. A replacement must improve that score, retain
at least 90% of the original slot's Laplacian-variance sharpness, and pass the
near-duplicate checks. The selector does not use reconstruction outcomes.

| Metric | Original indices rerun | Replacement |
|---|---:|---:|
| Registered frames | 24 / 72 | 25 / 72 |
| Registrations inside the original gap | 0 | 1 |
| Longest gap between registered views | 48.64 s | 48.39 s |
| Sparse points | 4,702 | 4,279 |
| Median sparse reprojection error | 0.770 px | 0.816 px |
| Verified image-pair edges | 272 | 279 |

The selector replaced 12 frames. For those slots the median improvement in
the weaker adjacent optical-flow overlap was 4.4 percentage points. The
replacement run preserved all 24 previously registered indices and
registered frame 693 at 11.55 seconds, just inside the original gap.
The longest gap contracted by 0.25 seconds. This selection strategy therefore
recovered one edge view while leaving the middle of the gap unresolved.
The additional pairwise edges did not translate to a substantial increase
in registration coverage. The evidence is specific to this clip and the
tested selector. The two runs use different retained observations, so their
reprojection summaries do not measure an accuracy change on shared points.

The published reconstruction also registers 24 views. Its 4,657 sparse
points refer to the published run. The 4,702 entry is the newly rerun
control used for this paired comparison.

- [Settings, chosen replacements and reconstruction metrics](fern_006_bridge_validation.json)
- [Registration timeline](fern_006_bridge_validation.png)

## Reproduction

Install the project requirements. From the repository root, run:

```sh
python outputs/validation/fern_006_diagnostics.py
git lfs pull --include="dataset/Fern/video/006.mp4"
python outputs/validation/fern_006_bridge_validation.py --output outputs/validation/fern_006_bridge_validation.json
```

The recorded runs used OpenCV 4.12.0, NumPy 2.3.1 and SciPy 1.17.1. The
validation video was checked against its Git LFS SHA-256
`41f98107bde39df663267dd6655099c67e10e70193d44dfb419363b8dcec3288`.

