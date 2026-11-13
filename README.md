# AI6121-CV-group-project

Group project for AI6121 Computer Vision.

## Project files

| Location | Contents |
|---|---|
| [Updated report](output/pdf/AI6121_SfM_Report_MultiVideo.pdf) | English report with stereo and meshing methods, both model comparisons, and six-video camera alignment |
| [Report source](report/report.tex) | Editable LaTeX source |
| [Red Solace surface](outputs/final/red-solace/surface/red_solace_surface.glb) | Colored GLB mesh with 63,770 triangles |
| [Fern surface](outputs/final/fern/surface/fern_surface.glb) | Observed wooden ribs and base from archived clip 004, with 22,557 triangles |
| `outputs/final/{fern,red-solace}/surface/` | GLB, PLY with vertex normals, colored surface / wireframe / face-normal figure, and metrics |
| `outputs/validation/` | Controlled comparisons and the archived Fern inputs for reproducing its surface |
| `outputs/case-study/` | Point-cloud views, six-video pose JSON, and the corresponding camera overlay |
| `sfm/`, `tools/` | Incremental SfM, stereo matching, mesh extraction and rendering source |

## Dataset

The dataset is unpacked under `dataset/`, with the original scene folders and file names preserved:

```text
dataset/
├── Fern/
│   ├── image/       # 74 JPG images
│   └── video/       # 8 MP4 videos
└── Red_Solace/
    ├── image/       # 41 JPG images
    └── video/       # 6 MP4 videos
```

Total: **129 files** (115 JPG images and 14 MP4 videos), **2,954,844,448 bytes** (approximately **2.75 GiB**).

The complete project ZIP includes the original JPG and MP4 files. Git
checkouts of the dataset use [Git LFS](https://git-lfs.com/):

```sh
git lfs install
git clone https://github.com/YangWei610/AI6121-CV-group-project.git
cd AI6121-CV-group-project
git lfs pull
```

## Run the reconstruction

From the project directory, install the dependencies and run both video scenes:

```sh
python -m pip install -r requirements.txt
python run_experiments.py
```

The video inputs are `Fern/video/006.mp4` (72 target keyframes) and
`Red_Solace/video/003.mp4` (40 target keyframes).
The sampler scores keyframes by overlap, image motion and sharpness. The mapper
verifies image pairs, builds consistent multi-view tracks, tests multiple seed
pairs and refines the resulting poses and points.

```sh
python run_experiments.py --scene fern --keyframes 72 --output-dir outputs/fern_72
python run_experiments.py --scene red-solace --ablation --output-dir outputs/red_solace_ablation
python run_experiments.py --scene fern --input photos --output-dir outputs/fern_photos
```

The ablation runs SIFT, RootSIFT and ORB with the same views and thresholds.
ORB uses Hamming distance. Metrics record registered views, match connectivity,
track support, reprojection errors, seed trials and camera parameters.
`point_clouds/` contains colored PLY files. `figures/` contains the view graph,
seed matches and point cloud plots. Semi-dense output is generated when sparse
support passes the checks in the runner. Added points must agree in at least
three views. The grid coverage metric refers to the whole image, including background.

### Camera calibration

Without calibration the focal length is an initial estimate, refined jointly
for frames from the same clip. Pass `--calibration camera.json` to use measured
intrinsics. Each key is a dataset-relative source file. Use its original decoded
resolution and the OpenCV order for distortion coefficients. The values below
illustrate the format and should be replaced with measured values.

```json
{
  "Fern/video/006.mp4": {
    "width": 3840,
    "height": 2160,
    "K": [[2500, 0, 1920], [0, 2500, 1080], [0, 0, 1]],
    "distortion": [0, 0, 0, 0, 0],
    "group": "fern_006"
  }
}
```

The loader scales the matrix with the resized frame, undistorts the image
before extracting features and keeps the calibrated intrinsics fixed.

### Validation on the supplied data

| Experiment | Registered views | Result |
|---|---:|---|
| Original pipeline, 50 archived Fern keyframes | 6 / 50 | 1,694 sparse points |
| Revised pipeline, same frames and descriptor matches | 26 / 50 | 4,931 sparse points |
| Revised pipeline, Fern `006.mp4` | 24 / 72 | 4,657 sparse points; middle section remains unregistered |
| Revised pipeline, Red Solace `003.mp4` | 40 / 40 | 12,693 sparse and 16,481 semi-dense points |

The archived-frame replay shares 2,400 RootSIFT features per frame, the same
raw matches and the same initial intrinsics across both pipelines. The videos
in the final two rows use different inputs. The Fern 72-frame replay tested
focal factors of 0.72, 0.85, 0.95 and 1.05 and registered 24 frames with each.
These results measure image registration and observed track support. Absolute
scale needs a scale reference, and surface accuracy needs independent geometry
for comparison.

## Reproduce the surface meshes

Run the following from the project root after installing the requirements.
The project ZIP includes the saved camera poses, stereo depth samples and
reference keyframes.

```sh
python -m tools.red_solace_surface --output-dir outputs/reproduced/red-solace
python -m tools.fern_surface_patch
```

Red Solace uses depth fusion from 35 reference views of clip 003. Fern's
selected mesh uses cross-view-checked depths from frame 662 of archived clip
004. The Fern command writes to `outputs/validation/fern_archive/reproduced/`.
The separate `tools.fern_surface` experiment uses clip 006.

Regenerate the Red Solace depth samples using the included reference frames
and pose estimates.

```sh
python -m tools.red_solace_stereo --scene red-solace --coverage expanded
```

Regenerate Fern's archived depth inputs using the archived poses, keyframes
and sparse cloud.

```sh
python -m tools.red_solace_stereo --scene fern --coverage expanded --metrics outputs/validation/fern_archive/experiment_metrics.json --keyframes outputs/validation/fern_archive/keyframes --sparse outputs/validation/fern_archive/fern_same_50_frames_sparse.ply --output outputs/validation/fern_archive/stereo_samples_expanded.npz
```

The three-panel renderings use `tools/red_solace_surface_views.py` for either
scene. Run `python -m tools.red_solace_surface_views --help` for the mesh,
output, scene and saved display-frame options.

## Reproduce the Red Solace camera overlay

The saved clip 003 poses and landmarks define the reference coordinate
system. Five additional clips contribute 33 registered frames, for 73
poses across six clips. The table in the report separates sampled frames
from successful registrations.

Run these commands from the project root with the bundled dataset and
anchor keyframes.

~~~sh
python -m tools.red_solace_multivideo
python -m tools.case_study_views --multi-video-poses outputs/case-study/red-solace-multi-video-poses.json
~~~

The report builds from `report/report.tex` and the included result figures.
Run `pdflatex report.tex` twice from the `report/` directory to rebuild it.
