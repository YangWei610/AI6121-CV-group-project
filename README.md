# AI6121-CV-group-project

Group project for AI6121 Computer Vision.

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

JPG and MP4 files are stored with Git LFS. Install [Git LFS](https://git-lfs.com/) before downloading the repository, then run:

```sh
git lfs install
git clone https://github.com/YangWei610/AI6121-CV-group-project.git
cd AI6121-CV-group-project
git lfs pull
```
