# Tile-Based Multi-Scale Detection: Designing a Simple and Effective Pipeline
## Overview

This repository contains the implementation of the Tile-Based Multi-Scale Detection (TBMSD) pipeline, designed for object detection on large images. The method leverages tile-based inference, multi-scale processing, and adaptive merging strategies to enhance detection accuracy.

This work is based on our paper:
Tile-Based Multi-Scale Detection: Designing a Simple and Effective Pipeline (add arXiv link or DOI when available).

## Features

- Efficient object detection on high-resolution images.
- Multi-scale inference for better detection of objects of varying sizes.
- Adaptive tile merging strategies to improve consistency.
- Supports various aggregation techniques (NMS, Soft-NMS, Score Voting).
- Implemented using YOLOv11 (Ultralytics).

## Installation

First, clone the repository and install dependencies using Poetry:
```
git clone https://github.com/your-repo/tbmsd.git
cd tbmsd
poetry install
```

## Usage
Use the following code to perform inference with multi-scale patch processing:
```
from tbmsd.inference import YOLOPatchInferenceWrapper, MultiPatchInference  

model = YOLOPatchInferenceWrapper("yolo11n-obb.pt")  
inferencer = MultiPatchInference(model)  
results = inferencer(img, patch_sizes=(1024, 2048), overlap=10)  
```

## Evaluation

Run evaluation on a dataset using the provided script:
```
python examples/predict_and_eval.py --dataset <path_to_dataset> --weights yolo11n-obb.pt
```

## Citation

If you use this code, please cite our paper:

```yaml
@article{your_paper,
  author    = {Your Name and Co-authors},
  title     = {Tile-Based Multi-Scale Detection: Designing a Simple and Effective Pipeline},
  journal   = {ArXiv/Conference},
  year      = {2025}
}
```
