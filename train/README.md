<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/poster_sparseInst-train.jpg"/>  

# Train SparseInst

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../../supervisely-ecosystem/SparseInst/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/SparseInst)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/SparseInst/train.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/SparseInst/train.png)](https://supervisely.com)

</div>

# Overview

SparseInst is a conceptually novel, efficient, and fully convolutional framework for real-time instance segmentation. In contrast to region boxes or anchors (centers), SparseInst adopts a sparse set of instance activation maps as object representation, to highlight informative regions for each foreground objects. Then it obtains the instance-level features by aggregating features according to the highlighted regions for recognition and segmentation. The bipartite matching compels the instance activation maps to predict objects in a one-to-one style, thus avoiding non-maximum suppression (NMS) in post-processing. Owing to the simple yet effective designs with instance activation maps, SparseInst has extremely fast inference speed and achieves 40 FPS and 37.9 AP on COCO (NVIDIA 2080Ti), significantly outperforms the counter parts in terms of speed and accuracy.

![sparseinst architecture](https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparseinst_architecture.png)

# How To Run

**Step 0.** Run the app from context menu of the project with annotations or from the Ecosystem

**Step 1.** Select if you want to use cached project or redownload it

<img src="https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparseinst_train_0.png" width="100%" style='padding-top: 10px'>

**Step 2.** Select train / val split

<img src="https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparseinst_train_1.png" width="100%" style='padding-top: 10px'>

**Step 3.** Select the classes you want to train model on

<img src="https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparseinst_train_2.png" width="100%" style='padding-top: 10px'>

**Step 4.** Select the model you want to train

<img src="https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparseinst_train_3.png" width="100%" style='padding-top: 10px'>

**Step 5.** Configure hyperaparameters and select whether you want to use model evaluation and convert checkpoints to ONNX and TensorRT

<img src="https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparseinst_train_4.png" width="100%" style='padding-top: 10px'>

**Step 6.** Enter experiment name and start training

<img src="https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparseinst_train_5.png" width="100%" style='padding-top: 10px'>

**Step 7.** Monitor training progress

<img src="https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparseinst_train_6.png" width="100%" style='padding-top: 10px'>

# Acknowledgment

This app is based on the great work [SparseInst](https://github.com/hustvl/SparseInst).