<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/poster_sparseInst-serve.jpg"/>  

# Serve SparseInst

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../../supervisely-ecosystem/SparseInst/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/SparseInst)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/SparseInst/serve.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/SparseInst/serve.png)](https://supervisely.com)

</div>

# Overview

SparseInst is a conceptually novel, efficient, and fully convolutional framework for real-time instance segmentation. In contrast to region boxes or anchors (centers), SparseInst adopts a sparse set of instance activation maps as object representation, to highlight informative regions for each foreground objects. Then it obtains the instance-level features by aggregating features according to the highlighted regions for recognition and segmentation. The bipartite matching compels the instance activation maps to predict objects in a one-to-one style, thus avoiding non-maximum suppression (NMS) in post-processing. Owing to the simple yet effective designs with instance activation maps, SparseInst has extremely fast inference speed and achieves 40 FPS and 37.9 AP on COCO (NVIDIA 2080Ti), significantly outperforms the counter parts in terms of speed and accuracy.

![sparseinst architecture](https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparseinst_architecture.png)

# How To Run

**Step 1.** Select pretrained model architecture and press the **Serve** button

![pretrained_models](https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparseinst_serve_0.png)

**Step 2.** Wait for the model to deploy

![deployed](https://github.com/supervisely-ecosystem/SparseInst/releases/download/v0.0.1/sparseinst_serve_1.png)

# Acknowledgment

This app is based on the great work [SparseInst](https://github.com/hustvl/SparseInst).