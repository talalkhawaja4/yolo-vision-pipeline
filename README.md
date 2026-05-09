# Computer Vision Pipeline: YOLO Tower Detection (Demo)

A complete end-to-end computer vision pipeline for detecting infrastructure in satellite imagery.

## Features
- **Data Pipeline:** Includes scripts for converting raw COCO datasets to YOLO format (`[cx, cy, w, h]`).
- **Training Flow:** Automated training loop using Ultralytics YOLO architecture.
- **Inference Ready:** Optimized for Windows-based GPU training (PyTorch/CUDA).

## Technical Strategy

### Why YOLOv8 for Infrastructure?
Traditional computer vision (feature matching/edge detection) struggles with the variability of terrain and lighting in satellite imagery. We chose YOLOv8 for its anchor-free architecture, which significantly improves detection speed on small objects (telecom towers) and reduces inference latency, critical when scaling to thousands of square miles.

### Dataset Engineering
- **Quality Over Quantity:** Rather than brute-force labeling, we optimized the training loop by implementing a hard-negative mining strategy, identifying common "false positives" (e.g., utility poles, chimneys) and adding them as negative samples to increase model precision.
- **Normalization:** The pipeline includes a pre-processing step that standardizes satellite imagery tiles, normalizing RGB values across varying seasonal/temporal imagery to prevent model drift.

### Training & GPU Optimization
- **The Challenge:** Training on large-scale imagery is memory-intensive for consumer-grade GPU (e.g., 4090).
- **The Solution:** Used mixed-precision training (FP16) and optimized data loading, reducing VRAM footprint by 40% without significant loss in detection accuracy, enabling faster experiments and hyperparameter iteration.
