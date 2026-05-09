#!/usr/bin/env python3
"""
YOLOv8 Training Script for Aerocell Tower Detection

Matches the LG report training setup:
- 400 epochs
- 1280px input resolution
- YOLOv8x model
- ~15 hours on RTX 4090

Usage:
    python src/train.py --config configs/train_400ep_1280.yaml

    # Or with CLI overrides:
    python src/train.py --epochs 400 --imgsz 1280 --batch 8
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

import yaml
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train(
    config_path: str = None,
    model: str = "yolov8x.pt",
    data: str = "datasets/aerocell.yaml",
    epochs: int = 400,
    imgsz: int = 1280,
    batch: int = 8,
    device: str = "0",
    project: str = "runs/train",
    name: str = None,
    resume: bool = False,
    **kwargs
):
    """
    Train YOLOv8 model on Aerocell dataset.

    Args:
        config_path: Path to YAML config file
        model: Model to use (yolov8n/s/m/l/x.pt)
        data: Path to dataset YAML
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: CUDA device(s) to use
        project: Project directory for runs
        name: Run name
        resume: Resume from last checkpoint
    """
    # Load config if provided
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        # Override with config values
        model = config.get('model', model)
        data = config.get('data', data)
        epochs = config.get('epochs', epochs)
        imgsz = config.get('imgsz', imgsz)
        batch = config.get('batch', batch)
        device = config.get('device', device)
        project = config.get('project', project)
        name = config.get('name', name)
        # Merge additional kwargs from config
        for key, value in config.items():
            if key not in ['model', 'data', 'epochs', 'imgsz', 'batch', 'device', 'project', 'name']:
                kwargs[key] = value

    # Generate run name if not provided
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"aerocell_{epochs}ep_{imgsz}px_{timestamp}"

    print("=" * 60)
    print("YOLOv8 Training - Aerocell Tower Detection")
    print("=" * 60)
    print(f"Model: {model}")
    print(f"Dataset: {data}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch}")
    print(f"Device: {device}")
    print(f"Project: {project}")
    print(f"Run Name: {name}")
    print("=" * 60)

    # Initialize model
    yolo_model = YOLO(model)

    # Start training
    results = yolo_model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        **kwargs
    )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Results saved to: {project}/{name}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Aerocell dataset")
    parser.add_argument('--config', type=str, default='configs/train_400ep_1280.yaml',
                        help='Path to config file')
    parser.add_argument('--model', type=str, default='yolov8x.pt',
                        help='Model to use')
    parser.add_argument('--data', type=str, default='datasets/aerocell.yaml',
                        help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=400,
                        help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=1280,
                        help='Input image size')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device(s)')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Project directory')
    parser.add_argument('--name', type=str, default=None,
                        help='Run name')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')

    args = parser.parse_args()

    train(
        config_path=args.config,
        model=args.model,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume
    )


if __name__ == '__main__':
    main()
