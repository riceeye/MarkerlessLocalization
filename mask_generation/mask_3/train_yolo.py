#!/usr/bin/env python
import os
import argparse
import random
import math
import torch
from ultralytics import YOLO

def save_checkpoint(model, base_filename="checkpoint", ext=".pt"):
    """
    Save the model checkpoint using an auto-incremented filename.
    If "checkpoint_01.pt" exists, it will save as "checkpoint_02.pt", etc.
    """
    i = 1
    filename = f"{base_filename}_{i:02d}{ext}"
    while os.path.exists(filename):
        i += 1
        filename = f"{base_filename}_{i:02d}{ext}"
    
    model.save(filename)
    print(f"Checkpoint saved as {filename}")

def main():
    parser = argparse.ArgumentParser(
        description="Train a YOLOv8 segmentation model using a YAML config and save checkpoints with auto-incremented filenames."
    )
    # Path to the model YAML file and pretrained weights.
    parser.add_argument("model_yaml", type=str,
                        help="Path to the YOLO model YAML configuration (e.g., 'yolo11n-seg.yaml').")
    parser.add_argument("pretrained_weights", type=str,
                        help="Path to the pretrained weights file (e.g., 'yolo11n.pt').")
    # Data YAML containing dataset information.
    parser.add_argument("data_yaml", type=str,
                        help="Path to the dataset YAML file (e.g., 'data.yaml').")
    # Training hyperparameters.
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--run_name", type=str, default="yolov8_custom",
                        help="Name of the training run (default: yolov8_custom)")
    parser.add_argument("--exist_ok", action="store_true",
                        help="Overwrite previous runs with the same name")
    args = parser.parse_args()

    # Build model from YAML and load pretrained weights.
    model = YOLO(args.model_yaml).load(args.pretrained_weights)

    # Train the model.
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.run_name,
        exist_ok=args.exist_ok
    )

    # Save checkpoint with auto-incremented filename.
    save_checkpoint(model, base_filename="checkpoint", ext=".pt")

if __name__ == "__main__":
    main()
