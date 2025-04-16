#!/bin/bash
# train_yolo.sh

# Set paths to model configuration, pretrained weights, and dataset YAML file.
MODEL_YAML="yolo11n-seg.yaml"
PRETRAINED_WEIGHTS="yolo11n.pt"
DATA_YAML="./new_data/new_data.yaml"  # updated location

# Set training hyperparameters.
EPOCHS=50
IMGSZ=640
BATCH=16
RUN_NAME="yolov11_01"
EXIST_OK="--exist_ok"

# Run the Python training script.
python train_yolo.py "$MODEL_YAML" "$PRETRAINED_WEIGHTS" "$DATA_YAML" --epochs "$EPOCHS" --imgsz "$IMGSZ" --batch "$BATCH" --run_name "$RUN_NAME" $EXIST_OK
