#!/bin/bash

# ======= CONFIGURABLE VARIABLES =======
SCRIPT="surgery.py"                            # Name of your Python script
DATASET_PATH="/Users/timli/MarkerlessLocalization/mask_generation/mask_3/masking/data/"                       # Root dataset directory
WEIGHTS="coco"                                     # Or path to a .h5 file like "path/to/weights.h5"
# LOG_DIR="logs"                                     # Optional: log directory
# ======================================

# Run training command
python $SCRIPT train \
  --dataset "$DATASET_PATH" \
# --subset train \
  --weights "$WEIGHTS" \
# --logs "$LOG_DIR"
