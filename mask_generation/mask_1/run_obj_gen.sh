#!/bin/bash

# Define variables
MODEL_PATH="cube.obj"
OUTPUT_FOLDER="cube_images"
NUM_IMAGES=15
IMAGE_SIZE=512

# Run the Python script
python generate_obj_images.py "$MODEL_PATH" \
    --output_folder "$OUTPUT_FOLDER" \
    --num_images "$NUM_IMAGES" \
    --image_size "$IMAGE_SIZE"
