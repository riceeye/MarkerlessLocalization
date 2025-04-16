#!/bin/bash

# A simple bash script to render any OBJ file to a PNG image using the render_mesh.py script.
# Usage: ./render.sh input_obj_file output_png_file

#if [ "$#" -ne 2 ]; then
#    echo "Usage: $0 input_obj_file output_png_file"
#    exit 1
#fi

INPUT_OBJ="cube.obj"
OUTPUT_PNG="new_data"

# Run the Python rendering script with the provided input and output filenames.
python gen_mesh_6.py "$INPUT_OBJ" "$OUTPUT_PNG"
