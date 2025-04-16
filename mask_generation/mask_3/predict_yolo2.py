from ultralytics import YOLO
import cv2
import os
import numpy as np

# Load your model
model = YOLO("runs/segment/yolov11_01/weights/best.pt")

# Input and output paths
source = "runs/segment/test_images"
output_dir = "runs/segment/test_output_filtered"
os.makedirs(output_dir, exist_ok=True)

# Confidence threshold
CONF_THRESHOLD = 0.9

# Run inference
results = model(source, stream=True)

for result in results:
    confidences = result.boxes.conf.cpu().numpy() if result.boxes else []

    # If there's no detection or no high-confidence object, skip
    if len(confidences) == 0 or np.max(confidences) < CONF_THRESHOLD:
        print(f"Skipped (no confidence ≥ {CONF_THRESHOLD}): {result.path}")
        continue

    # Render the full result (YOLO draws only the boxes/masks that passed the model's internal threshold)
    rendered_image = result.plot()

    # Save output image
    filename = os.path.splitext(os.path.basename(result.path))[0]
    out_path = os.path.join(output_dir, f"{filename}_filtered.png")
    cv2.imwrite(out_path, rendered_image)
    print(f"Saved: {out_path}")
