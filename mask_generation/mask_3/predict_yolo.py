from ultralytics import YOLO
import cv2
import os

# Load the trained YOLOv11 segmentation model
model = YOLO("runs/segment/yolov11_01/weights/best.pt")

# Path to your input images
source = "runs/segment/test_images"

# Output directory to save PNGs
output_dir = "runs/segment/test_output"
os.makedirs(output_dir, exist_ok=True)

# Run inference
results = model(source, stream=True)  # generator of Results objects

for i, result in enumerate(results):
    # Render result (includes masks, boxes, and labels)
    rendered_image = result.plot()

    # Get original image name from path (preserve name for saving)
    path = result.path  # full path to input image
    filename = os.path.basename(path)
    filename_wo_ext = os.path.splitext(filename)[0]

    # Save rendered image as PNG
    out_path = os.path.join(output_dir, f"{filename_wo_ext}_pred.png")
    cv2.imwrite(out_path, rendered_image)

    print(f"Saved: {out_path}")
