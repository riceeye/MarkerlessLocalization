#!/usr/bin/env python
import os
import argparse
import random
import math
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageDraw

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)

def compute_bounding_sphere(mesh):
    verts = mesh.verts_packed()
    center = verts.mean(dim=0)
    radius = (verts - center).norm(dim=1).max().item()
    return center, radius

def random_augment(image_np):
    """
    Apply random brightness and contrast adjustments.
    Expects image_np values in [0,1].
    """
    image = Image.fromarray((np.clip(image_np, 0, 1) * 255).astype(np.uint8))
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    return image

def compute_bbox(cameras, mesh, image_size):
    """
    Compute the smallest bounding box (x, y, width, height) in pixel coordinates
    that fully contains the projected mesh vertices.
    """
    verts = mesh.verts_packed().unsqueeze(0)  # shape (1, N, 3)
    # Pass image_size as a tuple (width, height)
    verts_screen = cameras.transform_points_screen(verts, image_size=(image_size, image_size))
    verts_screen = verts_screen[0, :, :2]  # use x,y coordinates
    x_min = verts_screen[:, 0].min().item()
    y_min = verts_screen[:, 1].min().item()
    x_max = verts_screen[:, 0].max().item()
    y_max = verts_screen[:, 1].max().item()
    return {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min,
        "height": y_max - y_min
    }

def annotate_image(cameras, mesh, image_size, filename, label="object"):
    bbox = compute_bbox(cameras, mesh, image_size)
    annotation = {
        "filename": filename,
        "size": 0,  # file size can be added later if needed
        "regions": [{
            "shape_attributes": {
                "name": "rect",
                "x": bbox["x"],
                "y": bbox["y"],
                "width": bbox["width"],
                "height": bbox["height"]
            },
            "region_attributes": {
                "label": label
            }
        }],
        "file_attributes": {}
    }
    return annotation, bbox

def draw_bbox(image, bbox, label="object"):
    draw = ImageDraw.Draw(image)
    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    rect_coords = [(x, y), (x + w, y + h)]
    draw.rectangle(rect_coords, outline="red", width=3)
    draw.text((x, y), label, fill="red")
    return image

def main():
    parser = argparse.ArgumentParser(
        description="Generate a training set with original and annotated images, and a JSON file with annotations."
    )
    parser.add_argument("input_obj", type=str,
                        help="Path to the input OBJ file (with associated MTL/texture files).")
    parser.add_argument("output_folder", type=str,
                        help="Main folder for the training set (will contain two subfolders and a JSON file).")
    parser.add_argument("--num_images", type=int, default=400,
                        help="Number of images to generate (default: 400)")
    parser.add_argument("--base_image_size", type=int, default=2048,
                        help="Base output image size in pixels (default: 2048)")
    parser.add_argument("--faces_per_pixel", type=int, default=10,
                        help="Faces per pixel for rasterization (default: 10)")
    parser.add_argument("--margin_min", type=float, default=1.2,
                        help="Minimum margin multiplier (default: 1.2)")
    parser.add_argument("--margin_max", type=float, default=1.4,
                        help="Maximum margin multiplier (default: 1.4)")
    parser.add_argument("--radius_threshold", type=float, default=0.1,
                        help="If the object's radius is below this value, resolution is increased (default: 0.1)")
    parser.add_argument("--min_camera_dist", type=float, default=1.0,
                        help="Minimum allowed camera distance (default: 1.0)")
    parser.add_argument("--label", type=str, default="object",
                        help="Label for annotation (default: object)")
    args = parser.parse_args()

    # Create main folder structure: two subfolders ("original" and "annotated")
    original_folder = os.path.join(args.output_folder, "original")
    annotated_folder = os.path.join(args.output_folder, "annotated")
    os.makedirs(original_folder, exist_ok=True)
    os.makedirs(annotated_folder, exist_ok=True)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    mesh = load_objs_as_meshes([args.input_obj], device=device)
    center, radius = compute_bounding_sphere(mesh)
    print(f"Computed mesh center: {center}, radius: {radius}")

    # Adjust resolution if the object is very small.
    if radius < args.radius_threshold:
        scale_factor = args.radius_threshold / radius
        image_size = int(args.base_image_size * scale_factor)
        print(f"Object is small (radius {radius:.4f}). Increasing image resolution to {image_size}x{image_size}.")
    else:
        image_size = args.base_image_size

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=args.faces_per_pixel,
    )

    all_annotations = {}

    for i in range(args.num_images):
        fov = random.uniform(30, 70)
        half_fov_rad = math.radians(fov / 2)
        safe_dist = radius / math.tan(half_fov_rad)
        margin = random.uniform(args.margin_min, args.margin_max)
        dist = max(safe_dist * margin, args.min_camera_dist)

        elev = random.uniform(10, 60)
        azim = random.uniform(0, 360)

        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=center.unsqueeze(0))
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

        light_x = random.uniform(-3, 3)
        light_y = random.uniform(-3, 3)
        light_z = random.uniform(-3, -1)
        lights = PointLights(device=device, location=[[light_x, light_y, light_z]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
        )

        images = renderer(mesh)
        image_np = images[0, ..., :3].cpu().numpy()
        gamma = 2.2
        image_np = np.clip(image_np, 0, 1) ** (1 / gamma)
        image_aug = random_augment(image_np)

        filename = f"image_{i:04d}.png"
        orig_path = os.path.join(original_folder, filename)
        image_aug.save(orig_path)
        print(f"Saved original image to {orig_path}")

        annotation, bbox = annotate_image(cameras, mesh, image_size, filename, label=args.label)
        all_annotations[filename] = annotation

        annotated_img = image_aug.copy()
        annotated_img = draw_bbox(annotated_img, bbox, label=args.label)
        anno_path = os.path.join(annotated_folder, filename)
        annotated_img.save(anno_path)
        print(f"Saved annotated image to {anno_path}")

    json_path = os.path.join(args.output_folder, "annotations.json")
    with open(json_path, "w") as f:
        json.dump(all_annotations, f, indent=4)
    print(f"Saved annotations JSON to {json_path}")
    print("Training set generation complete.")

if __name__ == "__main__":
    main()
