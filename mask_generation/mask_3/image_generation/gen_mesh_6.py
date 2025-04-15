#!/usr/bin/env python
import os
import argparse
import random
import math
import json
import cv2
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

#######################
# Utility Functions
#######################

def compute_bounding_sphere(mesh):
    """Computes the center and radius of the mesh in world coordinates."""
    verts = mesh.verts_packed()
    center = verts.mean(dim=0)
    radius = (verts - center).norm(dim=1).max().item()
    return center, radius

def random_augment(image_np):
    """
    Apply a mild random brightness and contrast adjustment.
    Expects image_np values in [0,1] and returns a PIL Image.
    """
    image = Image.fromarray((np.clip(image_np, 0, 1) * 255).astype(np.uint8))
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))
    return image

def compute_bbox(cameras, mesh, image_size):
    """
    Computes the smallest bounding box (x, y, width, height) in pixel coordinates
    that fully contains the projected mesh vertices.
    """
    verts = mesh.verts_packed().unsqueeze(0)  # [1, N, 3]
    verts_screen = cameras.transform_points_screen(verts, image_size=(image_size, image_size))
    verts_screen = verts_screen[0, :, :2]  # use x, y
    x_min = verts_screen[:, 0].min().item()
    y_min = verts_screen[:, 1].min().item()
    x_max = verts_screen[:, 0].max().item()
    y_max = verts_screen[:, 1].max().item()
    return {"x": x_min, "y": y_min, "width": x_max - x_min, "height": y_max - y_min}

def extract_polygon_from_grabcut(pil_image, bbox, num_points=50, iter_count=10):
    """
    Uses GrabCut to extract a foreground mask from the rendered PIL image using the bbox
    as the initial rectangle. Then finds the largest contour and resamples it to return
    exactly num_points points.
    Returns two lists: all_points_x and all_points_y (in absolute pixel coordinates).
    """
    # Convert PIL image (RGB) to BGR for OpenCV
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"]))
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype('uint8')
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return [], []
    largest_contour = max(contours, key=cv2.contourArea)
    pts = largest_contour.reshape(-1, 2).astype(np.float32)

    # Compute cumulative arc length of the contour.
    distances = [0]
    for i in range(1, len(pts)):
        d = np.linalg.norm(pts[i] - pts[i - 1])
        distances.append(d)
    distances = np.cumsum(distances)
    total_length = distances[-1]

    # Generate evenly spaced distances and perform linear interpolation.
    sample_dists = np.linspace(0, total_length, num_points, endpoint=False)
    resampled_pts = []
    j = 0
    for d in sample_dists:
        while j < len(distances) - 1 and distances[j + 1] < d:
            j += 1
        if j == len(distances) - 1:
            resampled_pts.append(pts[j])
        else:
            t = (d - distances[j]) / (distances[j + 1] - distances[j])
            point = (1 - t) * pts[j] + t * pts[j + 1]
            resampled_pts.append(point)
    resampled_pts = np.array(resampled_pts)
    all_points_x = resampled_pts[:, 0].tolist()
    all_points_y = resampled_pts[:, 1].tolist()
    return all_points_x, all_points_y

def convert_polygon_to_normalized(all_points_x, all_points_y, img_width, img_height):
    """
    Converts absolute polygon coordinates to normalized coordinates (0-1 range).
    Returns two lists: norm_x and norm_y.
    """
    norm_x = [x / img_width for x in all_points_x]
    norm_y = [y / img_height for y in all_points_y]
    return norm_x, norm_y

def create_yolo_annotation_txt(filename, rendered_pil, annotation_pts, label="cube"):
    """
    Given an image filename, the rendered image (PIL), and the annotation polygon points,
    returns a string formatted in the Ultralytics YOLO segmentation format:
    <class-index> <x1_norm> <y1_norm> <x2_norm> <y2_norm> ... <xn_norm> <yn_norm>
    Assumes that the label maps to class index 0 (change as needed).
    """
    # Open image to get width and height.
    img_width, img_height = rendered_pil.size
    all_points_x, all_points_y = annotation_pts

    # Convert absolute coordinates to normalized ones.
    norm_x, norm_y = convert_polygon_to_normalized(all_points_x, all_points_y, img_width, img_height)
    # In YOLO segmentation format, each row starts with the class index (here 0 for "cube")
    # followed by the normalized polygon coordinates (x1 y1 x2 y2 ...).
    coords = " ".join(str(val) for pair in zip(norm_x, norm_y) for val in pair)
    # If you wish to support multiple classes, adjust the class index accordingly.
    return f"0 {coords}"

def draw_polygon(image, all_points_x, all_points_y, label="cube"):
    """Draws the polygon outline on a copy of the image."""
    draw = ImageDraw.Draw(image)
    pts = list(zip(all_points_x, all_points_y))
    if pts:
        draw.line(pts + [pts[0]], fill="red", width=3)
        draw.text(pts[0], label, fill="red")
    return image

#############################
# Main Generation Script
#############################

def main():
    parser = argparse.ArgumentParser(
        description="Generate a YOLO segmentation dataset from a 3D mesh rendered with PyTorch3D.\n"
                    "This script renders images, extracts object masks via GrabCut and resamples a polygon\n"
                    "to produce a YOLO-format (per-image text file) segmentation annotation."
    )
    parser.add_argument("input_obj", type=str,
                        help="Path to the input OBJ file (with its associated MTL/texture files).")
    parser.add_argument("output_folder", type=str,
                        help="Main folder for the dataset (e.g., 'data'). This folder will contain four subfolders:\n"
                             "  - training (images and corresponding YOLO .txt annotation files)\n"
                             "  - validation (images and corresponding YOLO .txt files)\n"
                             "  - testing (images and corresponding YOLO .txt files)\n"
                             "  - annotated (annotated images with drawn polygon outlines)")
    parser.add_argument("--num_train", type=int, default=400,
                        help="Number of training images (default: 400)")
    parser.add_argument("--num_val", type=int, default=50,
                        help="Number of validation images (default: 50)")
    parser.add_argument("--num_test", type=int, default=50,
                        help="Number of testing images (default: 50)")
    parser.add_argument("--base_image_size", type=int, default=512,
                        help="Base output image size in pixels (default: 512)")
    parser.add_argument("--faces_per_pixel", type=int, default=10,
                        help="Faces per pixel for rasterization (default: 10)")
    parser.add_argument("--margin_min", type=float, default=1.0,
                        help="Minimum margin multiplier (default: 1.0)")
    parser.add_argument("--margin_max", type=float, default=1.2,
                        help="Maximum margin multiplier (default: 1.2)")
    parser.add_argument("--radius_threshold", type=float, default=0.1,
                        help="If the object's radius is below this value, resolution is increased (default: 0.1)")
    parser.add_argument("--min_camera_dist", type=float, default=0.5,
                        help="Minimum allowed camera distance (default: 0.5)")
    parser.add_argument("--label", type=str, default="cube",
                        help="Object label (default: cube), which maps to class index 0 in the annotation")
    args = parser.parse_args()

    # Create folder structure:
    # data/
    #   training/       (original images + .txt annotation files)
    #   validation/     (original images + .txt annotation files)
    #   testing/        (original images + .txt annotation files)
    #   annotated/      (annotated images with drawn polygons)
    train_folder = os.path.join(args.output_folder, "training")
    val_folder = os.path.join(args.output_folder, "validation")
    test_folder = os.path.join(args.output_folder, "testing")
    annotated_folder = os.path.join(args.output_folder, "annotated")
    for folder in [train_folder, val_folder, test_folder, annotated_folder]:
        os.makedirs(folder, exist_ok=True)

    total_images = args.num_train + args.num_val + args.num_test

    # Set up device and load mesh.
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    mesh = load_objs_as_meshes([args.input_obj], device=device)
    center, radius = compute_bounding_sphere(mesh)
    print(f"Computed mesh center: {center}, radius: {radius}")

    # Adjust image resolution if object is small.
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

    # For splitting the images randomly into three groups.
    splits = (["train"] * args.num_train + ["validation"] * args.num_val + ["test"] * args.num_test)
    random.shuffle(splits)

    for i in range(total_images):
        # Randomize camera settings.
        fov = random.uniform(30, 70)
        half_fov_rad = math.radians(fov / 2)
        safe_dist = radius / math.tan(half_fov_rad)
        margin = random.uniform(args.margin_min, args.margin_max)
        dist = max(safe_dist * margin, args.min_camera_dist)
        elev = random.uniform(10, 60)
        azim = random.uniform(0, 360)
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=center.unsqueeze(0))
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

        # Randomize a light source.
        light_x = random.uniform(-3, 3)
        light_y = random.uniform(-3, 3)
        light_z = random.uniform(-3, -1)
        lights = PointLights(device=device, location=[[light_x, light_y, light_z]])

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
        )

        # Render the image.
        images = renderer(mesh)
        image_np = images[0, ..., :3].cpu().numpy()
        gamma = 2.2
        image_np = np.clip(image_np, 0, 1) ** (1 / gamma)
        image_aug = random_augment(image_np)

        filename = f"image_{i:04d}.png"
        split = splits[i]
        if split == "train":
            orig_path = os.path.join(train_folder, filename)
        elif split == "validation":
            orig_path = os.path.join(val_folder, filename)
        else:
            orig_path = os.path.join(test_folder, filename)

        image_aug.save(orig_path)
        print(f"Saved original image to {orig_path}")

        # Generate a 50-point segmentation polygon using GrabCut.
        bbox = compute_bbox(cameras, mesh, image_size)
        pts_x, pts_y = extract_polygon_from_grabcut(image_aug, bbox, num_points=50, iter_count=10)
        # If GrabCut fails, fall back to using the bounding box corners.
        if not pts_x or not pts_y:
            x = bbox["x"]
            y = bbox["y"]
            w = bbox["width"]
            h = bbox["height"]
            pts_x = [x, x+w, x+w, x]
            pts_y = [y, y, y+h, y+h]

        # Create YOLO segmentation annotation text (normalized).
        annotation_txt = create_yolo_annotation_txt(filename, image_aug, (pts_x, pts_y), label=args.label)

        # Save annotation text file in the same folder as the original image.
        annotation_filename = os.path.splitext(filename)[0] + ".txt"
        if split == "train":
            anno_path = os.path.join(train_folder, annotation_filename)
        elif split == "validation":
            anno_path = os.path.join(val_folder, annotation_filename)
        else:
            anno_path = os.path.join(test_folder, annotation_filename)
        with open(anno_path, "w") as f:
            f.write(annotation_txt)
        print(f"Saved YOLO annotation to {anno_path}")

        # Optionally, draw the polygon on a copy for visualization and save it to annotated folder.
        annotated_img = image_aug.copy()
        annotated_img = draw_polygon(annotated_img, pts_x, pts_y, label=args.label)
        anno_vis_path = os.path.join(annotated_folder, filename)
        annotated_img.save(anno_vis_path)
        print(f"Saved annotated image to {anno_vis_path}")

    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
