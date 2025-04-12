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
    # Revert to original augmentation range
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

def extract_polygon_from_grabcut(pil_image, bbox, num_points=50, iter_count=10):
    """
    Use GrabCut to extract the object mask from pil_image, using bbox as the initial rectangle.
    Then extract the largest contour and resample it to get exactly num_points.
    Returns two lists: all_points_x and all_points_y.
    """
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask = np.zeros(image.shape[:2], np.uint8)
    rect = (int(bbox["x"]), int(bbox["y"]), int(bbox["width"]), int(bbox["height"]))
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==cv2.GC_FGD) | (mask==cv2.GC_PR_FGD), 1, 0).astype('uint8')
    contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return [], []
    largest_contour = max(contours, key=cv2.contourArea)
    pts = largest_contour.reshape(-1, 2).astype(np.float32)
    distances = [0]
    for i in range(1, len(pts)):
        d = np.linalg.norm(pts[i] - pts[i-1])
        distances.append(d)
    distances = np.cumsum(distances)
    total_length = distances[-1]
    sample_dists = np.linspace(0, total_length, num_points, endpoint=False)
    resampled_pts = []
    j = 0
    for d in sample_dists:
        while j < len(distances)-1 and distances[j+1] < d:
            j += 1
        if j == len(distances)-1:
            resampled_pts.append(pts[j])
        else:
            t = (d - distances[j]) / (distances[j+1] - distances[j])
            point = (1-t)*pts[j] + t*pts[j+1]
            resampled_pts.append(point)
    resampled_pts = np.array(resampled_pts)
    all_points_x = resampled_pts[:, 0].tolist()
    all_points_y = resampled_pts[:, 1].tolist()
    return all_points_x, all_points_y

def annotate_image(cameras, mesh, image_size, filename, rendered_pil, label="object"):
    """
    Create a VIA-style annotation using the polygon extracted via GrabCut.
    """
    bbox = compute_bbox(cameras, mesh, image_size)
    all_points_x, all_points_y = extract_polygon_from_grabcut(rendered_pil, bbox, num_points=50, iter_count=10)
    if not all_points_x or not all_points_y:
        x = bbox["x"]
        y = bbox["y"]
        w = bbox["width"]
        h = bbox["height"]
        all_points_x = [x, x+w, x+w, x]
        all_points_y = [y, y, y+h, y+h]
    annotation = {
        "filename": filename,
        "size": 0,
        "regions": {
            "0": {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": all_points_x,
                    "all_points_y": all_points_y
                },
                "region_attributes": {
                    "name": label
                }
            }
        },
        "file_attributes": {}
    }
    return annotation, bbox

def draw_polygon(image, all_points_x, all_points_y, label="object"):
    draw = ImageDraw.Draw(image)
    pts = list(zip(all_points_x, all_points_y))
    draw.line(pts + [pts[0]], fill="red", width=3)
    draw.text(pts[0], label, fill="red")
    return image

def main():
    parser = argparse.ArgumentParser(
        description="Generate a training set with original and annotated images, and JSON annotations in VIA format using GrabCut."
    )
    parser.add_argument("input_obj", type=str,
                        help="Path to the input OBJ file (with associated MTL/texture files).")
    parser.add_argument("output_folder", type=str,
                        help="Main folder for the training set. This folder (e.g. 'data') will contain four subfolders: training, validation, testing, and annotated.")
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
                        help="Label for annotation (default: cube)")
    args = parser.parse_args()

    # Create main folder structure under "data"
    train_folder = os.path.join(args.output_folder, "training")
    val_folder = os.path.join(args.output_folder, "validation")
    test_folder = os.path.join(args.output_folder, "testing")
    annotated_folder = os.path.join(args.output_folder, "annotated")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    os.makedirs(annotated_folder, exist_ok=True)

    total_images = args.num_train + args.num_val + args.num_test

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    mesh = load_objs_as_meshes([args.input_obj], device=device)
    center, radius = compute_bounding_sphere(mesh)
    print(f"Computed mesh center: {center}, radius: {radius}")

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

    annotations_train = {}
    annotations_val = {}
    annotations_test = {}

    # Randomly assign each generated image to a split.
    splits = (["train"] * args.num_train + ["val"] * args.num_val + ["test"] * args.num_test)
    random.shuffle(splits)

    for i in range(total_images):
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
        split = splits[i]
        if split == "train":
            orig_path = os.path.join(train_folder, filename)
        elif split == "val":
            orig_path = os.path.join(val_folder, filename)
        else:
            orig_path = os.path.join(test_folder, filename)
        image_aug.save(orig_path)
        print(f"Saved original image to {orig_path}")

        # Generate annotation with a 50-point polygon.
        annotation, bbox = annotate_image(cameras, mesh, image_size, filename, image_aug, label=args.label)
        if split == "train":
            annotations_train[filename] = annotation
        elif split == "val":
            annotations_val[filename] = annotation
        else:
            annotations_test[filename] = annotation

        pts_x, pts_y = extract_polygon_from_grabcut(image_aug, bbox, num_points=50, iter_count=10)
        annotated_img = image_aug.copy()
        annotated_img = draw_polygon(annotated_img, pts_x, pts_y, label=args.label)
        anno_path = os.path.join(annotated_folder, filename)
        annotated_img.save(anno_path)
        print(f"Saved annotated image to {anno_path}")

    train_json_path = os.path.join(train_folder, "annotations.json")
    with open(train_json_path, "w") as f:
        json.dump(annotations_train, f, indent=4)
    print(f"Saved training annotations JSON to {train_json_path}")

    val_json_path = os.path.join(val_folder, "annotations.json")
    with open(val_json_path, "w") as f:
        json.dump(annotations_val, f, indent=4)
    print(f"Saved validation annotations JSON to {val_json_path}")

    test_json_path = os.path.join(test_folder, "annotations.json")
    with open(test_json_path, "w") as f:
        json.dump(annotations_test, f, indent=4)
    print(f"Saved testing annotations JSON to {test_json_path}")

    print("Training set generation complete.")

if __name__ == "__main__":
    main()
