#!/usr/bin/env python
import os
import argparse
import random
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

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
    Apply random brightness and contrast adjustments to the input image.
    Expects image_np values in [0,1].
    """
    image = Image.fromarray((np.clip(image_np, 0, 1) * 255).astype(np.uint8))

    # Random brightness adjustment (factor between 0.8 and 1.2)
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random contrast adjustment (factor between 0.8 and 1.2)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(random.uniform(0.8, 1.2))

    return image

def main():
    parser = argparse.ArgumentParser(
        description="Generate a training set by rendering a 3D mesh with random camera and lighting settings, auto-adjusting resolution, and ensuring the object remains in frame."
    )
    parser.add_argument("input_obj", type=str,
                        help="Path to the input OBJ file (with associated MTL/texture files).")
    parser.add_argument("output_folder", type=str,
                        help="Folder where the training images will be saved.")
    parser.add_argument("--num_images", type=int, default=400,
                        help="Number of images to generate (default: 400)")
    parser.add_argument("--base_image_size", type=int, default=2048,
                        help="Base output image size (square) in pixels (default: 2048)")
    parser.add_argument("--faces_per_pixel", type=int, default=10,
                        help="Faces per pixel for rasterization (default: 10)")
    parser.add_argument("--margin_min", type=float, default=1.2,
                        help="Minimum margin multiplier (default: 1.2)")
    parser.add_argument("--margin_max", type=float, default=1.4,
                        help="Maximum margin multiplier (default: 1.4)")
    parser.add_argument("--radius_threshold", type=float, default=0.1,
                        help="If the object's radius is below this value, the output resolution is increased (default: 0.1)")
    parser.add_argument("--min_camera_dist", type=float, default=1.0,
                        help="Minimum allowed camera distance (default: 1.0)")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)

    # Use GPU if available, otherwise CPU.
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    # Load the mesh and compute its bounding sphere.
    mesh = load_objs_as_meshes([args.input_obj], device=device)
    center, radius = compute_bounding_sphere(mesh)
    print(f"Computed mesh center: {center}, radius: {radius}")

    # Adjust the image resolution for small objects.
    if radius < args.radius_threshold:
        scale_factor = args.radius_threshold / radius
        image_size = int(args.base_image_size * scale_factor)
        print(f"Object is small (radius {radius:.4f}). Increasing image resolution to {image_size}x{image_size}.")
    else:
        image_size = args.base_image_size

    # Set up constant rasterization settings.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=args.faces_per_pixel,
    )

    # Generate images.
    for i in range(args.num_images):
        # Randomly choose a field-of-view between 30 and 70 degrees.
        fov = random.uniform(30, 70)
        half_fov_rad = math.radians(fov / 2)
        # Compute safe distance: safe_dist = radius / tan(fov/2)
        safe_dist = radius / math.tan(half_fov_rad)
        # Apply a random margin.
        margin = random.uniform(args.margin_min, args.margin_max)
        # Ensure the camera distance is at least the minimum.
        dist = max(safe_dist * margin, args.min_camera_dist)

        # Randomize other camera parameters.
        elev = random.uniform(10, 60)   # elevation in degrees
        azim = random.uniform(0, 360)     # azimuth in degrees

        # Set up camera so it always looks at the computed center.
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=center.unsqueeze(0))
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)

        # Randomize light position.
        light_x = random.uniform(-3, 3)
        light_y = random.uniform(-3, 3)
        light_z = random.uniform(-3, -1)
        lights = PointLights(device=device, location=[[light_x, light_y, light_z]])

        # Create renderer.
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
        )

        # Render image.
        images = renderer(mesh)
        image_np = images[0, ..., :3].cpu().numpy()
        # Apply gamma correction.
        gamma = 2.2
        image_np = np.clip(image_np, 0, 1) ** (1 / gamma)

        # Apply random augmentations.
        image_aug = random_augment(image_np)

        # Save the image.
        out_path = os.path.join(args.output_folder, f"image_{i:04d}.png")
        image_aug.save(out_path)
        print(f"Saved {out_path}")

    print("Training set generation complete.")

if __name__ == "__main__":
    main()
