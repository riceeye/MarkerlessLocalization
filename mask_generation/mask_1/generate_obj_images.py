import argparse
import os
import math
import random
import numpy as np

import trimesh
import pyrender
from PIL import Image, ImageOps, ImageEnhance


def ensure_folder_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def random_camera_pose(distance_range=(0.5, 2.0)):
    """
    Creates a random camera pose matrix for rendering.
    Rotates around X/Y/Z, then places the camera at a random distance along +Z.
    """
    # Random angles
    angle_x = random.uniform(0, 2 * math.pi)
    angle_y = random.uniform(0, 2 * math.pi)
    angle_z = random.uniform(0, 2 * math.pi)

    # Create rotation matrices
    rx = trimesh.transformations.rotation_matrix(angle_x, [1, 0, 0])
    ry = trimesh.transformations.rotation_matrix(angle_y, [0, 1, 0])
    rz = trimesh.transformations.rotation_matrix(angle_z, [0, 0, 1])

    # Combine rotations
    rotation = rx.dot(ry).dot(rz)

    # Random distance
    distance = random.uniform(*distance_range)

    # Place camera along +Z
    camera_pose = rotation.copy()
    camera_pose[:3, 3] = [0, 0, distance]

    return camera_pose


def render_random_view(mesh, image_size=(512, 512), distance_range=(0.5, 2.0)):
    """
    Renders the mesh from a random viewpoint:
      - camera pose is randomized
      - a directional light is placed at the camera location
    Returns a PIL Image.
    """
    scene = pyrender.Scene()

    # Convert trimesh to pyrender mesh
    mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    scene.add(mesh_node)

    # Random camera
    camera_pose = random_camera_pose(distance_range)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
    scene.add(camera, pose=camera_pose)

    # Add a directional light at camera location
    light = pyrender.DirectionalLight(intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render offscreen
    r = pyrender.OffscreenRenderer(viewport_width=image_size[0],
                                   viewport_height=image_size[1])
    color, _ = r.render(scene)
    r.delete()

    return Image.fromarray(color)


def random_2d_augment(img):
    """
    Applies random 2D augmentations with Pillow:
      - flips
      - in-plane rotations
      - brightness
      - color adjustments
    """
    # Random horizontal flip
    if random.random() > 0.5:
        img = ImageOps.mirror(img)

    # Random vertical flip
    if random.random() > 0.5:
        img = ImageOps.flip(img)

    # Random in-plane rotation (0-359 degrees)
    angle = random.randint(0, 359)
    img = img.rotate(angle)

    # Random brightness
    if random.random() > 0.5:
        enhancer = ImageEnhance.Brightness(img)
        factor = 0.5 + random.random()  # [0.5, 1.5]
        img = enhancer.enhance(factor)

    # Random color adjustment
    if random.random() > 0.5:
        enhancer = ImageEnhance.Color(img)
        factor = 0.5 + random.random()  # [0.5, 1.5]
        img = enhancer.enhance(factor)

    return img


def main():
    parser = argparse.ArgumentParser(description="Generate random PNG images from a 3D .obj file.")
    parser.add_argument("obj_file", type=str, help="Path to the .obj file.")
    parser.add_argument("--output_folder", type=str, default="output_pngs",
                        help="Folder to save generated images.")
    parser.add_argument("--num_images", type=int, default=10,
                        help="Number of images to generate.")
    parser.add_argument("--image_size", type=int, default=512,
                        help="Width/height (square) of the output images.")
    parser.add_argument("--no_2d_augment", action="store_true",
                        help="If set, disable the 2D augmentations.")

    args = parser.parse_args()

    # Ensure output folder exists
    ensure_folder_exists(args.output_folder)

    # Load the mesh
    mesh = trimesh.load(args.obj_file)

    # Generate images
    for i in range(args.num_images):
        # (1) Render from a random 3D viewpoint
        rendered_img = render_random_view(mesh,
                                          image_size=(args.image_size, args.image_size),
                                          distance_range=(0.5, 2.0))

        # (2) Optionally apply 2D augmentations
        if not args.no_2d_augment:
            final_img = random_2d_augment(rendered_img)
        else:
            final_img = rendered_img

        # (3) Save
        out_path = os.path.join(args.output_folder, f"image_{i}.png")
        final_img.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
