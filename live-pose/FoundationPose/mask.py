## testing

import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import time
import os


def project_points(mesh, intrinsic, extrinsic):
    vertices = np.asarray(mesh.vertices)
    points_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))

    camera_matrix = intrinsic @ extrinsic[:3, :]
    projected = (camera_matrix @ points_homogeneous.T).T

    # Convert from homogeneous to 2D coordinates
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    return projected[:, :2].astype(int)

def generate_mask_from_mesh(obj_file, name, image, intrinsic_matrix, extrinsic_matrix):
    mask_path = f'./masks/mask_{name}.png'

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh.compute_vertex_normals()

    height, width = image.shape[:2]
    projected_points = project_points(mesh, intrinsic_matrix, extrinsic_matrix)

    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [projected_points], 255)

    cv2.imwrite(mask_path, mask)
    return mask_path

def process_meshes_folder(folder_path):
    # Configure RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    try:
        time.sleep(1)  # Allow camera to warm up
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise Exception("Could not capture color frame")

        image = np.asanyarray(color_frame.get_data())

        # Get camera intrinsics
        profile = pipeline.get_active_profile()
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsic_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])

        # Assume identity extrinsic (camera at world origin)
        extrinsic_matrix = np.eye(4)

        # Process all .obj files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(".obj"):
                obj_file = os.path.join(folder_path, filename)
                mask_file_path = generate_mask_from_mesh(obj_file, filename[:-4], image, intrinsic_matrix, extrinsic_matrix)
                print(f"Mask saved at: {mask_file_path}")

    finally:
        pipeline.stop()

if __name__ == "__main__":
    meshes_folder = "./meshes"  # Path to folder containing .obj files
    process_meshes_folder(meshes_folder)
