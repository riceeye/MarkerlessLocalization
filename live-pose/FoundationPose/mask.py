## testing

import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
import time

def project_points(mesh, intrinsic, extrinsic):
    vertices = np.asarray(mesh.vertices)
    points_homogeneous = np.hstack((vertices, np.ones((vertices.shape[0], 1))))

    camera_matrix = intrinsic @ extrinsic[:3, :]
    projected = (camera_matrix @ points_homogeneous.T).T

    # Convert from homogeneous to 2D coordinates
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    return projected[:, :2].astype(int)

def generate_mask_from_mesh(obj_file, name):
    mask_path = f'./masks/mask_{name}.png'

    # Load mesh
    mesh = o3d.io.read_triangle_mesh(obj_file)
    mesh.compute_vertex_normals()

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
        height, width = image.shape[:2]

 
        profile = pipeline.get_active_profile()
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsic_matrix = np.array([
            [intrinsics.fx, 0, intrinsics.ppx],
            [0, intrinsics.fy, intrinsics.ppy],
            [0, 0, 1]
        ])

        extrinsic_matrix = np.eye(4)

        projected_points = project_points(mesh, intrinsic_matrix, extrinsic_matrix)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [projected_points], 255)

        cv2.imwrite(mask_path, mask)
        return mask_path

    finally:
        pipeline.stop()

if __name__ == "__main__":
    obj_file = "path/to/your/mesh.obj"  # Replace with actual path
    mask_file_path = generate_mask_from_mesh(obj_file, "test")
    print(f"Mask saved at: {mask_file_path}")
