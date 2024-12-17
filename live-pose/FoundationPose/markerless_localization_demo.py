import pyrealsense2 as rs
from estimater import *
from FoundationPose.mask import *
import tkinter as tk
from tkinter import filedialog
from multiprocessing import Pool

parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--est_refine_iter', type=int, default=4)
parser.add_argument('--track_refine_iter', type=int, default=2)
args = parser.parse_args()

set_logging_format()
set_seed(0)

root = tk.Tk()
root.withdraw()

# RETRIEVE A LIST OF OBJ FILES
mesh_directory = filedialog.askdirectory()
if not mesh_directory:
    print("No mesh directory selected")
    exit(0)
    
mesh_paths = glob.glob(mesh_directory + "/*.obj")


# INITIALIZE MESH DATA
mask_file_path_list = [create_mask(os.path.splitext(os.path.split(mesh_path)[1])[0]) for mesh_path in mesh_paths]
mesh_list = [trimesh.load(mesh_path) for mesh_path in mesh_paths]
to_origin_list, extents_list = [], []
for mesh in mesh_list:
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    to_origin_list.append(to_origin)
    extents_list.append(extents)
bbox_list = [np.stack([-extents/2, extents/2], axis=0).reshape(2,3) for extents in extents_list]

# INITIALIZE ESTIMATORS
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est_list = [FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner,glctx=glctx) for mesh in mesh_list]

# START REALSENSE CAMERA
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipeline.start(config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale
align_to = rs.stream.color
align = rs.align(align_to)

i = 0

mask_list = [cv2.imread(mask_file_path, cv2.IMREAD_UNCHANGED) for mask_file_path in mask_file_path_list]
cam_K = np.array([[615.37701416, 0., 313.68743896],
                   [0., 615.37701416, 259.01800537],
                   [0., 0., 1.]])
                   
# DEFINE ESTIMATING FUNCTION
def estimate_pose(mask, est, to_origin):
    if i==0:
        if len(mask.shape)==3:
            for c in range(3):
                if mask[...,c].sum()>0:
                    mask = mask[...,c]
                    break
        mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
        
        pose = est.register(K=cam_K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
    else:
        pose = est.track_one(rgb=color, depth=depth, K=cam_K, iteration=args.track_refine_iter)
    center_pose = pose@np.linalg.inv(to_origin)
            
    return center_pose
    
# INITIALIZE OBJ DATA 
NUM_OBJS = len(mesh_list)

obj_info = [(mask_list[obj_index], est_list[obj_index], to_origin_list[obj_index]) for obj_index in range(NUM_OBJS)]
                   
# MAIN

Estimating = True
time.sleep(3)
# Streaming loop
try:
    while Estimating:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(aligned_depth_frame.get_data())/1e3
        color_image = np.asanyarray(color_frame.get_data())
        depth_image_scaled = (depth_image * depth_scale * 1000).astype(np.float32)
        if cv2.waitKey(1) == 13:
            Estimating = False
            break        
        H, W = color_image.shape[:2]
        color = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth_image_scaled, (W,H), interpolation=cv2.INTER_NEAREST)
        depth[(depth<0.1) | (depth>=np.inf)] = 0
        
        # ESTIMATE POSES
        center_pose_list = []
        for obj_index in range(NUM_OBJS):
            center_pose_list.append(estimate_pose(obj_info[obj_index][0], obj_info[obj_index][1], obj_info[obj_index][2]))
        
        for obj_index in range(NUM_OBJS):
            vis = draw_posed_3d_box(cam_K, img=color, ob_in_cam=center_pose_list[obj_index], bbox=bbox_list[obj_index])
            vis = draw_xyz_axis(color, ob_in_cam=center_pose_list[obj_index], scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
            
        cv2.imshow('1', vis[...,::-1])
        cv2.waitKey(1)        
        i += 1
        
finally:
    pipeline.stop()