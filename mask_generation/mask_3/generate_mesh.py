import os
import torch
import matplotlib.pyplot as plt

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

# Use GPU if available, otherwise CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Load your cube OBJ file (cube.obj should have its accompanying MTL and texture files)
obj_filename = "cube.obj"
mesh = load_objs_as_meshes([obj_filename], device=device)

# Adjust the camera settings to fill the frame with the cube
R, T = look_at_view_transform(dist=0.7, elev=30, azim=45)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=30)

# Increase output resolution and faces per pixel for better detail and anti-aliasing
raster_settings = RasterizationSettings(
    image_size=2048,      # Higher resolution for finer details
    blur_radius=0.0,
    faces_per_pixel=10,   # Consider more faces per pixel for smoother rendering
)

# Define a point light positioned to highlight the cube
lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

# Use a Phong shader to get smooth lighting and shading effects
renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
)

# Render the image
images = renderer(mesh)
image = images[0, ..., :3].cpu().numpy()

# Save the rendered image as "cube.png"
plt.imsave("cube.png", image)
print("Rendering complete! The image has been saved as cube.png")
