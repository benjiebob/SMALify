# Data structures and functions for rendering
import torch
import torch.nn.functional as F
from scipy.io  import loadmat
import numpy as np
import config

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OpenGLPerspectiveCameras, look_at_view_transform, look_at_rotation, 
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    PointLights, SoftPhongShader, SoftSilhouetteShader, Materials, Textures
)
from pytorch3d.io import load_objs_as_meshes
from utils import perspective_proj_withz

class Renderer(torch.nn.Module):
    def __init__(self, image_size, device):
        super(Renderer, self).__init__()

        self.image_size = image_size

        raster_settings = RasterizationSettings(
            image_size=self.image_size, 
            blur_radius=0.0, 
            faces_per_pixel=1,
        )

        R, T = look_at_view_transform(2.7, 0, 0) 
        self.cameras = OpenGLPerspectiveCameras(device=device, R=R, T=T)
        lights = PointLights(device=device, location=[[0.0, 1.0, 0.0]])

        self.mesh_color = torch.FloatTensor(config.MESH_COLOR, device=device)[None, None, :] / 255.0

        self.color_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftPhongShader(
                device=R.device, 
                cameras=self.cameras,
                lights=lights,
            )
        )

        self.silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras, 
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader()
        )

    def forward(self, vertices, points, faces, render_texture=False):
        tex = torch.ones_like(vertices) * self.mesh_color # (1, V, 3)
        textures = Textures(verts_rgb=tex)

        mesh = Meshes(verts=vertices, faces=faces, textures=textures)
        sil_images = self.silhouette_renderer(mesh).permute(0, 3, 1, 2)[:, [0], :, :]
        screen_size = torch.ones(vertices.shape[0], 2).to(vertices.device) * self.image_size
        proj_points = self.cameras.transform_points_screen(points, screen_size)[:, :, [1, 0]]

        if render_texture:
            color_image = self.color_renderer(mesh).permute(0, 3, 1, 2)[:, :3, :, :]
            return sil_images, proj_points, color_image
        else:
            return sil_images, proj_points