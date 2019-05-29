import torch
import torch.nn as nn
import neural_renderer as nr
import torch.nn.functional as F
from smal.smpl.smal_torch_batch import SMALModel
from smal.joint_catalog import SMALJointInfo

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

class SMAL3DRenderer(nn.Module):
    def __init__(self, image_size, z_distance = 2.0, elevation = 89.9, azimuth = 0.0):
        super(SMAL3DRenderer, self).__init__()
        
        self.smal_model = SMALModel()
        self.image_size = image_size
        self.smal_info = SMALJointInfo()

        self.renderer = nr.Renderer(camera_mode='look_at')
        self.renderer.eye = nr.get_points_from_angles(z_distance, elevation, azimuth)

        self.renderer.image_size = image_size
        self.renderer.light_intensity_ambient = 1.0
        # self.renderer.perspective = False

        with open("../smal/dog_texture.pkl", 'rb') as f:
            self.textures = pkl.load(f).cuda()

    def forward(self, batch_params, return_visuals = False):
        verts, joints_3d = self.smal_model(
            batch_params['betas'], 
            torch.cat((batch_params['global_rotation'], batch_params['joint_rotations']), dim = 1),
            batch_params['trans'])

        batch_size = batch_params['betas'].shape[0]
        faces = self.smal_model.faces[None, :, :].expand(batch_size, -1, -1)
        textures = self.textures.unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)

        rendered_joints = self.renderer.render_points(joints_3d[:, self.smal_info.annotated_classes])
        valid_landmark = torch.ones_like(rendered_joints)[:, 0]
        
        rendered_silhouettes = self.renderer.render_silhouettes(verts, faces)
        rendered_silhouettes = rendered_silhouettes[:, None, :, :]

        if return_visuals:
            rendered_images = self.renderer.render(verts, faces, textures)[0]
            rendered_images = torch.clamp(rendered_images, 0.0, 1.0)
        
            return rendered_images, rendered_silhouettes, rendered_joints, valid_landmark, verts, joints_3d
        else:
            return rendered_silhouettes, rendered_joints, valid_landmark