
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
    def __init__(self, image_size, n_betas, z_distance = 5.0, elevation = 89.9, azimuth = 0.0):
        super(SMAL3DRenderer, self).__init__()
        
        self.smal_model = SMALModel()
        self.image_size = image_size
        self.smal_info = SMALJointInfo()
        self.n_betas = n_betas

        self.renderer = nr.Renderer(camera_mode='look_at')
        self.renderer.eye = nr.get_points_from_angles(z_distance, elevation, azimuth)

        self.renderer.image_size = image_size
        self.renderer.light_intensity_ambient = 1.0
        self.renderer.light_direction = [0.0, 0.0, -1.0]
        self.renderer.viewing_angle = 10
        self.renderer.background_color = [1.0, 1.0, 1.0]

        with open("smal/dog_texture.pkl", 'rb') as f:
            self.textures = pkl.load(f).cuda()

        self.crop_boxes = torch.stack(
            torch.meshgrid(
                torch.linspace(0, 1, steps = self.image_size), 
                torch.linspace(0, 1, steps = self.image_size)), dim = -1)[None, :, :]

    def crop_to_silhouette(self, images, silhouettes, joints, pad = 1.2):
        batch_size = images.shape[0]
        crop_boxes = self.crop_boxes.expand(batch_size, self.image_size, self.image_size, 2).cuda()

        silhouettes = silhouettes[:, 0]

        min_h, max_h, min_w, max_w = [], [], [], []
        for batch_id in range(batch_size):
            nonzero_elements = torch.nonzero(silhouettes[batch_id])
            min_h.append(torch.min(nonzero_elements[:, 0]))
            max_h.append(torch.max(nonzero_elements[:, 0]))
            min_w.append(torch.min(nonzero_elements[:, 1]))
            max_w.append(torch.max(nonzero_elements[:, 1]))

        min_h = torch.stack(min_h, dim = -1).float()
        max_h = torch.stack(max_h, dim = -1).float()
        min_w = torch.stack(min_w, dim = -1).float()
        max_w = torch.stack(max_w, dim = -1).float()

        square_halfside = torch.max(max_h - min_h, max_w - min_w) / 2.0
        square_halfside *= pad

        square_size = torch.stack([max_h - min_h, max_w - min_w], dim = -1)
        top_left = torch.stack([min_h, min_w], dim = -1)

        square_centre = top_left + square_size / 2.0

        new_topleft = square_centre - square_halfside[:, None]
        new_bottomright = square_centre + square_halfside[:, None]

        new_squaresize = new_bottomright - new_topleft

        crop_boxes = crop_boxes * new_squaresize[:, None, None, :] + new_topleft[:, None, None, :]
        crop_boxes[:, :, :, 0] = (2.0 * crop_boxes[:, :, :, 0] / images.shape[2]) - 1.0
        crop_boxes[:, :, :, 1] = (2.0 * crop_boxes[:, :, :, 1] / images.shape[3]) - 1.0
        crop_boxes = torch.flip(crop_boxes, [3])
        
        cropped_images = F.grid_sample(images - 1.0, crop_boxes) + 1.0
        cropped_silhouettes = F.grid_sample(silhouettes[:, None, :, :], crop_boxes)[:, 0]

        scale_factor = torch.FloatTensor([images.shape[2], images.shape[3]]).cuda() / new_squaresize
        scaled_joints = (joints - new_topleft[:, None, :]) * scale_factor[:, None, :]

        return cropped_images, cropped_silhouettes, scaled_joints

    def forward(self, batch_params, return_visuals = False, reverse_view = False, crop_to_silhouette = False):
        batch_size = batch_params['joint_rotations'].shape[0]
        global_rotation = batch_params['global_rotation'].clone()

        if reverse_view:
            global_rotation += torch.FloatTensor([[0, 0, 3 * np.pi / 2]]).cuda()

        verts, joints_3d = self.smal_model(
            torch.cat([batch_params['betas'], torch.zeros(batch_size, 41 - self.n_betas).cuda()], dim = 1), # Pad remaining shape parameters with zeros
            torch.cat((global_rotation, batch_params['joint_rotations'].view(batch_size, -1)), dim = 1),
            batch_params['trans'], normalize = reverse_view)
    
        faces = self.smal_model.faces.unsqueeze(0).expand(batch_size, -1, -1)
        textures = self.textures.unsqueeze(0).expand(batch_size, -1, -1, -1, -1, -1)

        rendered_joints = self.renderer.render_points(joints_3d[:, self.smal_info.annotated_classes])
        valid_landmark = torch.ones_like(rendered_joints)[:, 0]
        
        rendered_silhouettes = self.renderer.render_silhouettes(verts, faces)
        rendered_silhouettes = rendered_silhouettes[:, None, :, :]

        if return_visuals:
            rendered_images = self.renderer.render(verts, faces, textures)[0]
            rendered_images = torch.clamp(rendered_images, 0.0, 1.0)

            if crop_to_silhouette:
                rendered_images, rendered_silhouettes, rendered_joints = self.crop_to_silhouette(rendered_images, rendered_silhouettes, rendered_joints)
        
            return rendered_images, rendered_silhouettes, rendered_joints, valid_landmark, verts, joints_3d
        else:
            return rendered_silhouettes, rendered_joints, valid_landmark