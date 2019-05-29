
import sys
sys.path.append('../')

from smal.smal3d_renderer import SMAL3DRenderer
from smal.draw_smal_joints import SMALJointDrawer

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle as pkl
import os
import scipy.misc

import torch
import torch.nn.functional as F
import torch.nn as nn

from functools import reduce

from joint_limits_prior import LimitPrior
from torchvision.utils import make_grid
import trimesh

SMAL_DATA_PATH = '../smal/smal_CVPR2017_data.pkl'

class SMALFitter(nn.Module):
    def __init__(self, data_batch, filename_batch, batch_size, shape_family, output_dir, n_betas = 20):
        super(SMALFitter, self).__init__()

        self.rgb_imgs, self.sil_imgs, self.target_joints, self.target_visibility = data_batch
        self.target_visibility = self.target_visibility.long()

        assert self.rgb_imgs.max() <= 1.0 and self.rgb_imgs.min() >= 0.0, "RGB Image range is incorrect"

        self.output_dirs = [] 
        for filename in filename_batch:
            filename_path = os.path.join(output_dir, os.path.splitext(filename)[0])
            self.output_dirs.append(filename_path)
            os.mkdir(filename_path)

        self.num_images = self.rgb_imgs.shape[0]
        self.batch_size = batch_size

        self.shape_family_list = np.array(shape_family)
        with open(SMAL_DATA_PATH, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            smal_data = u.load()

        model_covs = np.array(smal_data['cluster_cov'])[shape_family][0]

        invcov = np.linalg.inv(model_covs + 1e-5 * np.eye(model_covs.shape[0]))
        prec = np.linalg.cholesky(invcov)

        self.betas_prec = torch.FloatTensor(prec).cuda()
        self.mean_betas = torch.FloatTensor(smal_data['cluster_means'][shape_family][0]).cuda()

        self.smal_joint_drawer = SMALJointDrawer()

        limit_prior = LimitPrior()
        self.max_limits = torch.FloatTensor(limit_prior.max_values).cuda()
        self.min_limits = torch.FloatTensor(limit_prior.min_values).cuda()
        
        global_rotation = torch.FloatTensor([0, 0, np.pi / 2])[None, :].cuda().repeat(self.num_images, 1) # Global Init
        self.global_rotation = nn.Parameter(global_rotation)

        self.global_mask = torch.ones_like(global_rotation)
        # self.global_mask[:, :2] = 0.0

        trans = torch.FloatTensor([0.0, 0.0, 0.0])[None, :].cuda().repeat(self.num_images, 1) # Trans Init
        self.trans = nn.Parameter(trans)

        default_joints = torch.zeros(self.num_images, 32, 3).cuda()
        self.joint_rotations = nn.Parameter(default_joints)

        self.rotation_mask = torch.ones_like(default_joints)
        self.rotation_mask[:, 25:32] = 0.0
        self.rotation_mask[:, :, [0, 2]] = 0.0
      
        self.betas = nn.Parameter(self.mean_betas[:n_betas][None, :].clone())

        # setup renderers
        self.model_renderer = SMAL3DRenderer(256).cuda()
        self.alt_view_renderer = SMAL3DRenderer(256, elevation=89.9).cuda()

    def print_grads(self, grad_output):
        print (grad_output)

    def forward(self, batch_range, weights):
        w_j2d, w_reproj, w_betas, w_limit = weights

        betas = torch.cat([self.betas, torch.zeros(1, 21).cuda()], dim = 1).expand(self.num_images, 41)      
        joint_rotations = (self.joint_rotations * self.rotation_mask).view(self.num_images, -1)
        global_rotation = self.global_rotation * self.global_mask

        batch_params = {
            'global_rotation' : global_rotation[batch_range],
            'trans' : self.trans[batch_range],
            'betas' : betas[batch_range],
            'joint_rotations' : joint_rotations[batch_range],
        }

        target_joints = self.target_joints[batch_range].cuda()
        target_visibility = self.target_visibility[batch_range].cuda()
        sil_imgs = self.sil_imgs[batch_range].cuda()
    
        objs = {}
        rendered_silhouettes, rendered_joints, _ = self.model_renderer(batch_params)

        if w_j2d > 0:
            rendered_joints[~target_visibility.byte()] = -1.0
            target_joints[~target_visibility.byte()] = -1.0

            objs['joint'] = w_j2d * F.l1_loss(rendered_joints, target_joints)

        if w_limit > 0:
            zeros = torch.zeros_like(batch_params['joint_rotations'])
            objs['limit'] = w_limit * torch.mean(
                torch.max(batch_params['joint_rotations'] - self.max_limits, zeros) + \
                torch.max(self.min_limits - batch_params['joint_rotations'], zeros))
        
        if w_betas > 0:
            beta_error = torch.tensordot(batch_params['betas'][:, :20] - self.mean_betas[:20], self.betas_prec[:20, :20], dims = 1)
            objs['betas'] = w_betas * torch.mean(beta_error ** 2)
    
        if w_reproj > 0:
            objs['sil_reproj'] = w_reproj * F.l1_loss(rendered_silhouettes, sil_imgs)

        return reduce(lambda x, y: x + y, objs.values()), objs

    def get_temporal(self, w_temp):
        joint_rotations = (self.joint_rotations * self.rotation_mask).view(self.num_images, -1)
        global_rotation = self.global_rotation * self.global_mask

        joint_loss = 0
        global_loss = 0
        trans_loss = 0

        for i in range(0, self.num_images - 1):
            global_loss += F.mse_loss(global_rotation[i], global_rotation[i + 1]) * w_temp
            joint_loss += F.mse_loss(joint_rotations[i], joint_rotations[i + 1]) * w_temp
            trans_loss += F.mse_loss(self.trans[i], self.trans[i + 1]) * w_temp

        return joint_loss, global_loss, trans_loss

    def generate_visualization(self, stage, epoch):
        betas = torch.cat([self.betas, torch.zeros(1, 21).cuda()], dim = 1).expand(self.num_images, 41)
        joint_rotations = (self.joint_rotations * self.rotation_mask).view(self.num_images, -1)
        joint_rotations = self.joint_rotations.view(self.num_images, -1)

        global_rotation = self.global_rotation * self.global_mask

        for j in range(0, self.num_images, self.batch_size):
            batch_range = list(range(j, min(self.num_images, j + self.batch_size)))
            batch_params = {
                'global_rotation' : global_rotation[batch_range],
                'trans' : self.trans[batch_range],
                'betas' : betas[batch_range],
                'joint_rotations' : joint_rotations[batch_range],
            }

            target_joints = self.target_joints[batch_range]
            target_visibility = self.target_visibility[batch_range]
            rgb_imgs = self.rgb_imgs[batch_range].cuda()
            sil_imgs = self.sil_imgs[batch_range].cuda()

            with torch.no_grad():
                rendered_images, rendered_silhouettes, rendered_joints, _, verts, _ = self.model_renderer(batch_params, return_visuals = True)
                rev_images, rev_silhouettes, rev_joints, rev_valid, _, _ = self.alt_view_renderer(batch_params, return_visuals = True)

                target_vis = self.smal_joint_drawer.draw_joints(rgb_imgs, target_joints, visible = target_visibility, normalized=False)
                rendered_images_vis = self.smal_joint_drawer.draw_joints(rendered_images, rendered_joints, visible = target_visibility, normalized=False)
                rev_images_vis = self.smal_joint_drawer.draw_joints(rev_images, rev_joints, visible = target_visibility, normalized=False)

                silhouette_error = F.l1_loss(sil_imgs, rendered_silhouettes, reduction='none')
                silhouette_error = silhouette_error.expand_as(rgb_imgs).data.cpu()

                collage_rows = torch.cat([target_vis, rendered_images_vis, silhouette_error, rev_images_vis], dim = 3)

                # collage = make_grid(target_vis, nrow = 2, padding = 10, pad_value=1.0)

                # plt.suptitle("Stage: {0}, Ep: {1}".format(stage, epoch))
                # plt.imshow(np.transpose(collage.numpy(), (1, 2, 0)))
                # plt.draw()
                # plt.pause(0.01)

                for i, im_id in enumerate(batch_range):
                    collage_np = np.transpose(collage_rows[i].numpy(), (1, 2, 0))
                    scipy.misc.imsave(os.path.join(self.output_dirs[im_id], "st{0}_ep{1}.png".format(stage, epoch)), collage_np)

                    # Export parameters
                    img_parameters = { k: v[i].cpu().data.numpy() for (k, v) in batch_params.items() }
                    with open(os.path.join(self.output_dirs[im_id], "st{0}_ep{1}.pkl".format(stage, epoch)), 'wb') as f:
                        pkl.dump(img_parameters, f)
                
                    # Export mesh
                    vertices = verts[i].cpu().numpy()
                    vertices[:, 0] *= -1
                    mesh = trimesh.Trimesh(vertices = vertices, faces = self.model_renderer.smal_model.faces.cpu().data.numpy(), process = False)
                    mesh.export(os.path.join(self.output_dirs[im_id], "st{0}_ep{1}.ply".format(stage, epoch)))
            
            


