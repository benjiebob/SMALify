
from draw_smal_joints import SMALJointDrawer

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

from p3d_renderer import Renderer
from smal_model.smal_torch import SMAL
from priors.pose_prior_35 import Prior
from priors.joint_limits_prior import LimitPrior
import config

class SMALFitter(nn.Module):
    def __init__(self, device, data_batch, batch_size, shape_family, n_betas = 20):
        super(SMALFitter, self).__init__()

        self.rgb_imgs, self.sil_imgs, self.target_joints, self.target_visibility = data_batch
        self.target_visibility = self.target_visibility.long()

        assert self.rgb_imgs.max() <= 1.0 and self.rgb_imgs.min() >= 0.0, "RGB Image range is incorrect"

        self.device = device
        self.num_images = self.rgb_imgs.shape[0]
        self.image_size = self.rgb_imgs.shape[2]

        self.batch_size = batch_size
        self.n_betas = n_betas

        self.shape_family_list = np.array(shape_family)
        with open(config.SMAL_DATA_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            smal_data = u.load()

        model_covs = np.array(smal_data['cluster_cov'])[shape_family][0]

        invcov = np.linalg.inv(model_covs + 1e-5 * np.eye(model_covs.shape[0]))
        prec = np.linalg.cholesky(invcov)

        self.betas_prec = torch.FloatTensor(prec)[:n_betas, :n_betas].to(device)
        self.mean_betas = torch.FloatTensor(smal_data['cluster_means'][shape_family][0])[:n_betas].to(device)
        self.betas = nn.Parameter(self.mean_betas.clone().unsqueeze(0)) # Shape parameters (1 for the entire sequence... note expand rather than repeat)
        
        # self.betas.register_hook(self.print_grads)

        self.smal_joint_drawer = SMALJointDrawer()

        limit_prior = LimitPrior()
        self.pose_prior = Prior(config.WALKING_PRIOR_FILE, device)

        # self.max_limits = torch.FloatTensor(limit_prior.max_values).view(config.N_POSE, 3).to(device)
        # self.min_limits = torch.FloatTensor(limit_prior.min_values).view(config.N_POSE, 3).to(device)
        
        global_rotation = torch.FloatTensor([0, 0, 3 * np.pi / 2])[None, :].to(device).repeat(self.num_images, 1) # Global Init (Head-On)
        self.global_rotation = nn.Parameter(global_rotation)

        # Use this to restrict global rotation if necessary
        self.global_mask = torch.ones(1, 3).to(device)
        # self.global_mask[:2] = 0.0 

        trans = torch.FloatTensor([0.0, 0.0, 0.0])[None, :].to(device).repeat(self.num_images, 1) # Trans Init
        self.trans = nn.Parameter(trans)

        default_joints = torch.zeros(self.num_images, config.N_POSE, 3).to(device)
        self.joint_rotations = nn.Parameter(default_joints)

        # This is a very simple prior over joint rotations, preventing 'splaying' and 'twisting' motions.
        self.rotation_mask = torch.ones(config.N_POSE, 3).to(device)
        # self.rotation_mask[25:32] = 0.0

        # self.rotation_mask[:, [0, 2]] = 0.0
        # self.rotation_mask[:7, [0, 2]] = 1.0
        # self.rotation_mask[[15, 16], [0, 2]] = 1.0
      
        # setup renderers
        self.smal_model = SMAL(shape_family_id=shape_family)
        self.renderer = Renderer(self.image_size, device)

    def print_grads(self, grad_output):
        print (grad_output)

    def forward(self, batch_range, weights, stage_id):
        w_j2d, w_reproj, w_betas, w_pose, w_limit, w_splay = weights
        
        batch_params = {
            'global_rotation' : self.global_rotation[batch_range] * self.global_mask,
            'joint_rotations' : self.joint_rotations[batch_range] * self.rotation_mask,
            'betas' : self.betas.expand(len(batch_range), self.n_betas),
            'trans' : self.trans[batch_range],
        }

        target_joints = self.target_joints[batch_range].to(self.device)
        target_visibility = self.target_visibility[batch_range].to(self.device)
        sil_imgs = self.sil_imgs[batch_range].to(self.device)
    
        verts, joints, Rs, v_shaped = self.smal_model(
                batch_params['betas'], 
                torch.cat([
                    batch_params['global_rotation'].unsqueeze(1), 
                    batch_params['joint_rotations']], dim = 1))

        verts = verts + batch_params['trans'].unsqueeze(1)
        joints = joints + batch_params['trans'].unsqueeze(1)

        labelled_joints = joints[:, config.LABELLED_JOINTS]

        rendered_silhouettes, rendered_joints = self.renderer(
            verts, labelled_joints, self.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1))

        objs = {}

        if w_j2d > 0:
            rendered_joints[~target_visibility.bool()] = -1.0
            target_joints[~target_visibility.bool()] = -1.0

            objs['joint'] = w_j2d * F.mse_loss(rendered_joints, target_joints)

        # if w_limit > 0:
        #     zeros = torch.zeros_like(batch_params['joint_rotations'])
        #     objs['limit'] = w_limit * torch.mean(
        #         torch.max(batch_params['joint_rotations'] - self.max_limits, zeros) + \
        #         torch.max(self.min_limits - batch_params['joint_rotations'], zeros))

        if w_pose > 0:
            objs['pose'] = w_pose * self.pose_prior(batch_params['joint_rotations'])

        if w_splay > 0:
            objs['splay'] = w_splay * torch.sum(batch_params['joint_rotations'][:, :, [0, 2]] ** 2)
        
        if w_betas > 0:
            beta_error = torch.tensordot(batch_params['betas'] - self.mean_betas, self.betas_prec, dims = 1)
            objs['betas'] = w_betas * torch.mean(beta_error ** 2)
    
        if w_reproj > 0:
            objs['sil_reproj'] = w_reproj * F.l1_loss(rendered_silhouettes, sil_imgs)

        return reduce(lambda x, y: x + y, objs.values()), objs

    def get_temporal(self, w_temp):
        joint_rotations = self.joint_rotations * self.rotation_mask
        global_rotation = self.global_rotation * self.global_mask

        joint_loss = torch.tensor(0).to(self.device)
        global_loss = torch.tensor(0).to(self.device)
        trans_loss = torch.tensor(0).to(self.device)

        for i in range(0, self.num_images - 1):
            global_loss += F.mse_loss(global_rotation[i], global_rotation[i + 1]) * w_temp
            joint_loss += F.mse_loss(joint_rotations[i], joint_rotations[i + 1]) * w_temp
            trans_loss += F.mse_loss(self.trans[i], self.trans[i + 1]) * w_temp

        return joint_loss, global_loss, trans_loss

    def load_checkpoint(self, checkpoint_path, epoch):
        beta_list = []
       
        for frame_id in range(self.num_images):
            param_file = os.path.join(checkpoint_path, "{0:04}".format(frame_id), "{0}.pkl".format(epoch))
            with open(param_file, 'rb') as f:
                img_parameters = pkl.load(f)
                self.global_rotation[frame_id] = torch.from_numpy(img_parameters['global_rotation']).float().cuda()
                self.joint_rotations[frame_id] = torch.from_numpy(img_parameters['joint_rotations']).float().cuda().view(32, 3)
                self.trans[frame_id] = torch.from_numpy(img_parameters['trans']).float().cuda()
                beta_list.append(img_parameters['betas'][:self.n_betas])

        self.betas = torch.nn.Parameter(torch.from_numpy(np.mean(beta_list, axis = 0)).float().cuda())

    def generate_visualization(self, image_exporter):
        for j in range(0, self.num_images, self.batch_size):
            batch_range = list(range(j, min(self.num_images, j + self.batch_size)))
            batch_params = {
                'global_rotation' : self.global_rotation[batch_range] * self.global_mask,
                'joint_rotations' : self.joint_rotations[batch_range] * self.rotation_mask,
                'betas' : self.betas.expand(len(batch_range), self.n_betas),
                'trans' : self.trans[batch_range],
            }

            target_joints = self.target_joints[batch_range]
            target_visibility = self.target_visibility[batch_range]
            rgb_imgs = self.rgb_imgs[batch_range].to(self.device)
            sil_imgs = self.sil_imgs[batch_range].to(self.device)

            with torch.no_grad():
                verts, joints, Rs, v_shaped = self.smal_model(
                    batch_params['betas'], 
                    torch.cat([
                        batch_params['global_rotation'].unsqueeze(1), 
                        batch_params['joint_rotations']], dim = 1))

                verts = verts + batch_params['trans'].unsqueeze(1)
                joints = joints + batch_params['trans'].unsqueeze(1)

                labelled_joints = joints[:, config.LABELLED_JOINTS]

                rendered_silhouettes, rendered_joints, rendered_images = self.renderer(
                    verts, labelled_joints, 
                    self.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1), render_texture=True)

                _, rev_joints, rev_images = self.renderer(
                    verts, labelled_joints, 
                    self.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1), render_texture=True)

                target_vis = self.smal_joint_drawer.draw_joints(rgb_imgs, target_joints, visible = target_visibility, normalized=False)
                rendered_images_vis = self.smal_joint_drawer.draw_joints(rendered_images, rendered_joints, visible = target_visibility, normalized=False)
                rev_images_vis = self.smal_joint_drawer.draw_joints(rev_images, rev_joints, visible = target_visibility, normalized=False)

                silhouette_error = 1.0 - F.l1_loss(sil_imgs, rendered_silhouettes, reduction='none')
                silhouette_error = silhouette_error.expand_as(rgb_imgs).data.cpu()

                collage_rows = torch.cat([target_vis, rendered_images_vis, silhouette_error, rev_images_vis], dim = 3)

                for batch_id, global_id in enumerate(batch_range):
                    collage_np = np.transpose(collage_rows[batch_id].numpy(), (1, 2, 0))
                    img_parameters = { k: v[batch_id].cpu().data.numpy() for (k, v) in batch_params.items() }
                    image_exporter.export(
                        (collage_np * 255.0).astype(np.uint8), 
                        batch_id, global_id, img_parameters, 
                        verts, self.smal_model.faces)
            
            


