
from draw_smal_joints import SMALJointDrawer

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle as pkl
import os
import scipy.misc
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn.functional as F
import torch.nn as nn

from functools import reduce

from p3d_renderer import Renderer
from smal_model.smal_torch import SMAL
from priors.pose_prior_35 import Prior
from priors.joint_limits_prior import LimitPrior
import config
from utils import eul_to_axis

class SMALFitter(nn.Module):
    def __init__(self, device, data_batch, batch_size, shape_family, use_unity_prior):
        super(SMALFitter, self).__init__()

        self.rgb_imgs, self.sil_imgs, self.target_joints, self.target_visibility = data_batch
        self.target_visibility = self.target_visibility.long()

        assert self.rgb_imgs.max() <= 1.0 and self.rgb_imgs.min() >= 0.0, "RGB Image range is incorrect"

        self.device = device
        self.num_images = self.rgb_imgs.shape[0]
        self.image_size = self.rgb_imgs.shape[2]
        self.use_unity_prior = use_unity_prior

        self.batch_size = batch_size
        self.n_betas = config.N_BETAS

        self.shape_family_list = np.array(shape_family)
        with open(config.SMAL_DATA_FILE, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            smal_data = u.load()

        if use_unity_prior:
            unity_data = np.load(config.UNITY_SHAPE_PRIOR) 
            model_covs = unity_data['cov'][:-1, :-1]
            mean_betas = torch.from_numpy(unity_data['mean'][:-1]).float().to(device)
            self.mean_betas = mean_betas.clone()

            invcov = np.linalg.inv(model_covs + 1e-5 * np.eye(model_covs.shape[0]))
            prec = np.linalg.cholesky(invcov)        

            self.betas_prec = torch.FloatTensor(prec).to(device)
            self.betas = nn.Parameter(self.mean_betas[:20].clone()) # Shape parameters (1 for the entire sequence... note expand rather than repeat)
            # TODO: edit self.betas here according to N_BETAS. 
            # Either pad with zeros or restrict to less than 20
            self.log_beta_scales = torch.nn.Parameter(self.mean_betas[20:].clone())
        else:
            model_covs = np.array(smal_data['cluster_cov'])[[shape_family]][0]

            invcov = np.linalg.inv(model_covs + 1e-5 * np.eye(model_covs.shape[0]))
            prec = np.linalg.cholesky(invcov)

            self.betas_prec = torch.FloatTensor(prec)[:config.N_BETAS, :config.N_BETAS].to(device)
            self.mean_betas = torch.FloatTensor(smal_data['cluster_means'][[shape_family]][0])[:config.N_BETAS].to(device)
            self.betas = nn.Parameter(self.mean_betas.clone()) # Shape parameters (1 for the entire sequence... note expand rather than repeat)
            self.log_beta_scales = torch.nn.Parameter(
                torch.zeros(self.num_images, 6).to(device), requires_grad=False)

        self.pose_prior = Prior(config.WALKING_PRIOR_FILE, device)

        # TODO: Limit Prior, **shouldn't** be too important for simple sequences :)
        # limit_prior = LimitPrior()
        # self.max_limits = torch.FloatTensor(limit_prior.max_values).view(config.N_POSE, 3).to(device)
        # self.min_limits = torch.FloatTensor(limit_prior.min_values).view(config.N_POSE, 3).to(device)

        global_rotation_np = eul_to_axis(np.array([-np.pi / 2, 0, -np.pi / 2]))
        global_rotation = torch.from_numpy(global_rotation_np).float().to(device).unsqueeze(0).repeat(self.num_images, 1) # Global Init (Head-On)
        self.global_rotation = nn.Parameter(global_rotation)

        trans = torch.FloatTensor([0.0, 0.0, 0.0])[None, :].to(device).repeat(self.num_images, 1) # Trans Init
        self.trans = nn.Parameter(trans)

        default_joints = torch.zeros(self.num_images, config.N_POSE, 3).to(device)
        self.joint_rotations = nn.Parameter(default_joints)

        # Use this to restrict global rotation if necessary
        self.global_mask = torch.ones(1, 3).to(device)
        # self.global_mask[:2] = 0.0

        # Can be used to prevent certain joints rotating. 
        # Can be useful depending on sequence.
        self.rotation_mask = torch.ones(config.N_POSE, 3).to(device)
        # self.rotation_mask[25:32] = 0.0 # e.g. stop the tail moving

        # setup SMAL skinning & differentiable renderer
        self.smal_model = SMAL(device, shape_family_id=shape_family)
        self.renderer = Renderer(self.image_size, device)

    def print_grads(self, grad_output):
        print (grad_output)

    def forward(self, batch_range, weights, stage_id):
        w_j2d, w_reproj, w_betas, w_pose, w_limit, w_splay = weights
        
        batch_params = {
            'global_rotation' : self.global_rotation[batch_range] * self.global_mask,
            'joint_rotations' : self.joint_rotations[batch_range] * self.rotation_mask,
            'betas' : self.betas.expand(len(batch_range), self.n_betas),
            'log_betascale' : self.log_beta_scales.expand(len(batch_range), 6),
            'trans' : self.trans[batch_range],
        }

        target_joints = self.target_joints[batch_range].to(self.device)
        target_visibility = self.target_visibility[batch_range].to(self.device)
        sil_imgs = self.sil_imgs[batch_range].to(self.device)
    
        verts, joints, Rs, v_shaped = self.smal_model(
                batch_params['betas'], 
                torch.cat([
                    batch_params['global_rotation'].unsqueeze(1), 
                    batch_params['joint_rotations']], dim = 1),
                betas_logscale=batch_params['log_betascale'])

        verts = verts + batch_params['trans'].unsqueeze(1)
        joints = joints + batch_params['trans'].unsqueeze(1)

        canonical_model_joints = joints[:, config.CANONICAL_MODEL_JOINTS]

        rendered_silhouettes, rendered_joints = self.renderer(
            verts, canonical_model_joints, 
            self.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1))

        objs = {}

        if w_j2d > 0:
            rendered_joints[~target_visibility.bool()] = -1.0
            target_joints[~target_visibility.bool()] = -1.0

            objs['joint'] = w_j2d * F.mse_loss(rendered_joints, target_joints)

        # TODO
        # if w_limit > 0:
        #     zeros = torch.zeros_like(batch_params['joint_rotations'])
        #     objs['limit'] = w_limit * torch.mean(
        #         torch.max(batch_params['joint_rotations'] - self.max_limits, zeros) + \
        #         torch.max(self.min_limits - batch_params['joint_rotations'], zeros))

        if w_pose > 0:
            objs['pose'] = w_pose * self.pose_prior(
                torch.cat([
                    batch_params['global_rotation'].unsqueeze(1),
                    batch_params['joint_rotations']], dim = 1)).mean()

        if w_splay > 0:
            objs['splay'] = w_splay * torch.sum(batch_params['joint_rotations'][:, :, [0, 2]] ** 2)
        
        if w_betas > 0:
            if self.use_unity_prior:
                all_betas = torch.cat([batch_params['betas'], batch_params['log_betascale']], dim = 1)
            else:
                # TODO: Add a simple regularizer to penalize log_betascale for this case.
                all_betas = batch_params['betas']

            diff_betas = (all_betas - self.mean_betas.unsqueeze(0)) # N, B
            res = torch.tensordot(diff_betas , self.betas_prec, dims = ([1], [0]))
            objs['betas'] = w_betas * (res ** 2).mean()    
        if w_reproj > 0:
            objs['sil_reproj'] = w_reproj * F.l1_loss(rendered_silhouettes, sil_imgs)

        return reduce(lambda x, y: x + y, objs.values()), objs

    def get_temporal(self, w_temp):
        joint_rotations = self.joint_rotations * self.rotation_mask
        global_rotation = self.global_rotation * self.global_mask

        joint_loss = torch.tensor(0.0).float().to(self.device)
        global_loss = torch.tensor(0.0).float().to(self.device)
        trans_loss = torch.tensor(0.0).float().to(self.device)

        for i in range(0, self.num_images - 1):
            global_loss += F.mse_loss(global_rotation[i], global_rotation[i + 1]) * w_temp
            joint_loss += F.mse_loss(joint_rotations[i], joint_rotations[i + 1]) * w_temp
            trans_loss += F.mse_loss(self.trans[i], self.trans[i + 1]) * w_temp

        return joint_loss, global_loss, trans_loss

    def load_checkpoint(self, checkpoint_path, epoch):
        beta_list = []
        scale_list = []
       
        for frame_id in range(self.num_images):
            param_file = os.path.join(checkpoint_path, "{0:04}".format(frame_id), "{0}.pkl".format(epoch))
            with open(param_file, 'rb') as f:
                img_parameters = pkl.load(f)
                self.global_rotation[frame_id] = torch.from_numpy(img_parameters['global_rotation']).float().to(self.device)
                self.joint_rotations[frame_id] = torch.from_numpy(img_parameters['joint_rotations']).float().to(self.device).view(config.N_POSE, 3)
                self.trans[frame_id] = torch.from_numpy(img_parameters['trans']).float().to(self.device)
                beta_list.append(img_parameters['betas'][:self.n_betas])
                scale_list.append(img_parameters['log_betascale'])

        self.betas = torch.nn.Parameter(torch.from_numpy(np.mean(beta_list, axis = 0)).float().to(self.device))
        self.log_beta_scales = torch.nn.Parameter(torch.from_numpy(np.mean(scale_list, axis = 0)).float().to(self.device))

    def generate_visualization(self, image_exporter):
        rot_matrix = torch.from_numpy(R.from_euler('y', 180.0, degrees=True).as_dcm()).float().to(self.device)
        for j in range(0, self.num_images, self.batch_size):
            batch_range = list(range(j, min(self.num_images, j + self.batch_size)))
            batch_params = {
                'global_rotation' : self.global_rotation[batch_range] * self.global_mask,
                'joint_rotations' : self.joint_rotations[batch_range] * self.rotation_mask,
                'betas' : self.betas.expand(len(batch_range), self.n_betas),
                'log_betascale' : self.log_beta_scales.expand(len(batch_range), 6),
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
                        batch_params['joint_rotations']], dim = 1),
                    betas_logscale=batch_params['log_betascale'])

                verts = verts + batch_params['trans'].unsqueeze(1)
                joints = joints + batch_params['trans'].unsqueeze(1)

                canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]

                rendered_silhouettes, rendered_joints, rendered_images = self.renderer(
                    verts, canonical_joints, 
                    self.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1), render_texture=True)

                verts_mean = verts - torch.mean(verts, dim = 1, keepdim=True)
                joints_mean = canonical_joints - torch.mean(verts, dim = 1, keepdim=True)

                _, rev_joints, rev_images = self.renderer(
                    (rot_matrix @ verts_mean.unsqueeze(-1)).squeeze(-1), 
                    (rot_matrix @ joints_mean.unsqueeze(-1)).squeeze(-1), 
                    self.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1), render_texture=True)

                overlay_image = (rendered_images * 0.8) + (rgb_imgs * 0.2)
                
                target_vis = SMALJointDrawer.draw_joints(rgb_imgs, target_joints, visible = target_visibility)
                rendered_images_vis = SMALJointDrawer.draw_joints(rendered_images, rendered_joints, visible = target_visibility)
                rendered_overlay_vis = SMALJointDrawer.draw_joints(overlay_image, rendered_joints, visible = target_visibility)
                rev_images_vis = SMALJointDrawer.draw_joints(rev_images, rev_joints, visible = target_visibility)

                silhouette_error = 1.0 - F.l1_loss(sil_imgs, rendered_silhouettes, reduction='none')
                silhouette_error = silhouette_error.expand_as(rgb_imgs).data.cpu()

                collage_rows = torch.cat([
                    target_vis, rendered_images_vis, 
                    rendered_overlay_vis, silhouette_error, rev_images_vis
                ], dim = 3)

                for batch_id, global_id in enumerate(batch_range):
                    collage_np = np.transpose(collage_rows[batch_id].numpy(), (1, 2, 0))
                    img_parameters = { k: v[batch_id].cpu().data.numpy() for (k, v) in batch_params.items() }
                    image_exporter.export(
                        (collage_np * 255.0).astype(np.uint8), 
                        batch_id, global_id, img_parameters, 
                        verts, self.smal_model.faces.data.cpu().numpy())
            
            


