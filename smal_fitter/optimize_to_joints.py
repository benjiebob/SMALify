
import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import cv2
import argparse

import matplotlib.pyplot as plt
from smal_fitter import SMALFitter
import pickle as pkl

import torch
import imageio

from data_loader import load_badja_sequence, load_data_from_npz
import trimesh

import os, time
import sys

OPT_WEIGHTS = np.array([
    [25.0, 10.0, 7.5, 5.0], # Joint
    [0.0, 0.0, 100.0, 100.0], # Sil Reproj
    [0.0, 0.0, 0.0, 0.0], # Betas
    [0.0, 100.0, 100.0, 100.0], # Limits
    [0.0, 100.0, 100.0, 100.0], # Splay
    [500.0, 100.0, 100.0, 100.0], # Temporal
    [150, 500, 500, 500], # Num iterations
    [1e-1, 5e-3, 5e-3, 2.5e-3]]) # Learning Rate

class ImageExporter():
    def __init__(self, output_dir, filenames):
        self.output_dirs = self.generate_output_folders(output_dir, filenames)
        self.stage_id = 0
        self.epoch_name = 0

    def generate_output_folders(self, root_directory, filename_batch):
        if not os.path.exists(root_directory):
            os.mkdir(root_directory)

        output_dirs = [] 
        for filename in filename_batch:
            filename_path = os.path.join(root_directory, os.path.splitext(filename)[0])
            output_dirs.append(filename_path)
            os.mkdir(filename_path)
        
        return output_dirs

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces):
        imageio.imsave(os.path.join(self.output_dirs[global_id], "st{0}_ep{1}.png".format(self.stage_id, self.epoch_name)), collage_np)

        # Export parameters
        with open(os.path.join(self.output_dirs[global_id], "st{0}_ep{1}.pkl".format(self.stage_id, self.epoch_name)), 'wb') as f:
            pkl.dump(img_parameters, f)

        # Export mesh
        vertices = vertices[batch_id].cpu().numpy()
        mesh = trimesh.Trimesh(vertices = vertices, faces = faces, process = False)
        mesh.export(os.path.join(self.output_dirs[global_id], "st{0}_ep{1}.ply".format(self.stage_id, self.epoch_name)))

def main():
    BADJA_PATH = "smal_fitter/BADJA"
    INPUT_PATH = "/data/cvfs/bjb56/data/smal_data/smal_joints/hg/24_04/prediction/"

    OUTPUT_DIR = "smal_fitter/checkpoints/{0}".format(time.strftime("%Y%m%d-%H%M%S"))

    INPUT_NAME = "cosker-maggie"
    CLEANED_NAME = "20190522-140530_rocky_rl6_pop256" 

    SHAPE_FAMILY = [1]
    WINDOW_SIZE = 100
    CROP_SIZE = 256
    GPU_IDS = "1"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS

    # image_range = range(170, 174) # Reversed
    # image_range = range(145, 155) # Twist
    image_range = range(0, 4) # Run
    # image_range = None

    # data, filenames = load_badja_sequence(BADJA_PATH, "rs_dog", CROP_SIZE, image_range=image_range)
    data, filenames = load_data_from_npz(os.path.join(INPUT_PATH, INPUT_NAME, "cleaned_skeleton", CLEANED_NAME))

    dataset_size = len(filenames)
    print ("Dataset size: {0}".format(dataset_size))

    image_exporter = ImageExporter(OUTPUT_DIR, filenames)

    model = SMALFitter(data, WINDOW_SIZE, SHAPE_FAMILY)
    for stage_id, weights in enumerate(OPT_WEIGHTS.T):
        opt_weight = weights[:5]
        w_temp = weights[5]
        epochs = int(weights[6])
        lr = weights[7]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

        if stage_id == 0:
            model.joint_rotations.requires_grad = False
            model.betas.requires_grad = False
            model.target_visibility *= 0
            model.target_visibility[:, [9, 17, 3, 6, 11, 14]] = 1.0 # Turn on only torso points
        else:
            model.joint_rotations.requires_grad = True
            model.betas.requires_grad = True
            model.target_visibility = data[-1].clone()

        for epoch_id in range(epochs):
            image_exporter.stage_id = stage_id
            image_exporter.epoch_name = str(epoch_id)

            acc_loss = 0
            optimizer.zero_grad()
            for j in range(0, dataset_size, WINDOW_SIZE):
                batch_range = list(range(j, min(dataset_size, j + WINDOW_SIZE)))
                loss, losses = model(batch_range, opt_weight, stage_id)
                acc_loss += loss
                print ("Optimizing Stage: {}\t Epoch: {}, Range: {}, Loss: {}, Detail: {}".format(stage_id, epoch_id, batch_range, loss.data, losses))

            joint_loss, global_loss, trans_loss = model.get_temporal(w_temp)

            print ("EPOCH: Optimizing Stage: {}\t Epoch: {}, Loss: {}, Temporal: ({}, {}, {})".format(stage_id, epoch_id, acc_loss.data, joint_loss.data, global_loss.data, trans_loss.data))
            acc_loss += joint_loss + global_loss + trans_loss
            acc_loss.backward()
            optimizer.step()

            if epoch_id % 10 == 0:
                model.generate_visualization(image_exporter)

    image_exporter.stage_id = 10
    image_exporter.epoch_name = str(0)
    model.generate_visualization(image_exporter) # Final stage

if __name__ == '__main__':
    main()
