import numpy as np
import cv2
import argparse

import matplotlib.pyplot as plt
from smal_fitter import SMALFitter

import torch
import scipy.misc

from data_loader import load_badja_sequence

import os, time
import sys

OPT_WEIGHTS = np.array([
    [25.0, 10.0, 7.5, 5.0], # Joint
    [0.0, 0.0, 100.0, 100.0], # Sil Reproj
    [0.0, 0.1, 1.0, 0.1], # Betas
    [0.0, 50.0, 100.0, 100.0], # Limits
    [500.0, 100.0, 100.0, 100.0], # Temporal
    [150, 500, 500, 500], # Num iterations
    [1e-1, 2.5e-3, 5e-3, 5e-3]]) # Learning Rate

IM_FREQ = 100

def main():
    BADJA_PATH = "BADJA"
    SHAPE_FAMILY = [1]
    OUTPUT_DIR = "checkpoints/{0}".format(time.strftime("%Y%m%d-%H%M%S"))
    WINDOW_SIZE = 25
    CROP_SIZE = 256
    GPU_IDS = "1"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS

    data, filenames = load_badja_sequence(BADJA_PATH, "rs_dog", CROP_SIZE, num_images=4)

    dataset_size = len(filenames)
    print ("Dataset size: {0}".format(dataset_size))

    plt.figure()
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    model = SMALFitter(data, filenames, WINDOW_SIZE, SHAPE_FAMILY, OUTPUT_DIR)
    for stage_id, weights in enumerate(OPT_WEIGHTS.T):
        # torch.save(model.batch_params, os.path.join(out_dir, "st{0}.smal".format(stage_id)))
        opt_weight = weights[:4]
        w_temp = weights[4]
        epochs = int(weights[5])
        lr = weights[6]

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))

        if stage_id == 0:
            model.joint_rotations.requires_grad = False
            model.betas.requires_grad = False
        else:
            model.joint_rotations.requires_grad = True
            model.betas.requires_grad = True

        for epoch_id in range(epochs):
            acc_loss = 0
            optimizer.zero_grad()
            for j in range(0, dataset_size, WINDOW_SIZE):
                batch_range = list(range(j, min(dataset_size, j + WINDOW_SIZE)))
                loss, losses = model(batch_range, opt_weight)
                acc_loss += loss
                print ("Optimizing Stage: {}\t Epoch: {}, Range: {}, Loss: {}, Detail: {}".format(stage_id, epoch_id, batch_range, loss.data, losses))

            joint_loss, global_loss, trans_loss = model.get_temporal(w_temp)

            print ("EPOCH: Optimizing Stage: {}\t Epoch: {}, Loss: {}, Temporal: ({}, {}, {})".format(stage_id, epoch_id, acc_loss.data, joint_loss.data, global_loss.data, trans_loss.data))
            acc_loss += joint_loss + global_loss + trans_loss
            acc_loss.backward()
            optimizer.step()

            if epoch_id % 10 == 0:
                model.generate_visualization(stage_id, epoch_id)

    model.generate_visualization(5, 0) # Final stage

if __name__ == '__main__':
    main()