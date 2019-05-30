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


class ImageExporter():
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def export(self, collage_np, batch_id, global_id, batch_params, vertices, faces):
        scipy.misc.imsave(os.path.join(self.output_dir, "{0:04}.png".format(global_id)), collage_np)

def main():
    BADJA_PATH = "smal_fitter/BADJA"
    SHAPE_FAMILY = [1]
    CHECKPOINT_NAME = "20190529-210223"
    EPOCH_NAME = "st5_ep0"
    OUTPUT_DIR = os.path.join("smal_fitter", "exported", CHECKPOINT_NAME, EPOCH_NAME)
    WINDOW_SIZE = 25
    CROP_SIZE = 256
    GPU_IDS = "0"

    image_exporter = ImageExporter(OUTPUT_DIR)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS

    data, filenames = load_badja_sequence(BADJA_PATH, "rs_dog", CROP_SIZE)
    dataset_size = len(filenames)
    print ("Dataset size: {0}".format(dataset_size))

    plt.figure()
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model = SMALFitter(data, WINDOW_SIZE, SHAPE_FAMILY)
    model.load_checkpoint(os.path.join("smal_fitter", "checkpoints", CHECKPOINT_NAME), EPOCH_NAME)
    model.generate_visualization(image_exporter) # Final stage

if __name__ == '__main__':
    main()
