
####
### ffmpeg -framerate 50 -i %04d.png -pix_fmt yuv420p rs_dog.gif
###

import sys, os
sys.path.append(os.path.dirname(sys.path[0]))

import numpy as np
import cv2
import argparse

import matplotlib.pyplot as plt
from smal_fitter import SMALFitter

import torch
import imageio

from data_loader import load_badja_sequence
import time

import pickle as pkl

class ImageExporter():
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def export(self, collage_np, batch_id, global_id, img_parameters, vertices, faces):
        imageio.imsave(os.path.join(self.output_dir, "{0:04}.png".format(global_id)), collage_np)

        with open(os.path.join(self.output_dir, "{0:04}.pkl".format(global_id)), 'wb') as f:
            pkl.dump(img_parameters, f)

def main():
    BADJA_PATH = "data/BADJA"
    SHAPE_FAMILY = [1]

    CHECKPOINT_NAME = "rs_dog"
    EPOCH_NAME = "st10_ep0"

    OUTPUT_DIR = os.path.join("exported", CHECKPOINT_NAME, EPOCH_NAME)
    WINDOW_SIZE = 5
    CROP_SIZE = 256
    GPU_IDS = "0"

    image_exporter = ImageExporter(OUTPUT_DIR)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_IDS

    data, filenames = load_badja_sequence(BADJA_PATH, CHECKPOINT_NAME, CROP_SIZE)

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
