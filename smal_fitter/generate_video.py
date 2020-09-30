
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
import config

from data_loader import load_badja_sequence, load_stanford_sequence
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
    OUTPUT_DIR = os.path.join("exported", config.CHECKPOINT_NAME, config.EPOCH_NAME)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset, name = config.SEQUENCE_OR_IMAGE_NAME.split(":")

    if dataset == "badja":
        data, filenames = load_badja_sequence(
            config.BADJA_PATH, name, 
            config.CROP_SIZE, image_range=config.IMAGE_RANGE)
    else:
        data, filenames = load_stanford_sequence(
            config.STANFORD_EXTRA_PATH, name,
            config.CROP_SIZE
        )

    dataset_size = len(filenames)
    print ("Dataset size: {0}".format(dataset_size))

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    use_unity_prior = config.SHAPE_FAMILY == 1 and not config.FORCE_SMAL_PRIOR

    if not use_unity_prior and not config.ALLOW_LIMB_SCALING:
        print("WARNING: Limb scaling is only recommended for the new Unity prior. TODO: add a regularizer to constrain scale parameters.")
        config.ALLOW_LIMB_SCALING = False

    image_exporter = ImageExporter(OUTPUT_DIR)
    model = SMALFitter(device, data, config.WINDOW_SIZE, config.SHAPE_FAMILY, use_unity_prior)

    model.load_checkpoint(os.path.join("checkpoints", config.CHECKPOINT_NAME), config.EPOCH_NAME)
    model.generate_visualization(image_exporter) # Final stage

if __name__ == '__main__':
    main()
