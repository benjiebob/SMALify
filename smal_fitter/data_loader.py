
import sys
sys.path.append('../')

import numpy as np
import cv2
import torch
import imageio
from tqdm import tqdm
from utils import crop_to_silhouette
import os
import json

import config

def load_badja_sequence(BADJA_PATH, sequence_name, crop_size, image_range = None):
    annotated_classes = config.ANNOTATED_CLASSES

    file_names = []
    rgb_imgs = []
    sil_imgs = []
    joints = []
    visibility = []

    annotations_path = os.path.join(BADJA_PATH, "joint_annotations")
    json_path = os.path.join(annotations_path, "{0}.json".format(sequence_name))
    with open(json_path) as json_data:
        sequence_annotation = np.array(json.load(json_data))
        if image_range is not None:
            sequence_annotation = sequence_annotation[image_range]
        for image_annotation in tqdm(sequence_annotation):
            file_name = os.path.join(BADJA_PATH, image_annotation['image_path'])
            seg_name = os.path.join(BADJA_PATH, image_annotation['segmentation_path'])

            if os.path.exists(file_name) and os.path.exists(seg_name):
                landmarks = np.array(image_annotation['joints'])[annotated_classes]
                visibility.append(np.array(image_annotation['visibility'])[annotated_classes])

                rgb_img = imageio.imread(file_name) / 255.0
                sil_img = imageio.imread(seg_name)[:, :, 0] / 255.0

                rgb_h, rgb_w, _ = rgb_img.shape
                sil_img = cv2.resize(sil_img, (rgb_w, rgb_h), cv2.INTER_NEAREST)

                sil_img, rgb_img, landmarks = crop_to_silhouette(sil_img, rgb_img, landmarks, crop_size)

                rgb_imgs.append(rgb_img)
                sil_imgs.append(sil_img)
                joints.append(landmarks)
                file_names.append(os.path.basename(image_annotation['image_path']))

            elif os.path.exists(file_name):
                print ("BADJA SEGMENTATION file path: {0} is missing".format(seg_name))
            else:
                print ("BADJA IMAGE file path: {0} is missing".format(file_name))

    rgb = torch.FloatTensor(np.stack(rgb_imgs, axis = 0)).permute(0, 3, 1, 2)
    sil = torch.FloatTensor(np.stack(sil_imgs, axis = 0))[:, None, :, :]
    joints = torch.FloatTensor(np.stack(joints, axis = 0))
    visibility = torch.FloatTensor(np.stack(visibility, axis = 0).astype(np.float))

    return (rgb, sil, joints, visibility), file_names
