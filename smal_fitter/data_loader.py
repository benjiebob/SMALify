
import sys
sys.path.append('../')

import numpy as np
import cv2
import torch
import scipy.misc
from tqdm import tqdm
import os
import json

from smal.joint_catalog import SMALJointInfo

def crop_to_silhouette(sil_img, rgb_img, joints, target_size):
    assert len(sil_img.shape) == 2, "Silhouette image is not HxW"
    assert len(rgb_img.shape) == 3, "RGB image is not HxWx3"

    sil_h, sil_w = sil_img.shape
    pad_sil = np.zeros((sil_h * 4, sil_w * 4))
    pad_rgb = np.zeros((sil_h * 4, sil_w * 4, 3))

    pad_sil[sil_h * 2 : sil_h * 3, sil_w * 2 : sil_w * 3] = sil_img
    pad_rgb[sil_h * 2 : sil_h * 3, sil_w * 2 : sil_w * 3, :] = rgb_img

    foreground_pixels = np.where(pad_sil > 0)
    y_min, y_max, x_min, x_max = np.amin(foreground_pixels[0]), np.amax(foreground_pixels[0]), np.amin(foreground_pixels[1]), np.amax(foreground_pixels[1])

    square_half_side_length = int(1.05 * (max(x_max - x_min, y_max - y_min) / 2))
    centre_y = y_min + int((y_max - y_min) / 2)
    centre_x = x_min + int((x_max - x_min) / 2)

    square_sil = pad_sil[centre_y - square_half_side_length : centre_y + square_half_side_length, centre_x - square_half_side_length : centre_x + square_half_side_length]
    square_rgb = pad_rgb[centre_y - square_half_side_length : centre_y + square_half_side_length, centre_x - square_half_side_length : centre_x + square_half_side_length]

    sil_resize = cv2.resize(square_sil, (target_size, target_size), interpolation = cv2.INTER_NEAREST)
    rgb_resize = cv2.resize(square_rgb, (target_size, target_size))

    scaled_joints = np.zeros_like(joints)
    scaled_joints[:, 0] = joints[:, 0] + (sil_h * 2) - (centre_y - square_half_side_length)
    scaled_joints[:, 1] = joints[:, 1] + (sil_w * 2) - (centre_x - square_half_side_length)
    
    scale_factor = target_size / (square_half_side_length * 2.0)
    scaled_joints = scaled_joints * scale_factor
    
    return sil_resize, rgb_resize, scaled_joints

def load_badja_sequence(BADJA_PATH, sequence_name, crop_size, num_images = None):
    annotated_classes = SMALJointInfo().annotated_classes

    file_names = []
    rgb_imgs = []
    sil_imgs = []
    joints = []
    visibility = []

    annotations_path = os.path.join(BADJA_PATH, "joint_annotations")
    json_path = os.path.join(annotations_path, "{0}.json".format(sequence_name))
    with open(json_path) as json_data:
        sequence_annotation = json.load(json_data)
        if num_images is not None:
            sequence_annotation = sequence_annotation[:num_images]
        for image_annotation in tqdm(sequence_annotation):
            file_name = os.path.join(BADJA_PATH, image_annotation['image_path'])
            seg_name = os.path.join(BADJA_PATH, image_annotation['segmentation_path'])

            if os.path.exists(file_name) and os.path.exists(seg_name):
                landmarks = np.array(image_annotation['joints'])[annotated_classes]
                visibility.append(np.array(image_annotation['visibility'])[annotated_classes])

                rgb_img = scipy.misc.imread(file_name, mode='RGB') / 255.0
                sil_img = scipy.misc.imread(seg_name, mode='RGB')[:, :, 0] / 255.0

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
    visibility = torch.FloatTensor(np.stack(visibility, axis = 0))

    return (rgb, sil, joints, visibility), file_names
