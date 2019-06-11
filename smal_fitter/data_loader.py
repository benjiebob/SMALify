
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

from smal.joint_catalog import SMALJointInfo

def load_data_from_npz(input_folder):
    smal_info = SMALJointInfo()
    # include_classes = [0, 2, 5, 11, 12, 13, 7, 8, 9, 14, 15, 21, 22, 23, 17, 18, 19, 24, 27, 30, 31, 32, 35, 34, 33] # Flip Front and Back
    include_classes = [0, 2, 5, 11, 12, 13, 7, 8, 9, 14, 15, 21, 22, 23, 17, 18, 19, 24, 27, 30, 31, 32, 35] # Flip Front and Back
    input_files = list(filter(lambda x: os.path.splitext(x)[1] == ".npz", sorted(os.listdir(input_folder))))

    rgb_imgs = []
    sil_imgs = []
    target_joints = []
    target_visibility = []
    for fn in input_files:
        npz_file = os.path.join(input_folder, fn)
        data = np.load(npz_file)
        rgb_imgs.append(data['rgb_img'] / 255.0)
        sil_imgs.append(data['target_img'])
        target_joints.append(data['joint_coords'][include_classes][:, ::-1])
        visibility = np.invert(data['null_joints'][include_classes]).astype(float)
        
        visibility[np.all(np.where(target_joints) == [0.0, 0.0])] == 0.0
        visibility[-2:] = 0.0

        target_visibility.append(visibility)

    rgb = torch.FloatTensor(np.stack(rgb_imgs, axis = 0)).permute(0, 3, 1, 2)
    sil = torch.FloatTensor(np.stack(sil_imgs, axis = 0))[:, None, :, :]
    joints = torch.FloatTensor(np.stack(target_joints, axis = 0))
    target_visibility = torch.FloatTensor(np.stack(target_visibility, axis = 0).astype(np.float))

    return (rgb, sil, joints, target_visibility), input_files

def load_badja_sequence(BADJA_PATH, sequence_name, crop_size, image_range = None):
    annotated_classes = SMALJointInfo().annotated_classes

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
