import numpy as np
import cv2
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import config

class SMALJointDrawer():
    @staticmethod
    def draw_joints(image, landmarks, visible = None):
        image_np = np.transpose(image.cpu().data.numpy(), (0, 2, 3, 1))
        landmarks_np = landmarks.cpu().data.numpy()
        if visible is not None:
            visible_np = visible.cpu().data.numpy()
        else:
            visible_np = visible

        return_stack = SMALJointDrawer.draw_joints_np(image_np, landmarks_np, visible_np)
        return torch.FloatTensor(np.transpose(return_stack, (0, 3, 1, 2)))

    @staticmethod
    def draw_joints_np(image_np, landmarks_np, visible_np = None):
        image_np = (image_np * 255.0).astype(np.uint8)

        bs, nj, _ = landmarks_np.shape
        if visible_np is None:
            visible_np = np.ones((bs, nj), dtype=bool)

        return_images = []
        for image_sgl, landmarks_sgl, visible_sgl in zip(image_np, landmarks_np, visible_np):
            image_sgl = image_sgl.copy()
            inv_ctr = 0
            for joint_id, ((y_co, x_co), vis) in enumerate(zip(landmarks_sgl, visible_sgl)):
                color = np.array(config.MARKER_COLORS)[joint_id]
                marker_type = np.array(config.MARKER_TYPE)[joint_id]
                if not vis:
                    x_co, y_co = inv_ctr * 10, 0
                    inv_ctr += 1
                cv2.drawMarker(image_sgl, (x_co, y_co), (int(color[0]), int(color[1]), int(color[2])), marker_type, 8, thickness = 3)

            return_images.append(image_sgl)

        return_stack = np.stack(return_images, 0)
        return_stack = return_stack / 255.0
        return return_stack

