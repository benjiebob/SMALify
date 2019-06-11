import cv2
import numpy as np

def crop_to_silhouette(sil_img, rgb_img, joints, target_size):
    assert len(sil_img.shape) == 2, "Silhouette image is not HxW"
    assert len(rgb_img.shape) == 3, "RGB image is not HxWx3"

    sil_h, sil_w = sil_img.shape
    pad_sil = np.zeros((sil_h * 4, sil_w * 4))
    pad_rgb = np.ones((sil_h * 4, sil_w * 4, 3))

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