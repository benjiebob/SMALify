import cv2
import numpy as np
from nibabel import eulerangles

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

## Function courtesy of SMALST
def perspective_proj_withz(X, cam, offset_z=0, cuda_device=0,norm_f=1., norm_z=0.,norm_f0=0.):
    """
    X: B x N x 3
    cam: B x 3: [f, cx, cy] 
    offset_z is for being compatible with previous code and is not used and should be removed
    """

    # B x 1 x 1
    #f = norm_f * cam[:, 0].contiguous().view(-1, 1, 1)
    f = norm_f0+norm_f * cam[:, 0].contiguous().view(-1, 1, 1)
    # B x N x 1
    z = norm_z + X[:, :, 2, None]

    # Will z ever be 0? We probably should max it..
    eps = 1e-6 * torch.ones(1).cuda(device=cuda_device)
    z = torch.max(z, eps)
    image_size_half = cam[0,1]
    scale = f / (z*image_size_half)

    # Offset is because cam is at -1
    return torch.cat((scale * X[:, :, :2], z+offset_z),2)

def eul_to_axis(euler_value):
    theta, vector = eulerangles.euler2angle_axis(euler_value[2], euler_value[1], euler_value[0])
    return vector * theta