"""
This file contains definitions of useful data stuctures and the paths
for the datasets and data files necessary to run the code.
Things you need to change: *_ROOT that indicate the path to each dataset
"""
from os.path import join

# Define paths to each dataset

BADJA_PATH = '/nvme_scratch1/bjb10042/data/dogs_v2'
CLIP_NAME = 'rs_dog'
data_path = 'data'

# SMAL
SMAL_FILE = join(data_path, 'smal', 'my_smpl_00781_4_all.pkl')
SMAL_DATA_FILE = join(data_path, 'smal', 'my_smpl_data_00781_4_all.pkl')
SMAL_UV_FILE = join(data_path, 'smal', 'my_smpl_00781_4_all_template_w_tex_uv_001.pkl')
SMAL_SYM_FILE = join(data_path, 'smal', 'symIdx.pkl')

# PRIORS
WALKING_PRIOR_FILE = join(data_path, 'priors', 'walking_toy_symmetric_pose_prior_with_cov_35parts.pkl')
UNITY_SHAPE_PRIOR = join(data_path, 'priors', 'unity_betas.npz')
SMAL_DOG_TOY_IDS = [0, 1, 2] # Olly TODO

# DATALOADER
IMG_RES = 224

# RENDERER
PROJECTION = 'perspective'
NORM_F0 = 2700.0
NORM_F = 2700.0
NORM_Z = 20.0

MESH_COLOR = [0, 172, 223]

# MESH_NET
NZ_FEAT = 100

# ASSOCIATING SMAL TO ANNOTATED JOINTS
# LABELLED_JOINTS = [
#   14, 13, 12, # left front (0, 1, 2)
#   24, 23, 22, # left rear (3, 4, 5)
#   10, 9, 8, # right front (6, 7, 8)
#   20, 19, 18, # right rear (9, 10, 11)
#   25, 31, # tail start -> end (12, 13)
#   34, 33, # right ear, left ear (14, 15)
#   35, 36, # nose, chin (16, 17)
#   38, 37, # right tip, left tip (18, 19)
#   39, 40] # left eye, right eye

LABELLED_JOINTS = [
    10, 9, 8, # upper_right
    14, 13, 12, # upper_left
    15, # neck ????
    20, 19, 18, # lower_right
    24, 23, 22, # lower_left
    25, 28, 31, # tail
    35, 36, # nose, chin
    38, 37] # right_ear, left_ear

ANNOTATED_CLASSES = [
    8, 9, 10, # upper_right
    12, 13, 14, # upper_left
    15, # neck
    18, 19, 20, # lower_right
    22, 23, 24, # lower_left
    25, 28, 31, # tail
    32, 33, # head
    35, # right_ear
    36] # left_ear

# NUMBER OF KEYPOINTS CONSIDERED. FOR NOW ONLY INCLUDE UP TO JOINT 21 - IGNORING WITHERS AND THROAT
N_KEYPOINTS = 22

N_POSE = 34
N_BETAS = 20