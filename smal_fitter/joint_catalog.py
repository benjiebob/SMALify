from enum import Enum
import numpy as np
import cv2

class SMALJointCatalog(Enum):
    body_0 = 0 # 0
    # body_1 = 1
    # body_2 = 2
    body_3 = 3 # 1
    # body_4 = 4
    # body_5 = 5
    body_6 = 6 # 2
    upper_right_0 = 7
    upper_right_1 = 8 # 3
    upper_right_2 = 9 # 4
    upper_right_3 = 10 # 5
    # upper_left_0 = 11
    upper_left_1 = 12 # 6
    upper_left_2 = 13 # 7
    upper_left_3 = 14 # 8
    neck_lower = 15 # 9
    neck_upper = 16 # 10
    # lower_right_0 = 17
    lower_right_1 = 18 # 11
    lower_right_2 = 19 # 12
    lower_right_3 = 20 # 13
    # lower_left_0 = 21
    lower_left_1 = 22 # 14
    lower_left_2 = 23 # 15
    lower_left_3 = 24 # 16
    tail_0 = 25 # 17
    # tail_1 = 26
    # tail_2 = 27
    tail_3 = 28 # 18
    # tail_4 = 29
    # tail_5 = 30
    tail_6 = 31 # 19
    jaw = 32 # 20
    nose = 33 # 21
    chin = 34 # 22
    right_ear = 35 # 23
    left_ear = 36 # 24

class SMALJointInfo():
    def __init__(self):
        self.annotated_classes = [
            0, 3, 6, # body
            8, 9, 10, # upper_right
            12, 13, 14, # upper_left
            15, 16, # neck
            18, 19, 20, # lower_right
            22, 23, 24, # lower_left
            25, 28, 31, # tail
            32, 33, 34] # head
            # 35, # right_ear
            # 36] # left_ear

        self.marker_type = np.array([
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS, cv2.MARKER_STAR, cv2.MARKER_TRIANGLE_DOWN,
            cv2.MARKER_CROSS,
            cv2.MARKER_CROSS])

        self.joint_region_all = np.array([ 
            0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 
            2, 2, 2, 2, 
            3, 3,
            4, 4, 4, 4, 
            5, 5, 5, 5, 
            6, 6, 6, 6, 6, 6, 6,
            7, 7, 7,
            8, 
            9])

        
        self.joint_region = self.joint_region_all[self.annotated_classes]
        self.region_colors = np.array([
            [250, 190, 190], # body, light pink
            [60, 180, 75], # upper_right, green
            [230, 25, 75], # upper_left, red
            [128, 0, 0], # neck, maroon
            [0, 130, 200], # lower_right, blue
            [255, 255, 25], # lower_left, yellow
            [240, 50, 230], # tail, majenta
            [245, 130, 48], # jaw / nose / chin, orange
            [29, 98, 115], # right_ear, turquoise
            [255, 153, 204]]) # left_ear, pink
        
        self.joint_colors = np.array(self.region_colors)[self.joint_region]