import pickle as pkl
import numpy as np
from chumpy import Ch
import cv2

name2id35 = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1, 'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15, 'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11, 'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27, 'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22, 'root': 0, 'LEar': 33, 'REar' : 34}
id2name35 = {v: k for k, v in name2id35.items()}

name2id33 = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1, 'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15, 'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11, 'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27, 'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22, 'root': 0}
id2name33 = {v: k for k, v in name2id33.items()}

name2id31 = {'RFoot': 12, 'RFootBack': 22, 'LLegBack1': 15, 'spine1': 2, 'Head': 14, 'RLegBack1': 19, 'RLegBack2': 20, 'RLegBack3': 21, 'LLegBack2': 16, 'LLegBack3': 17, 'spine3': 4, 'spine2': 3, 'Mouth': 30, 'Neck': 13, 'LFootBack': 18, 'LLeg1': 5, 'RLeg2': 10, 'RLeg3': 11, 'LLeg3': 7, 'RLeg1': 9, 'LLeg2': 6, 'spine': 1, 'LFoot': 8, 'Tail7': 29, 'Tail6': 28, 'Tail5': 27, 'Tail4': 26, 'Tail3': 25, 'Tail2': 24, 'Tail1': 23, 'root': 0}
id2name31 = {v: k for k, v in name2id31.items()}


def get_ignore_names(path):
    if 'notail' in path:
        # ignore_joints = range(22, 30)
        ignore_names = [key for key in name2id.keys() if 'Tail' in key]
    elif 'bodyneckelbowtail' in path:
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack']
    elif 'body_indept_limbstips' in path:
        # Ignore joints:
        # (head:13), + tail 22:28
        # ignore_joints = [13] + range(22, 30)
        ignore_names = ['Neck', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']

    elif 'prior_bodyelbow' in path:
        # Ignore joints:
        # 6, 7, 10, 11, (neck, head:12, 13), 16, 17, 20:28
        # ignore_joints = [6, 7, 10, 11, 12, 13, 16, 17] + range(20, 30)
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'RFoot', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']

    elif 'bodyneckheadelbow' in path:
        # Ignore joints:
        # 6, 7, 10, 11, 16, 17, 20:29
        # ignore_joints = [6, 7, 10, 11, 16, 17] + range(20, 30)
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']
    elif 'bodyneckelbow' in path:
        # Ignore joints:
        # 6, 7, 10, 11, (head:13), 16, 17, 20:28
        # ignore_joints = [6, 7, 10, 11, 13, 16, 17] + range(20, 30)
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']
    elif 'bodyneck' in path:
        ignore_names = ['LLeg1', 'LLeg2', 'LLeg3', 'RLeg1', 'RLeg2', 'RLeg3', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack']
    elif 'backlegstail' in path:
        ignore_names = ['root', 'RFoot', 'RFootBack', 'spine1', 'Head', 'pelvis0', 'spine0', 'spine3', 'spine2', 'Mouth', 'Neck', 'LFootBack', 'RLeg3', 'RLeg2', 'LLeg1', 'LLeg3', 'RLeg1', 'LLeg2', 'spine', 'LFoot']
    elif '_body.pkl' in path:
        # Ignore joints:
        # 5, 6, 7, 9, 10, 11, (neck, head:12, 13), 15, 16, 17, 19:29
        # ignore_joints = range(5, 8) + range(9, 14) + range(15, 18) + range(19,
        #                                                                    30)
        ignore_names = ['LLeg1', 'LLeg2', 'LLeg3', 'RLeg1', 'RLeg2', 'RLeg3', 'RFoot', 'Neck', 'LLegBack1', 'LLegBack2', 'LLegBack3', 'RLegBack1', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']
    else:
        ignore_names = []

    return ignore_names

import torch
class Prior(object):
    def __init__(self, prior_path):
        with open(prior_path, "rb") as f:
            res = pkl.load(f, encoding='latin1')
        self.precs = torch.from_numpy(res['pic'].r).float().cuda()
        self.mean = torch.from_numpy(res['mean_pose']).float().cuda()

        self.mean_ch = res['mean_pose']
        self.precs_ch = res['pic']

        # if 'cov' in res.keys():
        #     self.cov = torch.from_numpy(res['cov']).float().cuda()
        # Mouth closed!
        # self.mean[-2] = -0.4
        # Ignore the first 3 global rotation.
        prefix = 3
        if '33parts' in prior_path:
            pose_len = 99
            id2name = id2name33
            name2id = name2id33
        elif '35parts' in prior_path:
            pose_len = 35*3
            id2name = id2name35
            name2id = name2id35
        else:
            pose_len = 93
            id2name = id2name31
            name2id = name2id31

        self.use_ind = np.ones(pose_len, dtype=bool)
        self.use_ind[:prefix] = False

        self.use_ind_tch = torch.from_numpy(self.use_ind).float().cuda()

        ignore_names = get_ignore_names(prior_path)

        if len(ignore_names) > 0:
            ignore_joints = sorted([name2id[name] for name in ignore_names])
            #ignore_inds = np.hstack([3 + np.array(ig * 3) + np.arange(0, 3)
            #                         for ig in ignore_joints])
            # Silvia: do not start from 0
            ignore_inds = np.hstack([np.array(ig * 3) + np.arange(0, 3)
                                     for ig in ignore_joints])
            self.use_ind[ignore_inds] = False

            # Test to make sure that the ignore indices are ok.
            # test = use_ind.reshape((-1, 3))
            # # Make sure all rows are the same
            # assert(np.all(test[:, 0] == test[:, 1]) and np.all(test[:, 0] == test[:, 1]))
            # if pose_len == 99:
            #     id2name = id2name33
            # else:
            #     id2name = id2name31
            # print('Using parts:')
            # print([id2name[id] for id in (test[:, 0] == True).nonzero()[0]])
            # print('Ignoring parts:')
            # print([id2name[id] for id in (test[:, 0] == False).nonzero()[0]])
            # import ipdb; ipdb.set_trace()

    def __call__(self, x):
        # return row(x[self.prefix:] - self.mean).dot(self.precs)
        # return (x[self.use_ind] - self.mean).dot(self.precs)
        # res = (x[self.use_ind] - self.mean).dot(self.precs)
        mean_sub = x - self.mean.unsqueeze(0)
        res = torch.tensordot(mean_sub, self.precs, dims = ([1], [0])) * self.use_ind_tch
        res_ch = (x.data.cpu().numpy() - self.mean_ch).dot(self.precs_ch) * self.use_ind

        error = np.linalg.norm(res.data.cpu().numpy() - res_ch)

        print (error)

        assert error < 1e-2, "ERROR VERY LARGE"
        return res

def abs_to_rel(r_abs, model):
    r_rel = np.zeros_like(r_abs)
    partSet = model.kintree_table[1, :]
    parents = model.kintree_table[0, :]
    for part in partSet:
        parent = parents[part]
        if parent != 4294967295:
            R0_parent, _ = cv2.Rodrigues(r_abs[parent, :])
            R0_part, _ = cv2.Rodrigues(r_abs[part, :])
            R = R0_parent.T.dot(R0_part)
            r, _ = cv2.Rodrigues(R)
            r_rel[part, :] = r.T

    return r_rel


def load_silvia_data(seq_path, model):
    with open(seq_path) as f:
        data = pkl.load(f)
    r_abs_all = data['r_abs_all']
    res = []
    for r_abs in r_abs_all:
        res.append(abs_to_rel(r_abs, model))

    poses = np.dstack(res).reshape(31*3, -1).T

    return poses


def model31_to_33(pose):
    new_pose = np.zeros((99))
    new_pose[:3] = pose[:3]
    new_pose[3:6] = pose[:3]
    new_pose[6:9] = pose[3:6]
    new_pose[9:12] = pose[3:6]
    new_pose[12:] = pose[6:]
    return new_pose


def reflect_pose(pose, name2id, model=None):
    # Parts that has to be swapped:
    right = ['RLeg1', 'RLeg2', 'RLeg3', 'RFootBack', 'RLegBack1', 'RLegBack2', 'RLegBack3', 'RFoot']
    left =  ['LLeg1', 'LLeg2', 'LLeg3', 'LFootBack', 'LLegBack1', 'LLegBack2', 'LLegBack3','LFoot']

    asis = [name for name in name2id.keys() if name not in right + left]

    if model is not None:
        mv = MeshViewer()
        model.pose[:] = pose
        model.trans[1] = 0.80
        orig_r = model.r.copy()
        model.trans[1] = 0.

    # Swap left and right.
    new_pose = pose.copy()
    for rname, lname in zip(right, left):
        rind = name2id[rname] * 3 + np.arange(0, 3)
        lind = name2id[lname] * 3 + np.arange(0, 3)
        tmp = new_pose[rind]
        new_pose[rind] = new_pose[lind]
        new_pose[lind] = tmp
        # I think x & z comp needs to flip sign
        new_pose[[rind[0], rind[2]]] *= -1
        new_pose[[lind[0], lind[2]]] *= -1
    for name in asis:
        inds = name2id[name] * 3 + np.arange(0, 3)
        # I think x & z comp needs to flip sign
        new_pose[[inds[0], inds[2]]] *= -1
    if model is not None:
        # Visualize
        model.pose[:] = new_pose
        mv.set_dynamic_meshes([Mesh(model.r, model.f), Mesh(orig_r, model.f,vc=name_to_rgb['steel blue'])])
        import ipdb; ipdb.set_trace()

    return new_pose

