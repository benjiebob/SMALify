import pickle as pkl
import numpy as np
from chumpy import Ch
# from body.matlab import row
import cv2
# from psbody.mesh.colors import name_to_rgb



name2id33 = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1, 'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15, 'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11, 'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27, 'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22, 'root': 0}
id2name33 = {v: k for k, v in name2id33.items()}

name2id31 = {'RFoot': 12, 'RFootBack': 22, 'LLegBack1': 15, 'spine1': 2, 'Head': 14, 'RLegBack1': 19, 'RLegBack2': 20, 'RLegBack3': 21, 'LLegBack2': 16, 'LLegBack3': 17, 'spine3': 4, 'spine2': 3, 'Mouth': 30, 'Neck': 13, 'LFootBack': 18, 'LLeg1': 5, 'RLeg2': 10, 'RLeg3': 11, 'LLeg3': 7, 'RLeg1': 9, 'LLeg2': 6, 'spine': 1, 'LFoot': 8, 'Tail7': 29, 'Tail6': 28, 'Tail5': 27, 'Tail4': 26, 'Tail3': 25, 'Tail2': 24, 'Tail1': 23, 'root': 0}
id2name31 = {v: k for k, v in name2id31.items()}

name2id35 = {'RFoot': 14, 'RFootBack': 24, 'spine1': 4, 'Head': 16, 'LLegBack3': 19, 'RLegBack1': 21, 'pelvis0': 1, 'RLegBack3': 23, 'LLegBack2': 18, 'spine0': 3, 'spine3': 6, 'spine2': 5, 'Mouth': 32, 'Neck': 15, 'LFootBack': 20, 'LLegBack1': 17, 'RLeg3': 13, 'RLeg2': 12, 'LLeg1': 7, 'LLeg3': 9, 'RLeg1': 11, 'LLeg2': 8, 'spine': 2, 'LFoot': 10, 'Tail7': 31, 'Tail6': 30, 'Tail5': 29, 'Tail4': 28, 'Tail3': 27, 'Tail2': 26, 'Tail1': 25, 'RLegBack2': 22, 'root': 0, 'LEar':33, 'REar':34}
id2name35 = {v: k for k, v in name2id35.items()}

def get_ignore_names(path):
    if 'notail' in path:
        ignore_names = [key for key in name2id.keys() if 'Tail' in key]
    elif 'bodyneckelbowtail' in path:
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack']
    elif 'body_indept_limbstips' in path:
        ignore_names = ['Neck', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']

    elif 'prior_bodyelbow' in path:
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'RFoot', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']

    elif 'bodyneckheadelbow' in path:
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']
    elif 'bodyneckelbow' in path:
        ignore_names = ['LLeg2', 'LLeg3', 'RLeg2', 'RLeg3', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']
    elif 'bodyneck' in path:
        ignore_names = ['LLeg1', 'LLeg2', 'LLeg3', 'RLeg1', 'RLeg2', 'RLeg3', 'Neck', 'LLegBack2', 'LLegBack3', 'RLegBack2', 'RLegBack3', 'RFootBack']
    elif 'backlegstail' in path:
        if '33parts' in path:
            ignore_names = ['root', 'RFoot', 'RFootBack', 'spine1', 'Head', 'pelvis0', 'spine0', 'spine3', 'spine2', 'Mouth', 'Neck', 'LFootBack', 'RLeg3', 'RLeg2', 'LLeg1', 'LLeg3', 'RLeg1', 'LLeg2', 'spine', 'LFoot']
        if '35parts' in path:
            ignore_names = ['root', 'RFoot', 'RFootBack', 'spine1', 'Head', 'pelvis0', 'spine0', 'spine3', 'spine2', 'Mouth', 'Neck', 'LFootBack', 'RLeg3', 'RLeg2', 'LLeg1', 'LLeg3', 'RLeg1', 'LLeg2', 'spine', 'LFoot', 'LEar', 'REar']
    elif '_body.pkl' in path:
        ignore_names = ['LLeg1', 'LLeg2', 'LLeg3', 'RLeg1', 'RLeg2', 'RLeg3', 'RFoot', 'Neck', 'LLegBack1', 'LLegBack2', 'LLegBack3', 'RLegBack1', 'RLegBack2', 'RLegBack3', 'RFootBack', 'Tail1', 'Tail2', 'Tail3', 'Tail4', 'Tail5', 'Tail6', 'Tail7']
    else:
        ignore_names = []


    return ignore_names

import torch
class Prior(object):
    def __init__(self, prior_path, device):
        with open(prior_path, "rb") as f:
            res = pkl.load(f, encoding='latin1')
            
        self.mean_ch = res['mean_pose']
        self.precs_ch = res['pic']

        self.precs = torch.from_numpy(res['pic'].r.copy()).float().to(device)
        self.mean = torch.from_numpy(res['mean_pose']).float().to(device)

        # Mouth closed!
        # self.mean[-2] = -0.4
        # Ignore the first 3 global rotation.
        prefix = 3
        if '33parts' in prior_path:
            pose_len = 99
            id2name = id2name33
            name2id = name2id33
        elif '35parts' in prior_path:
            pose_len = 105
            id2name = id2name35
            name2id = name2id35
        else:
            pose_len = 93
            id2name = id2name31
            name2id = name2id31

        self.use_ind = np.ones(pose_len, dtype=bool)
        self.use_ind[:prefix] = False

        self.use_ind_tch = torch.from_numpy(self.use_ind).float().to(device)

        ignore_names = get_ignore_names(prior_path)

        if len(ignore_names) > 0:
            ignore_joints = sorted([name2id[name] for name in ignore_names])
            #ignore_inds = np.hstack([3 + np.array(ig * 3) + np.arange(0, 3)
            #                         for ig in ignore_joints])
            # Silvia: why we do not start from 0?
            ignore_inds = np.hstack([np.array(ig * 3) + np.arange(0, 3)
                                     for ig in ignore_joints])
            self.use_ind[ignore_inds] = False

            # Test to make sure that the ignore indices are ok.
            '''
            test = use_ind.reshape((-1, 3))
            # Make sure all rows are the same
            assert(np.all(test[:, 0] == test[:, 1]) and np.all(test[:, 0] == test[:, 1]))
            if pose_len == 99:
                id2name = id2name33
            elif pose_len == 105:
                id2name = id2name35
            else:
                id2name = id2name31
            print('Using parts:')
            print([id2name[id] for id in (test[:, 0] == True).nonzero()[0]])
            print('Ignoring parts:')
            print([id2name[id] for id in (test[:, 0] == False).nonzero()[0]])
            import ipdb; ipdb.set_trace()
            '''

    def __call__(self, x):
        # return row(x[self.prefix:] - self.mean).dot(self.precs)
        # return (x[self.use_ind] - self.mean).dot(self.precs)
        # res = (x[self.use_ind] - self.mean).dot(self.precs)

        mean_sub = x.reshape(-1, 35*3) - self.mean.unsqueeze(0)
        res = torch.tensordot(mean_sub, self.precs, dims = ([1], [0])) * self.use_ind_tch
        # res_ch = (x.data.cpu().numpy() - self.mean_ch).dot(self.precs_ch) * self.use_ind

        # error = np.linalg.norm(res.data.cpu().numpy() - res_ch)
        # assert error < 1e-2, "ERROR VERY LARGE"

        return res ** 2

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
def model31_to_35(pose):
    new_pose = np.zeros((105))
    new_pose[:3] = pose[:3]
    new_pose[3:6] = pose[:3]
    new_pose[6:9] = pose[3:6]
    new_pose[9:12] = pose[3:6]
    new_pose[12:99] = pose[6:]
    return new_pose


def reflect_pose(pose, name2id, model=None, is35=False):
    # Parts that has to be swapped:
    if is35:
        right = ['RLeg1', 'RLeg2', 'RLeg3', 'RFootBack', 'RLegBack1', 'RLegBack2', 'RLegBack3', 'RFoot', 'LEar', 'REar']
        left =  ['LLeg1', 'LLeg2', 'LLeg3', 'LFootBack', 'LLegBack1', 'LLegBack2', 'LLegBack3','LFoot', 'LEar', 'REar']
    else:
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

if __name__ == '__main__':
    ''' Create pose prior from the walking sequence '''
    from os.path import join, exists
    data_dir = '../../data/for_pose_prior/' #'/scratch1/projects/MPI_data/smpl_animal_models/'

    seq_path_walking = join(data_dir, 'walking_sequence.pkl')
    seq_path_silvia = join(data_dir, 'lioness_more_torso_parts_new_feet_new_back_legs_animation_mirrored_poses_prior.pkl') #'lioness_jaw_tail_straight_eyes_filled_animation_mirrored_poses.pkl')
    #path_toy = '/scratch1/Dropbox/research/animal_proj/smpl_models/smpl_align_results_all_mix_from_56.pkl'
    #path_toy = join(data_dir, 'smpl_align_results_all_mix_from_56.pkl')
    #path_toy = join(data_dir, 'smpl_align_results_all_0073nm_im.pkl')

    seq_paths = [seq_path_walking, path_toy]
    #seq_paths = [seq_path_toys]
    data_save_dir = join(data_dir, 'pose_prior')

    FIX_TAIL = False #True
    SYMMETRIC = True

    prefix = []
    for seq in seq_paths:
        if seq == seq_path_walking:
            if FIX_TAIL:
                prefix += ['walking-meantail']
            else:
                prefix += ['walking']
        elif seq == seq_path_silvia:
            prefix += ['silvia']
        elif seq == path_toy:
            prefix += ['toy']

    prefix = '_'.join(prefix)

    if SYMMETRIC:
        prefix += '_symmetric'

    # The postfix of this pathnames decides which body parts to use.:
    # Gaussian over the entire parts:
    out_path = join(data_save_dir, prefix + '_pose_prior.pkl')
    # Gaussian over body neck elbow:
    # out_path = join(data_save_dir, prefix + '_pose_prior_bodyneckelbow.pkl')
    # Gaussian over body neck:
    # out_path = join(data_save_dir, prefix + '_pose_prior_bodyneck.pkl')
    # Silvia: Gaussian over back legs, pelvis and tail
    out_path = join(data_save_dir, prefix + '_pose_prior_backlegstail.pkl')

    # Load model
    from psbody.mesh import Mesh
    from psbody.mesh.meshviewer import MeshViewer, MeshViewers
    #model_dir = '/scratch1/Dropbox/research/animal_proj/smpl_models'
    model_dir = '../../smpl_models'
    #model_path = join(model_dir, 'my_smpl_0001_4_all.pkl')
    #model_path = join(model_dir, 'my_smpl_0073nm_im_4_all.pkl')
    model_path = join(model_dir, 'my_smpl_00781_4_all.pkl')
    from psbody.smpl.serialization import load_model
    model = load_model(model_path)


    is33parts = False
    is35parts = True
    if model.pose.shape[0] == 99:
        # Pelvis [0] and Spine [1] is now doubled.
        prefix = prefix + '_33parts'
        out_path = out_path.replace('.pkl', '_33parts.pkl')
        is33parts = True
        id2name = id2name33
        name2id = name2id33
    elif model.pose.shape[0] == 105:
        prefix = prefix + '_35parts'
        out_path = out_path.replace('.pkl', '_35parts.pkl')
        is35parts = True
        id2name = id2name35
        name2id = name2id35
    else:
        id2name = id2name31
        name2id = name2id31


    if FIX_TAIL:
        # Use the tail from here.
        #mean_pose = np.load('/scratch1/projects/MPI_data/smpl_animal_models/pose_prior/toy_mean_pose.npz')['mean_pose']
        mean_pose = np.load('../../data/for_pose_prior/pose_prior/toy_symmetric_35parts_mean_pose.npz')['mean_pose']
        tail_joints = sorted([name2id[name] for name in name2id.keys() if 'Tail' in name])
        tail_inds = np.hstack([np.array(ig * 3) + np.arange(0, 3)
                               for ig in tail_joints])


    print('Prior: %s, is33parts %d' % (out_path, is33parts))

    pose_samples = []
    for seq_path in seq_paths:
        if seq_path == seq_path_walking:
            with open(seq_path) as f:
                data = pkl.load(f)
                poses = data['poses']
            if is33parts:
                poses = np.array([model31_to_33(pose) for pose in poses])
            if is35parts:
                poses = np.array([model31_to_35(pose) for pose in poses])
            if FIX_TAIL:
                poses[:, tail_inds] = mean_pose[tail_inds]
        elif seq_path == seq_path_silvia:
            poses = load_silvia_data(seq_path, model)
            if is33parts:
                poses = model31_to_33(poses)
            if is35parts:
                poses = model31_to_35(poses)
        elif seq_path == path_toy:
            with open(seq_path) as f: data = pkl.load(f)
            poses = data['P']
            # This is already 33 parts.
            #assert(is33parts)
        pose_samples.append(poses)

    pose_samples = np.vstack(pose_samples)

    if SYMMETRIC:
        pose_samples_sym = [reflect_pose(pose, name2id, is35=is35parts) for pose in pose_samples]
        pose_samples = np.vstack((pose_samples, pose_samples_sym))
        '''
        mv = MeshViewers(shape=(2,2))
        for pose, sym_pose in zip(pose_samples[::-1], pose_samples_sym[::-1]):
            model.pose[:] = pose
            M1 = Mesh(model.r, model.f, vc=name_to_rgb['steel blue'])
            v = model.r
            M1.v[:,0] = v[:,0].copy()
            M1.v[:,1] = v[:,2].copy()
            M1.v[:,2] = -v[:,1].copy()
            M2 = Mesh(model.r, model.f, vc=name_to_rgb['steel blue'])
            M2.v[:,0] = v[:,1].copy()
            M2.v[:,1] = v[:,2].copy()
            M2.v[:,2] = v[:,0].copy()
            mv[0][0].set_static_meshes([M1])
            mv[0][1].set_static_meshes([M2])

            model.pose[:] = sym_pose
            model.trans[1] = 2.
            v = model.r
            mesh_sym = Mesh(v, model.f)
            mesh_sym.v[:,0] = v[:,0].copy()
            mesh_sym.v[:,1] = v[:,2].copy()
            mesh_sym.v[:,2] = -v[:,1].copy()
            mesh_sym2 = Mesh(v, model.f)
            mesh_sym2.v[:,0] = v[:,1].copy()
            mesh_sym2.v[:,1] = v[:,2].copy()
            mesh_sym2.v[:,2] = v[:,0].copy()
            mv[1][0].set_static_meshes([mesh_sym])
            mv[1][1].set_static_meshes([mesh_sym2])
            import ipdb; ipdb.set_trace()
        '''

    # Visualize the samples:
    # mv = MeshViewer()
    # for pose in pose_samples]:
    #     model.pose[:] = pose
    #     mv.set_dynamic_meshes([Mesh(model.r, model.f)])
    #     import ipdb; ipdb.set_trace()

    mean_pose_path = join(data_save_dir, prefix + '_mean_pose.npz')

    if not exists(mean_pose_path):
        mean_pose = np.mean(pose_samples, axis=0)
        model.pose[:] = mean_pose
        mv = MeshViewer()
        mv.set_dynamic_meshes([Mesh(model.r, model.f)])
        #import ipdb; ipdb.set_trace()
        np.savez(mean_pose_path, mean_pose=mean_pose)
    # else:
    #     mean_pose = np.load(mean_pose_path)['mean_pose']
    #     import ipdb; ipdb.set_trace()


    # Ignore the first 3-dims bc they're global rotation
    prefix = 3
    use_ind = np.ones(pose_samples.shape[1], dtype=bool)
    use_ind[:prefix] = False

    ignore_names = get_ignore_names(out_path)

    if len(ignore_names) > 0:
        print('Ignoring parts:')
        print(ignore_names)
        ignore_joints = sorted([name2id[name] for name in ignore_names])
        # Silvia: why do we start from 3?
        ignore_inds = np.hstack([
            np.array(ig * 3) + np.arange(0, 3) for ig in ignore_joints])
        #ignore_inds = np.hstack([
        #    3 + np.array(ig * 3) + np.arange(0, 3) for ig in ignore_joints])
        use_ind[ignore_inds] = False

    # Silvia: be sure we do not use the root (global rot)
    use_ind[:prefix] = False

    if not exists(out_path):
        # print out_path
        print('Computing pose prior..')
        # try:
        #     pose_invcov = np.linalg.inv(np.cov(pose_samples[:, use_ind].T))
        #     pic = Ch(np.linalg.cholesky(pose_invcov))
        # except:
            # print 'error in pose prior'
        # This will always be true with walking data:
        if 'body_indept_limbstips' in out_path:
            # Compute the entire covariance matrix.
            pose_cov = np.cov(pose_samples.T) + 0.00001 * np.eye(pose_samples.shape[1])
            # Make indept joints block diagonal
            indept_joints = [10, 11, 6, 7, 20, 21, 16, 17]
            for ind in indept_joints:
                istart = 3 + 3*ind
                iend = 3 + 3*ind + 3
                pose_cov[:istart, istart:iend] = 0.
                pose_cov[iend:, istart:iend] = 0.
                pose_cov[istart:iend, :istart] = 0.
                pose_cov[istart:iend, iend:] = 0.
            # import ipdb; ipdb.set_trace()
            # import matplotlib.pyplot as plt
            # plt.ion()
            # plt.matshow(pose_cov)
            pose_invcov = np.linalg.inv(pose_cov)
            # Now get rid of the ignore joints:
            pose_invcov = pose_invcov[use_ind, :]
            pose_invcov = pose_invcov[:, use_ind]
            pic = Ch(np.linalg.cholesky(pose_invcov))
        else:
            pose_invcov = np.linalg.inv(
                np.cov(pose_samples[:, use_ind].T) + 0.00001 * np.eye(
                    pose_samples[:, use_ind].shape[1]))
            pic = Ch(np.linalg.cholesky(pose_invcov))

        mean_pose = np.mean(pose_samples[:, use_ind], axis=0)

        with open(out_path, 'w') as f:
            dic = {'pic': pic, 'mean_pose': mean_pose}
            pkl.dump(dic, f)
    else:
        with open(out_path) as f:
            res = pkl.load(f)
            pic = res['pic']
            mean_pose = res['mean_pose']

    mv = MeshViewer()
    model.pose[:] = pose_samples[0, :]
    mv.set_dynamic_meshes([Mesh(model.r, model.f)])

    pose_prior = Prior(out_path)
    noprior_ind = ~pose_prior.use_ind
    # Keep global
    noprior_ind[:3] = False
    # Try sampling to double check.
    cov = np.linalg.inv(pose_prior.precs.dot(pose_prior.precs.T).r)
    import matplotlib.pyplot as plt
    plt.ion()
    from psbody.mesh.colors import name_to_rgb

    for i in range(pose_samples.shape[0]):
        #model.pose[:] = pose_samples[i]

        model.pose[:] = np.zeros(model.pose.shape)
        rand_pose = np.random.multivariate_normal(pose_prior.mean, cov)
        model.pose[use_ind] = rand_pose
        #model.pose[:3] = np.zeros(3)

        #P = pose_prior(model.pose.r)
        mv.set_dynamic_meshes([Mesh(model.r, model.f)])

        # Random values in tail does not change prior:
        #model.pose[:] = pose_samples[i, :]

        '''
        old_r = model.r
        model.pose[noprior_ind] = np.random.randn(np.sum(noprior_ind)) * 0.4
        P0 = pose_prior(model.pose.r)
        mv.set_dynamic_meshes([Mesh(model.r, model.f), Mesh(old_r, model.f, vc=name_to_rgb['steel blue'])])
        print('with random values in no-prior zone: %g, orig %g' % (np.sum(np.abs(P0)), np.sum(np.abs(P0))))
        # model.pose[3+3*23:3+3*29] = np.random.randn(len(model.pose[3+3*23:3+3*29]))
        # mv.set_dynamic_meshes([Mesh(model.r, model.f)])
        # P = pose_prior(model.pose.r)
        assert(np.sum(np.abs(P0)) == np.sum(np.abs(P)))

        print np.sum(np.abs(P))
        '''
        import ipdb
        ipdb.set_trace()

        #for j in range(33):
        #    model.pose[:] = pose_samples[i, :]
        #    #model.pose[3+3*j] = np.pi/2
        #    mv.set_dynamic_meshes([Mesh(model.r, model.f)])
        #    mv.set_titlebar('part %d' % j)
        #    import ipdb; ipdb.set_trace()
