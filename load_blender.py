import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split

def get_render_poses(basedir, time, cam):
    with open(os.path.join(basedir, 'transforms_all.json'), 'r') as fp:
        metas = json.load(fp)
    
    assert time.shape == cam.shape, 'time and cam should be same shape'

    poses = []
    all_poses = []
    obj_poses = []
    all_obj_poses = []
    for t, c in zip(time, cam):
        for f in metas['frames']:
            if f['camera_idx'] == c and f['frame_idx'] == t:
                poses.append(np.array(f['transform_matrix']))
                T = np.array(f['object_transformation_matrix'])
                T[:3,:3] *= 10.0
                obj_poses.append(np.linalg.inv(T))

    poses = np.array(poses).astype(np.float32)
    all_poses.append(poses)
    poses = np.concatenate(all_poses, 0)
    render_poses = torch.tensor(poses)

    obj_poses = np.array(obj_poses).astype(np.float32)
    all_obj_poses.append(obj_poses)
    obj_poses = np.concatenate(all_obj_poses, 0)
    render_obj_pose = torch.tensor(obj_poses)

    # print(render_poses)
    # print(render_obj_pose)

    return render_poses, render_obj_pose


def load_blender_data_star(basedir, half_res=False, testskip=1, time=None, cam=None, r_a_t=True):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_obj_poses = []
    counts = [0]

    render_obj_pose = None

    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        obj_poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            if s == 'test':
                print(fname)
            T = np.array(frame['object_transformation_matrix'])
            T[:3,:3] *= 10.0
            if r_a_t and render_obj_pose is None:
                render_obj_pose = np.linalg.inv(T)
                render_obj_pose = render_obj_pose.astype(np.float32)
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
            obj_poses.append(np.linalg.inv(T))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        obj_poses = np.array(obj_poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_obj_poses.append(obj_poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    obj_poses = np.concatenate(all_obj_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if r_a_t:
        render_obj_pose = torch.tensor(render_obj_pose)
        render_obj_pose = render_obj_pose.repeat(render_poses.shape[0],1,1)
    else:
        render_poses, render_obj_pose = get_render_poses(basedir, time=time, cam=cam)

    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, obj_poses, render_poses, render_obj_pose, [H, W, focal], i_split


# def load_blender_data_star(basedir, half_res=False, testskip=1, time=1, cam_number=1, r_a_t=True):
#     splits = ['train', 'val', 'test']
#     metas = {}
#     for s in splits:
#         with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
#             metas[s] = json.load(fp)

#     all_imgs = []
#     all_poses = []
#     all_obj_poses = []
#     counts = [0]

#     render_obj_pose = None

#     for s in splits:
#         meta = metas[s]
#         imgs = []
#         poses = []
#         obj_poses = []
#         if s=='train' or testskip==0:
#             skip = 1
#         else:
#             skip = testskip
            
#         for frame in meta['frames'][::skip]:
#             fname = os.path.join(basedir, frame['file_path'] + '.png')
#             if s=='test':
#                 print(fname)
#             T = np.array(frame['object_transformation_matrix'])
#             T[:3,:3] *= 10.0
#             if render_obj_pose is None and frame['file_path'].endswith('t_' + str(time)) and r_a_t:
#                 render_obj_pose = np.linalg.inv(T)
#                 render_obj_pose = render_obj_pose.astype(np.float32)
#             imgs.append(imageio.imread(fname))
#             poses.append(np.array(frame['transform_matrix']))
#             obj_poses.append(np.linalg.inv(T))
#         imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
#         poses = np.array(poses).astype(np.float32)
#         obj_poses = np.array(obj_poses).astype(np.float32)
#         counts.append(counts[-1] + imgs.shape[0])
#         all_imgs.append(imgs)
#         all_poses.append(poses)
#         all_obj_poses.append(obj_poses)
    
#     i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
#     imgs = np.concatenate(all_imgs, 0)
#     poses = np.concatenate(all_poses, 0)
#     obj_poses = np.concatenate(all_obj_poses, 0)
    
#     H, W = imgs[0].shape[:2]
#     camera_angle_x = float(meta['camera_angle_x'])
#     focal = .5 * W / np.tan(.5 * camera_angle_x)
    
#     render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
#     if r_a_t:
#         print('here in rat')
#         render_obj_pose = torch.tensor(render_obj_pose)
#         render_obj_pose = render_obj_pose.repeat(render_poses.shape[0],1,1)
    
#     # print(render_poses)
#     # print(render_obj_pose)

#     if half_res:
#         H = H//2
#         W = W//2
#         focal = focal/2.

#         imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
#         for i, img in enumerate(imgs):
#             imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
#         imgs = imgs_half_res
#         # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
#     return imgs, poses, obj_poses, render_poses, render_obj_pose, [H, W, focal], i_split


