import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from .provider import *
import os
from glob import glob
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


def load_dtu(data_dir):
    print('Load data: Begin')
    data_dir = data_dir
    render_cameras_name = 'cameras_sphere.npz'
    camera_dict = np.load(os.path.join(data_dir, render_cameras_name))
    camera_dict = camera_dict
    images_lis = sorted(glob(os.path.join(data_dir, 'image/*.png')))
    n_images = len(images_lis)
    images_np = np.stack([cv.imread(im_name) for im_name in images_lis]) / 255.0
    masks_lis = sorted(glob(os.path.join(data_dir, 'mask/*.png')))
    masks_np = np.stack([cv.imread(im_name) for im_name in masks_lis]) / 255.0
    # world_mat is a projection matrix from world to image
    world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    scale_mats_np = []
    # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
    scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats_np, world_mats_np):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        intrinsics, pose = load_K_Rt_from_P(None, P)
        intrinsics_all.append(intrinsics)
        pose_all.append(pose)
    H, W = images_np.shape[1], images_np.shape[2]
    poses = np.stack(pose_all, axis=0)
    images_np = images_np[..., :3][..., ::-1]
    return images_np, masks_np, poses, intrinsics_all[0], H, W, images_lis


class DTUDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10, normalize=True, optimize_camera=False, max_data_num=np.inf, **kwargs):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.mode = opt.mode # colmap, blender
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.
        self.normalize = normalize
        self.optimize_camera = optimize_camera
        self.max_data_num = max_data_num
        self.error_map = None

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        self.images, self.masks, self.poses, intrinsics, self.H, self.W, self.f_paths = load_dtu(opt.path)
        self.images = np.concatenate([self.images, self.masks[..., :1]], axis=-1)

        self.poses = np.stack(self.poses, axis=0)
        if self.normalize:
            self.poses, _ = normalize_cps(self.poses, scale=1.2)
        # check_poses(poses)
        self.poses = self.poses[:min(self.max_data_num, len(self.poses))]
        self.images = self.images[:min(self.max_data_num, len(self.images))]
        self.f_paths = self.f_paths[:min(self.max_data_num, len(self.f_paths))]

        if 'plane_transform' in kwargs.keys() and kwargs['plane_transform'] is not None:
            plane_transform = kwargs['plane_transform']
            self.poses = np.einsum('ab,nbc->nac', plane_transform, self.poses)

        self.poses = torch.from_numpy(self.poses)
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]

        print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        if self.fp16 and self.opt.color_space != 'linear':
            dtype = torch.half
        else:
            dtype = torch.float
        self.images = self.images.to(dtype)
        if self.preload:
            self.poses = self.poses.to(self.device)
            self.images = self.images.to(self.device)

        self.intrinsics = np.array([intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]])
        self.intrinsics_tensor = torch.from_numpy(self.intrinsics).cuda().float()


    def collate(self, index):

        B = len(index) # always 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]
        
        rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map)
        
        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']
        
        return results


    def collate_trainable_camera(self, index):

        B = len(index) # always 1

        poses = self.poses[index].to(self.device) # [B, 4, 4]
        error_map = None if self.error_map is None else self.error_map[index]
        
        def get_results_func(dRs, dts, dfs):
            R = poses[..., :3, :3]
            t = poses[..., :3, 3]
            K = self.intrinsics_tensor.clone()
            dR = pytorch3d.transforms.axis_angle_to_matrix(dRs[index])
            R = torch.matmul(dR, R)
            t = dts[index] + t
            K[:2] = dfs[index].squeeze() + K[:2]
            poses_new = torch.zeros_like(poses)
            poses_new[..., :3, :3] = R
            poses_new[..., :3, 3] = t
            rays = get_rays(poses_new, K, self.H, self.W, self.num_rays, error_map)
            results = {
                'H': self.H,
                'W': self.W,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'index': index,
            }
            if self.images is not None:
                images = self.images[index].to(self.device) # [B, H, W, 3/4]
                if self.training:
                    C = images.shape[-1]
                    images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
                results['images'] = images
        
            # need inds to update error_map
            if error_map is not None:
                results['index'] = index
                results['inds_coarse'] = rays['inds_coarse']
            return results
        
        return get_results_func

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        collate_fn_ = self.collate if not self.optimize_camera else self.collate_trainable_camera
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=collate_fn_, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        return loader

    @property
    def length(self):
        return self.poses.shape[0]

    def nn_image(self, pose, W, H):
        poses = self.poses.cpu().numpy()
        centers = poses[..., :3, 3]
        center = pose[:3, 3]
        idx = np.linalg.norm(centers - center, axis=-1).argmin()
        image = self.images[idx].cpu().numpy()
        if image.shape[-1] == 4:
            image = image[..., :3] * image[..., 3:]
        image = cv2.resize(np.float32(image), (W, H))
        return image
