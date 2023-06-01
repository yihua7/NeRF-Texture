import os
import cv2
import glob
import json
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation
import pytorch3d
import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays, srgb_to_linear


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    sphere = trimesh.creation.icosphere(radius=1)
    objects = [axes, sphere]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


def check_poses(poses, z_val=0.01, check_path='./check_poses/'):
    points = np.array([[0., 0., 0.],
                       [-1., -1., 1.],
                       [-1., 1., 1.],
                       [1., -1., 1.],
                       [1., 1., 1.]])
    faces = np.array([[0, 1, 2],
                      [0, 3, 1],
                      [0, 4, 3],
                      [0, 2, 4],
                      [1, 3, 2],
                      [2, 3, 4]])
    points *= z_val
    new_points = np.einsum('na,mba->mnb', np.concatenate([points, np.ones_like(points[..., :1])], axis=-1), poses)
    new_points = new_points[..., :-1].reshape([-1, 3])
    new_faces = np.stack([faces + points.shape[0] * i for i in range(poses.shape[0])], axis=0).reshape([-1, 3])
    colors = np.linspace(0, 255, poses.shape[0], dtype=np.int32)
    colors = np.broadcast_to(np.stack([colors] * 3, axis=-1)[:, np.newaxis], [colors.shape[0], points.shape[0], 3]).reshape([-1, 3])

    if not os.path.exists(check_path):
        os.makedirs(check_path)
    
    str_v = [f"v {new_points[i][0]} {new_points[i][1]} {new_points[i][2]} {colors[i][0]} {colors[i][0]} {colors[i][0]}\n" for i in range(new_points.shape[0])]
    str_f = [f"f {new_faces[i][0]+1} {new_faces[i][1]+1} {new_faces[i][2]+1}\n" for i in range(new_faces.shape[0])]
    with open(check_path + '/poses.obj', 'w') as file:
        file.write(f'{"".join(str_v)}{"".join(str_f)}')


def normalize_cps(cps, scale=1.2):
    cps = centralize_cps(cps)
    dists = np.linalg.norm(cps[:, :3, 3], axis=-1)
    radius = 1.1 * np.max(dists) + 1e-5
    # Corresponding parameters change
    cps[:, :3, 3] /= radius / scale
    return cps, radius


def centralize_cps(cps):
    cps = np.array(cps, dtype=np.float32)
    avg_center = min_line_dist_center(cps[:, :3, 3], cps[:, :3, 2])
    cps[:, :3, 3] -= avg_center
    return cps


def min_line_dist_center(rays_o, rays_d):
    if len(np.shape(rays_d)) == 2:
        rays_o = rays_o[..., np.newaxis]
        rays_d = rays_d[..., np.newaxis]
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
    b_i = -A_i @ rays_o
    pt_mindist = np.squeeze(-np.linalg.inv((A_i @ A_i).mean(0)) @ (b_i).mean(0))
    return pt_mindist


class NeRFDataset:
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

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)
        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        
        # for colmap, manually interpolate a test set.
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)

        else:
            # for colmap, manually split a valid set (the first frame).
            if self.mode == 'colmap':
                if type == 'train':
                    frames = frames[1:]
                elif type == 'val':
                    frames = frames[:1]
                # else 'all' or 'trainval' : use all frames
            
            self.poses = []
            self.images = []
            self.f_paths = []
            for f in tqdm.tqdm(frames, desc=f'Loading {type} data:'):
                f_path = os.path.join(self.root_path, f['file_path'])
                if self.mode == 'blender' or (f_path[-4:] != '.png' and f_path[-4:] != '.jpg' and f_path[-4:] != '.jpeg'):
                    f_path += '.png' # so silly...

                # there are non-exist paths in fox...
                if not os.path.exists(f_path):
                    continue
                
                pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
                pose = nerf_matrix_to_ngp(pose, scale=self.scale)

                image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
                if self.H is None or self.W is None:
                    self.H = image.shape[0] // downscale
                    self.W = image.shape[1] // downscale

                # add support for the alpha channel as a mask.
                if image.shape[-1] == 3: 
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

                if image.shape[0] != self.H or image.shape[1] != self.W:
                    image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)
                    
                image = image.astype(np.float32) / 255 # [H, W, 3/4]

                self.poses.append(pose)
                self.images.append(image)
                self.f_paths.append(f_path)
        
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

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        if self.images is not None:
            # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
            if self.fp16 and self.opt.color_space != 'linear':
                dtype = torch.half
            else:
                dtype = torch.float
        self.images = self.images.to(dtype)
        if self.preload:
            self.poses = self.poses.to(self.device)
            self.images = self.images.to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device).to(dtype)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.H / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.W / 2)
    
        self.intrinsics = np.array([fl_x, fl_y, cx, cy])
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
