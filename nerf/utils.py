import os
import glob
import tqdm
import random
import tensorboardX
import numpy as np
import time
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import trimesh
import mcubes
from rich.console import Console
from torch_ema import ExponentialMovingAverage
from packaging import version as pver
from plyfile import PlyElement, PlyData
from functools import partial
from scipy.spatial.transform import Rotation as R
from PIL import Image
import pytorch3d
from tools.shape_tools import write_ply
import imageio


def RT2Pose(R=np.eye(3), T=np.zeros([3])):
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = T
    return pose


def surrounding_plane_poses(frame_num=40, radius=1.5, round=5, around_y=False, fix_phi=False, fix_theta=False, up_y=False):
    """Create Camera Poses"""
    epsilon = np.pi / 2 / 10
    if not fix_theta:
        thetas = np.linspace(epsilon, np.pi / 2 - epsilon, frame_num) if not fix_phi else np.linspace(0, np.pi / 2 - epsilon, frame_num)
        phis = np.linspace(- np.pi, (2 * round - 1) * np.pi, frame_num) if not fix_phi else np.zeros([frame_num]) # - np.pi / 2 * np.ones([frame_num])
    else:
        thetas = np.pi / 2 * np.ones([frame_num]) #- np.pi / 6
        phis = np.linspace(- np.pi, (2 * round - 1) * np.pi, frame_num)
    # Translation
    xs = np.cos(phis) * np.sin(thetas)
    ys = np.sin(phis) * np.sin(thetas)
    zs = np.cos(thetas)
    ts = radius * np.stack([xs, ys, zs], axis=-1)
    # Rotation
    zaxis = - ts.copy()
    xaxis = np.cross(np.array([0, 0, -1]), zaxis)
    mask = np.linalg.norm(xaxis, axis=-1) == 0
    xaxis[mask] = np.array([0, 1, 0], dtype=np.float32)
    yaxis = np.cross(zaxis, xaxis)
    xaxis = xaxis / (np.linalg.norm(xaxis, axis=-1, keepdims=True) + 1e-10)
    yaxis = yaxis / (np.linalg.norm(yaxis, axis=-1, keepdims=True) + 1e-10)
    zaxis = zaxis / (np.linalg.norm(zaxis, axis=-1, keepdims=True) + 1e-10)
    rs = np.stack([xaxis, yaxis, zaxis], axis=2)
    # Poses
    poses = np.broadcast_to(np.eye(4, dtype=np.float32)[np.newaxis], [frame_num, 4, 4]).copy()
    poses[:, :3, :3] = rs
    poses[:, :3, 3] = ts
    # Around y axis
    if around_y:
        rotation = np.array([[0., 1., 0., 0.],
                             [0., 0., 1., 0.],
                             [1., 0., 0., 0.],
                             [0., 0., 0., 1.]])
        poses = np.einsum('ab,nbc->nac', rotation, poses)
    elif up_y:
        rotation = np.array([[0., 1., 0., 0.],
                             [-1., 0., 0., 0.],
                             [0., 0., 1., 0.],
                             [0., 0., 0., 1.]])
        poses = np.einsum('ab,nbc->nac', rotation, poses)
    return poses

def convert_pose(C2W):
    # iPad data is the same as instant-ngp.
    # however, the data loader in torch-ngp does a convertion.
    flip_yz = np.eye(4, dtype=C2W.dtype)
    # flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W


def surrounding360_poses(num=40, radius=1.5):
    angle = np.linspace(0, np.pi * 2, num=num, endpoint=False)
    axis = np.array([0, 1, 0], dtype=np.float32)
    rot_vec = axis[np.newaxis] * angle[..., np.newaxis]
    rs = R.from_rotvec(rotvec=rot_vec)
    rs = rs.as_matrix()
    target_camera = np.array([0, 0, -radius])
    target_world = np.einsum('nab,b->na', rs, target_camera)
    poses = np.eye(4, dtype=np.float32)
    poses = np.stack([poses] * num, axis=0)
    poses[..., :3, 3] = - target_world
    poses[..., :3, :3] = rs
    for i in range(poses.shape[0]):
        poses[i] = convert_pose(poses[i])
    # check_poses(poses, z_val=0.1, check_path='test_data/')
    return poses

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


@torch.jit.script
def linear_to_srgb(x):
    return torch.where(x < 0.0031308, 12.92 * x, 1.055 * x ** 0.41666 - 0.055)


@torch.jit.script
def srgb_to_linear(x):
    return torch.where(x < 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


@torch.cuda.amp.autocast(enabled=False)
def get_rays_scale_resolution(poses, intrinsics, H, W, scale=1, return_cosine=False):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, scale: int
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''
    W_, H_ = int(W * scale), int(H * scale)
    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics
    i, j = custom_meshgrid(torch.linspace(0, W-1, W_, device=device), torch.linspace(0, H-1, H_, device=device))
    i = i.t().reshape([1, H_*W_]).expand([B, H_*W_]) + 0.5
    j = j.t().reshape([1, H_*W_]).expand([B, H_*W_]) + 0.5
    results = {}
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)
    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]
    results['rays_o'] = rays_o
    results['rays_d'] = rays_d
    results['H'] = H_
    results['W'] = W_

    if return_cosine:
        cam_dir = poses[0, :3, 2]
        rays_d_ = rays_d / rays_d.norm(dim=-1, keepdim=True)
        cosine = (rays_d_ * cam_dir).sum(dim=-1)
        results['cosine'] = cosine
    
    return results


@torch.cuda.amp.autocast(enabled=False)
def get_rays(poses, intrinsics, H, W, N=-1, error_map=None, return_cosine=False):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = custom_meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device))
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]

    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    if return_cosine:
        cam_dir = poses[0, :3, 2]
        rays_d_ = rays_d / rays_d.norm(dim=-1, keepdim=True)
        cosine = (rays_d_ * cam_dir).sum(dim=-1)
        results['cosine'] = cosine

    return results


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True


def torch_vis_2d(x, renormalize=False):
    # x: [3, H, W] or [1, H, W] or [H, W]
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 3:
            x = x.permute(1,2,0).squeeze()
        x = x.detach().cpu().numpy()
        
    print(f'[torch_vis_2d] {x.shape}, {x.dtype}, {x.min()} ~ {x.max()}')
    
    x = x.astype(np.float32)
    
    # renormalize
    if renormalize:
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True) + 1e-8)

    plt.imshow(x)
    plt.show()


def extract_fields(bound_min, bound_max, resolution, query_func, S=128):

    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(S)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(S)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(S)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [S, 3]
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy() # [S, 1] --> [x, y, z]
                    u[xi * S: xi * S + len(xs), yi * S: yi * S + len(ys), zi * S: zi * S + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    #print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)

    #print(u.shape, u.max(), u.min(), np.percentile(u, 50))
    
    vertices, triangles = mcubes.marching_cubes(u, threshold)

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def remove_isolated_piecies(mesh, min_face_num=100):
    cc = trimesh.graph.connected_components(mesh.face_adjacency, min_len=min_face_num)
    mask = np.zeros(len(mesh.faces), dtype=bool)
    if len(cc) > 0:
        mask[np.concatenate(cc)] = True
    mesh.update_faces(mask)
    mesh.remove_unreferenced_vertices()
    return mesh


class PSNRMeter:
    def __init__(self):
        self.V = 0
        self.N = 0

    def clear(self):
        self.V = 0
        self.N = 0

    def prepare_inputs(self, *inputs):
        outputs = []
        for i, inp in enumerate(inputs):
            if torch.is_tensor(inp):
                inp = inp.detach().cpu().numpy()
            outputs.append(inp)

        return outputs

    def update(self, preds, truths):
        preds, truths = self.prepare_inputs(preds, truths) # [B, N, 3] or [B, H, W, 3], range[0, 1]
          
        # simplified since max_pixel_value is 1 here.
        psnr = -10 * np.log10(np.mean((preds - truths) ** 2))
        
        self.V += psnr
        self.N += 1

    def measure(self):
        return self.V / self.N

    def write(self, writer, global_step, prefix=""):
        writer.add_scalar(os.path.join(prefix, "PSNR"), self.measure(), global_step)

    def report(self):
        return f'PSNR = {self.measure():.6f}'


def farthest_sampling(poses, initial=0, K=8):
    zs = poses[:, :3, 2]
    zs = zs / np.linalg.norm(zs, axis=-1, keepdims=True)
    dists = np.einsum('na,ma->nm', zs, zs)
    dists = -dists  # cosine distance: larger->closer
    min_dist = dists[initial]
    selected = [initial]
    for _ in range(1, K):
        idx = np.argmax(min_dist)
        min_dist = np.minimum(min_dist, dists[idx])
        selected.append(idx)
    return np.stack(selected)        


def write_ply_rgb(points, RGB, filename):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as PLY file """
    N = points.shape[0]
    vertex = []
    for i in range(N):
        vertex.append((points[i, 0], points[i, 1], points[i, 2], RGB[i][0], RGB[i][1], RGB[i][2]))
    vertex = np.array(vertex,
                    dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(filename)


class Trainer(object):
    def __init__(self, 
                 name, # name of this experiment
                 opt, # extra conf
                 model, # network 
                 criterion=None, # loss function, if None, assume inline implementation in train_step
                 optimizer=None, # optimizer
                 ema_decay=None, # if use EMA, set the decay
                 lr_scheduler=None, # scheduler
                 metrics=[], # metrics for evaluation, if None, use val_loss to measure performance, else use the first metric.
                 local_rank=0, # which GPU am I
                 world_size=1, # total num of GPUs
                 device=None, # device to use, usually setting to None is OK. (auto choose device)
                 mute=False, # whether to mute all print
                 fp16=False, # amp optimize level
                 eval_interval=1, # eval once every $ epoch
                 max_keep_ckpt=2, # max num of saved ckpts in disk
                 workspace='workspace', # workspace to save logs & ckpts
                 best_mode='min', # the smaller/larger result, the better
                 use_loss_as_metric=True, # use loss as the first metric
                 report_metric_at_train=False, # also report metrics at training
                 use_checkpoint="latest", # which ckpt to use at init time
                 use_tensorboardX=True, # whether to use tensorboard for logging
                 scheduler_update_every_step=False, # whether to call scheduler.step() after every train step
                 ckpt_folder=None,
                 update_gridfield=True,
                 num_iterations_per_stage=500,
                 num_iterations_diffuse=6000,
                 ):
        self.error_map = None
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        self.console = Console()
        self.ckpt_folder = 'checkpoints' if ckpt_folder is None else ckpt_folder
        self.update_gridfield=update_gridfield
        self.num_iterations_per_stage = num_iterations_per_stage
        self.num_iterations_diffuse = num_iterations_diffuse
        self.teacher_model = None
        self.distillation = False
        self.field_dict_cache = None
        self.euler = np.zeros([3], dtype=np.float32)

        model.to(self.device)
        if self.world_size > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        self.model = model

        if isinstance(criterion, nn.Module):
            criterion.to(self.device)
        self.criterion = criterion

        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=5e-4) # naive adam
        else:
            self.optimizer = optimizer(self.model)
        self.optimizer_func = partial(optim.Adam, lr=5e-3, weight_decay=5e-4) if optimizer is None else optimizer

        if lr_scheduler is None:
            self.lr_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1) # fake scheduler
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)
        self.lr_scheduler_func = partial(optim.lr_scheduler.LambdaLR, lr_lambda=lambda epoch: 1) if lr_scheduler is None else lr_scheduler

        if ema_decay is not None:
            parameters = [p for p in list(self.model.parameters()) if p.dtype != torch.long]
            self.ema = ExponentialMovingAverage(parameters, decay=ema_decay)
        else:
            self.ema = None
        self.ema_decay = ema_decay

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)

        # variable init
        self.epoch = 1
        self.global_step = 0
        self.local_step = 0
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "results": [], # metrics[0], or valid_loss
            "checkpoints": [], # record path of saved ckpt, to automatically remove old ckpt
            "best_result": None,
            }

        # auto fix
        if len(metrics) == 0 or self.use_loss_as_metric:
            self.best_mode = 'min'

        # workspace prepare
        self.log_ptr = None
        if self.workspace is not None:
            os.makedirs(self.workspace, exist_ok=True)        
            self.log_path = os.path.join(workspace, f"log_{self.name}.txt")
            self.log_ptr = open(self.log_path, "a+")

            self.ckpt_path = os.path.join(self.workspace, self.ckpt_folder, self.model.field_name)
            self.best_path = f"{self.ckpt_path}/{self.name}.pth"
            os.makedirs(self.ckpt_path, exist_ok=True)
            
        self.log(f'[INFO] Trainer: {self.name} | {self.time_stamp} | {self.device} | {"fp16" if self.fp16 else "fp32"} | {self.workspace}')
        self.log(f'[INFO] #parameters: {sum([p.numel() for p in model.parameters() if p.requires_grad])}')

        if self.workspace is not None:
            if self.use_checkpoint == "scratch":
                self.log("[INFO] Training from scratch ...")
            elif self.use_checkpoint == "latest":
                self.log("[INFO] Loading latest checkpoint ...")
                self.load_checkpoint()
            elif self.use_checkpoint == "latest_model":
                self.log("[INFO] Loading latest checkpoint (model only)...")
                self.load_checkpoint(model_only=True)
            elif self.use_checkpoint == "best":
                if os.path.exists(self.best_path):
                    self.log("[INFO] Loading best checkpoint ...")
                    self.load_checkpoint(self.best_path)
                else:
                    self.log(f"[INFO] {self.best_path} not found, loading latest ...")
                    self.load_checkpoint()
            else: # path to ckpt
                self.log(f"[INFO] Loading {self.use_checkpoint} ...")
                self.load_checkpoint(self.use_checkpoint)
        

    def __del__(self):
        if self.log_ptr: 
            self.log_ptr.close()


    def log(self, *args, **kwargs):
        if self.local_rank == 0:
            if not self.mute: 
                #print(*args)
                self.console.print(*args, **kwargs)
            if self.log_ptr: 
                print(*args, file=self.log_ptr)
                self.log_ptr.flush() # write immediately to file


    def load_teacher_model(self):
        if self.teacher_model is None:
            from nerf.network_tcnn import NeRFNetwork as nerf
            opt = self.opt
            model = nerf(
                encoding="hashgrid",
                bound=opt.bound,
                cuda_ray=opt.cuda_ray,
                density_scale=1,
                min_near=opt.min_near,
                density_thresh=opt.density_thresh,
                bg_radius=opt.bg_radius,
                cal_dist_loss=False,
            )
            model.to(self.device)
            ckpt_path = os.path.join(self.workspace, self.ckpt_folder, model.field_name)
            checkpoint_list = sorted(glob.glob(f'{ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
            else:
                print('No teacher check point found. Model: ', model.field_name)
                return False
            checkpoint_dict = torch.load(checkpoint, map_location=self.device)
            if 'model' not in checkpoint_dict:
                model.load_state_dict(checkpoint_dict)
                self.log("[INFO] loaded model.")
            else:
                model.load_state_dict(checkpoint_dict['model'], strict=False)
            model.eval()
            self.teacher_model = model
            print('Loaded Teacher Checkpoints From: ', checkpoint)
        else:
            print('Already load teacher')
        return True

    ### ------------------------------	

    def train_step(self, data):

        if self.model.optimize_camera:
            # get rays from learnable camera parameters
            data = self.model.get_results(data)

        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        index = data['index'] if 'index' in data.keys() else None

        if self.distillation:
            if self.teacher_model is None:
                if not self.load_teacher_model():
                    print('No Check Points for Teacher Model Found. Back to no distillation')
                    self.distillation = False
            self.teacher_model.eval()
        
        distillation_prob = .75
        rand_x = np.random.rand()
        if self.distillation and rand_x < distillation_prob:
            xyzs, dirs, _, _, prefix = self.model.sample(rays_o, rays_d, **vars(self.opt))
            sigmas, rgbs, _ = self.model(xyzs, dirs, frame_index=index)
            sigmas_t, rgbs_t, _ = self.teacher_model(xyzs, dirs)
            lambda_ = 1.
            sigmas_remap = 1 / lambda_ * (1 - torch.exp(-lambda_ * sigmas))
            sigmas_remap_t = 1 / lambda_ * (1 - torch.exp(-lambda_ * sigmas_t))
            loss = ((rgbs - rgbs_t.detach()) ** 2).mean() + ((sigmas_remap - sigmas_remap_t.detach()) ** 2).mean()
            pred_rgb = rgbs[0]
            gt_rgb = rgbs_t[0]
        else:
            images = data['images'] # [B, N, 3/4]
            B, N, C = images.shape
            if self.opt.color_space == 'linear':
                images[..., :3] = srgb_to_linear(images[..., :3])
            if C == 3 and self.model.bg_radius > 0:
                bg_color = 1
            # train with random background color if not using a bg model and has alpha channel.
            else:
                bg_color = torch.rand_like(images[..., :3]) # [N, 3], pixel-wise random.
            if C == 4:
                gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
            else:
                gt_rgb = images
            outputs = self.model.render(rays_o, rays_d, staged=False, bg_color=bg_color, perturb=True, force_all_rays=False, index=index, **vars(self.opt))
            pred_rgb = outputs['image']
            rgb_loss = self.criterion(pred_rgb, gt_rgb).mean(-1) # [B, N, 3] --> [B, N]
            # special case for CCNeRF's rank-residual training
            if len(rgb_loss.shape) == 3: # [K, B, N]
                rgb_loss = rgb_loss.mean(0)
            # update error_map
            if self.error_map is not None:
                index = data['index'] # [B]
                inds = data['inds_coarse'] # [B, N]
                # take out, this is an advanced indexing and the copy is unavoidable.
                error_map = self.error_map[index] # [B, H * W]
                error = rgb_loss.detach().to(error_map.device) # [B, N], already in [0, 1]
                # ema update
                ema_error = 0.1 * error_map.gather(1, inds) + 0.9 * error
                error_map.scatter_(1, inds, ema_error)
                # put back
                self.error_map[index] = error_map
            # sparse_loss = self.model.random_density(dtype=rgb_loss.dtype, device=rgb_loss.device).mean()
            distortion_loss = outputs['distortion_loss']
            if distortion_loss is None:
                distortion_loss = 0.
            if self.model.regularization:
                regular_loss = self.model.regular_loss(step=self.global_step)
            else:
                regular_loss = 0.
            if 'gamma_loss' in outputs.keys() and outputs['gamma_loss'] is not None:
                gamma_loss = outputs['gamma_loss']
            else:
                gamma_loss = 0.
            
            cosine_threshold = np.cos(np.pi/8) if not(hasattr(self.model, 'no_visibility') and self.model.no_visibility) else 1.
            use_cosine_loss = True

            normal_error = 0.
            if 'normal_grad' in outputs.keys() and ('normal_est' in outputs.keys() or 'normal' in outputs.keys()):
                normal_grad = outputs['normal_grad'].detach()[0]
                normal_est = outputs['normal_est'][0] if 'normal_est' in outputs.keys() else outputs['normal'][0]
                nan_mask = torch.logical_not(normal_grad.isnan().any(dim=-1))
                if use_cosine_loss:
                    normal_grad = normal_grad[nan_mask] /(normal_grad[nan_mask].norm(dim=-1, keepdim=True) + 1e-5)
                    normal_est = normal_est[nan_mask] / (normal_est[nan_mask].norm(dim=-1, keepdim=True) + 1e-5)
                    normal_error = - torch.minimum((normal_grad * normal_est).sum(dim=-1), cosine_threshold*torch.ones_like(normal_grad[..., 0])).mean()
                else:
                    normal_error = ((normal_grad[nan_mask] - normal_est[nan_mask]) ** 2).mean()
                if 'normal_coarse' in outputs.keys():
                    normal_error = normal_error + 1e-4 * ((outputs['normal_coarse'] - normal_est) ** 2).mean()
                # normal_error = 1e1 * normal_error
            if 'normal_error' in outputs.keys() and outputs['normal_error'] is not None:
                nan_mask = torch.logical_not(outputs['normal_error'][0].isnan().any(dim=-1))
                normal_error = 1e-1 * outputs['normal_error'][0][nan_mask].mean() + normal_error
            loss = rgb_loss.mean() + 1e-2 * distortion_loss + regular_loss + normal_error + gamma_loss

        return pred_rgb, gt_rgb, loss

    def eval_step(self, data):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]
        B, H, W, C = images.shape
        if self.opt.color_space == 'linear':
            images[..., :3] = srgb_to_linear(images[..., :3])
        # eval with fixed background color
        bg_color = 1
        if C == 4:
            gt_rgb = images[..., :3] * images[..., 3:] + bg_color * (1 - images[..., 3:])
        else:
            gt_rgb = images
        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(B, H, W, 3)
        pred_depth = outputs['depth'].reshape(B, H, W)
        loss = self.criterion(pred_rgb, gt_rgb).mean()
        return pred_rgb, pred_depth, gt_rgb, loss

    # moved out bg_color and perturb for more flexible control...
    def test_step(self, data, bg_color=None, perturb=False, fine_render=False, simple_render=False):  
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        H, W = data['H'], data['W']
        if bg_color is not None:
            bg_color = bg_color.to(self.device)
        force_staged = False  # (self.model.light_model == 'Envmap')
        if fine_render:
            opt = dict(**vars(self.opt))
            opt['max_steps'] = 4096
            outputs = self.model.render(rays_o, rays_d, staged=True, force_staged=force_staged, bg_color=bg_color, perturb=perturb, euler=self.euler, is_gui_mode=True, **opt)
        elif simple_render:
            opt = dict(**vars(self.opt))
            opt['max_steps'] = 128
            outputs = self.model.render(rays_o, rays_d, staged=True, force_staged=force_staged, bg_color=bg_color, perturb=perturb, euler=self.euler, is_gui_mode=True, **opt)
        else:
            outputs = self.model.render(rays_o, rays_d, staged=True, force_staged=force_staged, bg_color=bg_color, perturb=perturb, euler=self.euler, is_gui_mode=True, **vars(self.opt))
        pred_rgb = outputs['image'].reshape(-1, H, W, 3)
        pred_depth = outputs['depth'].reshape(-1, H, W)
        out_dict = {}
        out_dict['mask'] = outputs['mask'].reshape(-1, H, W)
        return pred_rgb, pred_depth, out_dict

    def save_mesh(self, save_path=None, resolution=256, threshold=10):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', f'{self.name}_{self.epoch}.obj')
        self.log(f"==> Saving mesh to {save_path}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        def query_func(pts):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    sigma = self.model.density(pts.to(self.device))['sigma']
            return sigma
        vertices, triangles = extract_geometry(self.model.aabb_infer[:3], self.model.aabb_infer[3:], resolution=resolution, threshold=threshold, query_func=query_func)
        mesh = trimesh.Trimesh(vertices, triangles, process=False) # important, process=True leads to seg fault...
        factor = 5
        clean_mesh = remove_isolated_piecies(mesh, min_face_num=int(mesh.faces.shape[0] / factor))
        if clean_mesh.vertices.shape[0] == 0:
            factor *= 2
            clean_mesh = remove_isolated_piecies(mesh, min_face_num=int(mesh.faces.shape[0] / factor))
        clean_mesh.export(save_path)
        self.log(f"==> Finished saving mesh.")
        return save_path

    def take_photo(self, pose, intrinsics, W, H, bg_color):
        from datetime import datetime
        sv_path = self.workspace + '/photos/' + self.model.field_name + '/'
        if not os.path.exists(sv_path):
            os.makedirs(sv_path)
        
        if os.path.exists(sv_path + '/poses/target.npz'):
            data = np.load(sv_path + '/poses/target.npz')
            pose, intrinsics, W, H = data['pose'][0], data['intrinsics'], int(data['W']), int(data['H'])

        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        rays = get_rays(pose, intrinsics, H, W, -1, return_cosine=True)
        
        self.model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                outputs = self.model.render(rays['rays_o'], rays['rays_d'], bg_color=bg_color, perturb=False, euler=self.euler, is_gui_mode=True, max_steps=4096)
                pred_rgb = outputs['image'].reshape(-1, H, W, 3)
                pred_depth = outputs['depth'].reshape(-1, H, W)
                out_dict = {}
                out_dict['mask'] = outputs['mask'].reshape(-1, H, W)
        mask_ = out_dict['mask'].detach().cpu().numpy()[0]
        mask = np.array(mask_ * 255, dtype=np.uint8)
        rgb = pred_rgb[0].detach().cpu().numpy()
        rgb = np.clip(rgb, 0, 1)
        rgb = np.array(rgb * 255, dtype=np.uint8)
        msked_rgb = np.array(rgb * (mask[..., None] / 255) + (255 - mask[..., None]), dtype=np.uint8)
        depth = pred_depth[0].detach().cpu().numpy() * rays['cosine'].cpu().numpy().reshape(pred_depth[0].shape)
        depth_mean = (depth * mask).sum() / mask.sum()
        depth = np.where(mask, depth, depth_mean * np.ones_like(depth))
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)
        depth = np.where(mask, depth, np.zeros_like(depth))
        depth = np.array(depth * 255, dtype=np.uint8)

        mask = cv2.resize(mask, (W, H))
        rgb = cv2.resize(rgb, (W, H))
        depth = cv2.resize(depth, (W, H))

        if not os.path.exists(sv_path + '/views'):
            os.makedirs(sv_path + '/views')
        if not os.path.exists(sv_path + '/msked_views'):
            os.makedirs(sv_path + '/msked_views')
        if not os.path.exists(sv_path + '/masks'):
            os.makedirs(sv_path + '/masks')
        if not os.path.exists(sv_path + '/depthes'):
            os.makedirs(sv_path + '/depthes')
        if not os.path.exists(sv_path + '/poses'):
            os.makedirs(sv_path + '/poses')
        now = datetime.now()
        f_name = now.strftime("%m%d%Y_%H:%M:%S") + '.png'
        Image.fromarray(rgb).save(sv_path + '/views/' + f_name)
        Image.fromarray(msked_rgb).save(sv_path + '/msked_views/' + f_name)
        Image.fromarray(mask).save(sv_path + '/masks/' + f_name)
        Image.fromarray(depth).save(sv_path + '/depthes/' + f_name)
        np.savez(sv_path + '/poses/' + f_name, pose=pose.cpu().numpy(), intrinsics=intrinsics, W=W, H=H)
        
        return sv_path

    def render_train(self, bg_color):
        is_training = self.model.training
        self.model.eval()

        H, W = self.train_loader._data.H, self.train_loader._data.W
        poses = self.train_loader._data.poses.clone().to(self.device)
        K = self.train_loader._data.intrinsics_tensor.clone().to(self.device)
        if self.model.optimize_camera:
            R = poses[..., :3, :3]
            t = poses[..., :3, 3]
            dR = pytorch3d.transforms.axis_angle_to_matrix(self.model.dRs)
            R = torch.matmul(dR, R)
            t = self.model.dts + t
            new_K = torch.zeros([poses.shape[0], *K.shape]).float().to(self.device)
            new_K[..., :] = K
            new_K[..., :2] = self.model.dfs.squeeze() + new_K[..., :2]
            poses_new = torch.zeros_like(poses)
            poses_new[..., :3, :3] = R
            poses_new[..., :3, 3] = t
        else:
            poses_new = poses
            new_K = K.unsqueeze(0).expand([poses_new.shape[0], *K.shape])
        
        sv_path = self.workspace + '/render_train/' + self.model.field_name + '/'
        if not os.path.exists(sv_path):
            os.makedirs(sv_path)
        for i in tqdm.tqdm(range(poses_new.shape[0])):
            f_name = self.train_loader._data.f_paths[i].split('/')[-1]
            rays = get_rays_scale_resolution(poses_new[i:i+1], new_K[i], self.train_loader._data.H, self.train_loader._data.W, scale=1, return_cosine=True)
            self.model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.model.render(rays['rays_o'], rays['rays_d'], bg_color=bg_color, perturb=False, euler=None, is_gui_mode=True, max_steps=4096)
                    pred_rgb = outputs['image'].reshape(-1, rays['H'], rays['W'], 3)
                    pred_depth = outputs['depth'].reshape(-1, rays['H'], rays['W'])
                    out_dict = {}
                    out_dict['mask'] = outputs['mask'].reshape(-1, rays['H'], rays['W'])
            
            mask_ = out_dict['mask'].detach().cpu().numpy()[0]
            mask = np.array(mask_ * 255, dtype=np.uint8)
            rgb = pred_rgb[0].detach().cpu().numpy()
            rgb = np.clip(rgb, 0, 1)
            rgb = np.array(rgb * 255, dtype=np.uint8)
            depth = pred_depth[0].detach().cpu().numpy() * rays['cosine'].detach().cpu().numpy().reshape(pred_depth[0].shape)
            depth_mean = (depth * mask).sum() / mask.sum()
            depth = np.where(mask, depth, depth_mean * np.ones_like(depth))
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)
            depth = np.where(mask, depth, np.zeros_like(depth))
            depth = np.array(depth * 255, dtype=np.uint8)

            mask = cv2.resize(mask, (W, H))
            rgb = cv2.resize(rgb, (W, H))
            depth = cv2.resize(depth, (W, H))

            if not os.path.exists(sv_path + '/views'):
                os.makedirs(sv_path + '/views')
            if not os.path.exists(sv_path + '/masks'):
                os.makedirs(sv_path + '/masks')
            if not os.path.exists(sv_path + '/depthes'):
                os.makedirs(sv_path + '/depthes')
            Image.fromarray(rgb).save(sv_path + '/views/' + f_name)
            Image.fromarray(mask).save(sv_path + '/masks/' + f_name)
            Image.fromarray(depth).save(sv_path + '/depthes/' + f_name)
        
        if is_training:
            self.model.train()
        return sv_path

    def render_round(self, intrinsics, W, H, bg_color=None, render_light=False, fix_phi=False, fix_theta=False):
        # Rendering around y axis in shape mode
        is_shape_rendering = (hasattr(self.model, 'meshfea_field') and self.model.meshfea_field.imported_type == 'shape')
        around_y=(is_shape_rendering or not self.model.meshfea_field.imported)
        radius = 2  # 2.5 if is_shape_rendering else 1
        poses = surrounding_plane_poses(frame_num=480, radius=radius, round=.5, around_y=around_y, fix_phi=fix_phi, fix_theta=fix_theta, up_y=not around_y)
        
        if render_light:
            poses_light = poses[:int(poses.shape[0] / 4)].copy()
            poses_light[:] = poses[0]
            poses_light_num = poses_light.shape[0]
            euler_idx = 1 if is_shape_rendering else 0
            euler_range = np.linspace(0, 2 * np.pi, poses_light_num)
            poses = np.concatenate([poses_light, poses], axis=0)
        else:
            poses_light_num = 0
        
        sv_path = self.workspace + '/render_round'
        if render_light:
            sv_path += '_light'
        if fix_phi:
            sv_path += '_0phi'
        if fix_theta:
            sv_path += '_0theta'
        sv_path += '/'

        if not os.path.exists(sv_path):
            os.makedirs(sv_path)
        video = cv2.VideoWriter(sv_path + '/views.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
        video_d = cv2.VideoWriter(sv_path + '/depthes.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H), False)
        for i in tqdm.tqdm(range(poses.shape[0])):
            pose = poses[i]
            pose = torch.from_numpy(pose).float().unsqueeze(0).to(self.device)
            rays = get_rays(pose, intrinsics, H, W, -1, return_cosine=True)
            self.model.eval()

            if i < poses_light_num:
                self.euler[euler_idx] = euler_range[i]
            else:
                self.euler[:] = 0.
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.model.render(rays['rays_o'], rays['rays_d'], bg_color=bg_color, perturb=False, euler=self.euler, is_gui_mode=True, max_steps=4096)
                    pred_rgb = outputs['image'].reshape(-1, H, W, 3)
                    pred_depth = outputs['depth'].reshape(-1, H, W)
                    out_dict = {}
                    out_dict['mask'] = outputs['mask'].reshape(-1, H, W)
            
            mask_ = out_dict['mask'].detach().cpu().numpy()[0]
            mask = np.array(mask_ * 255, dtype=np.uint8)
            rgb = pred_rgb[0].detach().cpu().numpy()
            rgb = np.clip(rgb, 0, 1)
            rgb = np.array(rgb * 255, dtype=np.uint8)
            depth = pred_depth[0].detach().cpu().numpy() * rays['cosine'].cpu().numpy().reshape(pred_depth[0].shape)
            depth_mean = (depth * mask).sum() / mask.sum()
            depth = np.where(mask, depth, depth_mean * np.ones_like(depth))
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-5)
            depth = np.where(mask, depth, np.zeros_like(depth))
            depth = np.array(depth * 255, dtype=np.uint8)
            if not os.path.exists(sv_path + '/views'):
                os.makedirs(sv_path + '/views')
            if not os.path.exists(sv_path + '/masks'):
                os.makedirs(sv_path + '/masks')
            if not os.path.exists(sv_path + '/depthes'):
                os.makedirs(sv_path + '/depthes')
            Image.fromarray(rgb).save(sv_path + '/views/%03d.png' % i)
            Image.fromarray(mask).save(sv_path + '/masks/%03d.png' % i)
            Image.fromarray(depth).save(sv_path + '/depthes/%03d.png' % i)
            video.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            video_d.write(depth)
        video.release()
        video_d.release()
        
        from nerf.provider import check_poses
        check_poses(poses=poses, z_val=.1, check_path=sv_path + '/')

        return sv_path

    def train(self, train_loader, valid_loader, max_epochs):
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(os.path.join(self.workspace, "run", self.name))
        # mark untrained region (i.e., not covered by any camera from the training dataset)
        if self.model.cuda_ray:
            self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)
        # get a ref to error_map
        self.error_map = train_loader._data.error_map
        for epoch in range(self.epoch, max_epochs + 1):
            self.epoch = epoch
            self.train_one_epoch(train_loader)
            if self.workspace is not None and self.local_rank == 0:
                self.save_checkpoint(full=True, best=False)
            if self.epoch % self.eval_interval == 0:
                self.evaluate_one_epoch(valid_loader)
                self.save_checkpoint(full=False, best=True)
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer.close()

    def evaluate(self, loader, name=None):
        self.use_tensorboardX, use_tensorboardX = False, self.use_tensorboardX
        self.evaluate_one_epoch(loader, name)
        self.use_tensorboardX = use_tensorboardX

    def test(self, loader, save_path=None, name=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'results')
        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'
        os.makedirs(save_path, exist_ok=True)
        self.log(f"==> Start Test, save results to {save_path}")
        pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.model.eval()
        with torch.no_grad():
            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
            for i, data in enumerate(loader):
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, _ = self.test_step(data)                
                path = os.path.join(save_path, f'{name}_{i:04d}.png')
                path_depth = os.path.join(save_path, f'{name}_{i:04d}_depth.png')
                #self.log(f"[INFO] saving test image to {path}")
                if self.opt.color_space == 'linear':
                    preds = linear_to_srgb(preds)
                pred = preds[0].detach().cpu().numpy()
                pred_depth = preds_depth[0].detach().cpu().numpy()
                cv2.imwrite(path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                cv2.imwrite(path_depth, (pred_depth * 255).astype(np.uint8))
                pbar.update(loader.batch_size)
        self.log(f"==> Finished Test.")
    
    def train_gui(self, train_loader, step=16):
        # update grid
        if self.model.cuda_ray:
            with torch.cuda.amp.autocast(enabled=self.fp16):
                self.model.update_extra_state()
        self.model.train()
        total_loss = torch.tensor([0], dtype=torch.float32, device=self.device)
        loader = iter(train_loader)
        for _ in range(step):
            # mimic an infinite loop dataloader (in case the total dataset is smaller than step)
            try:
                data = next(loader)
            except StopIteration:
                loader = iter(train_loader)
                data = next(loader)
            # mark untrained grid
            if self.global_step == 0:
                self.model.mark_untrained_grid(train_loader._data.poses, train_loader._data.intrinsics)
                self.error_map = train_loader._data.error_map
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % 16 == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()   
            if self.update_gridfield and hasattr(self.model, 'update_gridfield'):
                if self.model.update_gridfield(target_stage=int(self.global_step // self.num_iterations_per_stage)):
                    self.update_optimizer_scheduler()
            self.global_step += 1
            self.optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler_update_every_step:
                self.lr_scheduler.step()
            total_loss += loss.detach()
        if self.ema is not None:
            self.ema.update()
        average_loss = total_loss.item() / step
        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()
        outputs = {
            'loss': average_loss,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        
        return outputs

    def test_gui(self, pose, intrinsics, W, H, bg_color=None, spp=1, downscale=1, simple_render=False): 
        # render resolution (may need downscale to for better frame rate)
        rH = int(H * downscale)
        rW = int(W * downscale)
        intrinsics = intrinsics * downscale
        pose = torch.from_numpy(pose).unsqueeze(0).to(self.device)
        rays = get_rays(pose, intrinsics, rH, rW, -1)
        data = {
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
            'H': rH,
            'W': rW,
        }
        self.model.eval()
        # if self.ema is not None:
        #     self.ema.store()
        #     self.ema.copy_to()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                # here spp is used as perturb random seed!
                preds, preds_depth, _ = self.test_step(data, bg_color=bg_color, perturb=spp, simple_render=simple_render)
        # if self.ema is not None:
        #     self.ema.restore()
        # interpolation to the original resolution
        if downscale != 1:
            # TODO: have to permute twice with torch...
            preds = F.interpolate(preds.permute(0, 3, 1, 2), size=(H, W), mode='nearest').permute(0, 2, 3, 1).contiguous()
            preds_depth = F.interpolate(preds_depth.unsqueeze(1), size=(H, W), mode='nearest').squeeze(1)
        if self.opt.color_space == 'linear':
            preds = linear_to_srgb(preds)
        pred = np.clip(preds[0].detach().cpu().numpy(), 0, 1)
        pred_depth = preds_depth[0].detach().cpu().numpy()
        outputs = {
            'image': pred,
            'depth': pred_depth,
        }
        return outputs
    
    def save_poses(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.log(f"==> Saving point cloud to {save_path}")
        train_loader = self.train_loader._data
        poses = train_loader.poses.cpu().numpy()
        from nerf.provider import check_poses
        radius = np.linalg.norm(poses[..., :3, 3], axis=-1).max()
        check_poses(poses=poses, z_val=0.05 * radius, check_path=save_path)
        return save_path + '/poses.obj'

    def save_point_cloud(self, save_path=None, spp=1, downscale=None):
        import open3d as o3d
        if save_path is None:
            save_path = os.path.join(self.workspace, 'meshes', 'pcl.ply')
        if not os.path.exists('/'.join(save_path.split('/')[:-1])):
            os.makedirs('/'.join(save_path.split('/')[:-1]))
        self.log(f"==> Saving point cloud to {save_path}")
        train_loader = self.train_loader._data
        poses = train_loader.poses.cpu().numpy()
        pick_ids = farthest_sampling(poses, initial=0, K=16)
        poses = poses[pick_ids]
        gt_rgba = train_loader.images[pick_ids].cpu().numpy()
        # render resolution (may need downscale to for better frame rate)
        if downscale is None:
            max_h = 256
            downscale = 1. if train_loader.H < max_h else (max_h / train_loader.H)
        rH = int(train_loader.H * downscale)
        rW = int(train_loader.W * downscale)
        intrinsics = train_loader.intrinsics * downscale
        rgbs, coors = [], []
        for i in range(poses.shape[0]):
            pose = torch.from_numpy(poses[i]).unsqueeze(0).to(self.device)
            rays = get_rays(pose, intrinsics, rH, rW, -1)

            data = {
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],
                'H': rH,
                'W': rW,
            }
            self.model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    # here spp is used as perturb random seed!
                    rgb, preds_depth, out_dict = self.test_step(data, perturb=spp)
            if self.opt.color_space == 'linear':
                rgb = linear_to_srgb(rgb)
            mask = out_dict['mask'][0].reshape([-1]).detach().cpu().numpy()
            rgb = rgb[0].reshape([-1, 3]).detach().cpu().numpy()
            coor = (rays['rays_o'] + preds_depth.reshape([1, -1, 1]) * rays['rays_d']).reshape([-1, 3]).detach().cpu().numpy()
            nan_mask = np.logical_not(np.isnan(coor).any(-1))
            coor = coor[nan_mask]
            rgb = rgb[nan_mask]
            mask = out_dict['mask'][0].reshape([-1]).detach().cpu().numpy()[nan_mask]
            coor = coor[mask]
            rgb = rgb[mask]
            rgbs.append(rgb)
            coors.append(coor)
        rgbs = np.concatenate(rgbs, axis=0)
        coors = np.concatenate(coors, axis=0)
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(coors)
        pcl.colors = o3d.utility.Vector3dVector(rgbs.astype(np.float32))
        pcl.voxel_down_sample(voxel_size=1e-3)
        o3d.io.write_point_cloud(save_path, pcl)
        self.log(f"==> Finished saving point cloud.")
        return save_path
    
    def render_rays(self, rays, chunk=128**2):
        prefix = rays.shape[:-1]
        rays = rays.reshape([-1, 6])
        rgbd = np.zeros_like(rays[..., :4])
        start = 0
        self.model.eval()
        while start < rays.shape[0]:
            end = min(start + chunk, rays.shape[0])
            rays_o = torch.from_numpy(rays[start: end, :3]).float().to(self.device)
            rays_d = torch.from_numpy(rays[start: end, 3:]).float().to(self.device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    outputs = self.model.render(rays_o, rays_d, staged=False, **vars(self.opt))
                    if self.opt.color_space == 'linear':
                        outputs['image'] = linear_to_srgb(outputs['image'])
                    rgbd[start: end, :3] = outputs['image'].detach().cpu().numpy().reshape([-1, 3])
                    rgbd[start: end, 3] = outputs['depth'].detach().cpu().numpy().reshape([-1])
            start = end
        rgbd = rgbd.reshape([*prefix, 4])
        return rgbd
    
    def save_field(self, save_path=None, record_rgb=True):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'field', self.model.field_name)
        if not os.path.exists(os.path.join(self.workspace, 'field')):
            os.makedirs(os.path.join(self.workspace, 'field'))
        scan_pcl_path = os.path.join(self.workspace, 'meshes', 'scan_pcl.ply')
        if not os.path.exists(scan_pcl_path):
            self.save_point_cloud(save_path=scan_pcl_path)
        picked_faces_path = os.path.join(self.workspace, 'meshes', 'picked_faces.obj')
        field_dict = self.model.export_field(scan_pcl_path=scan_pcl_path, picked_faces_path=picked_faces_path, record_rgb=record_rgb, work_space=self.workspace)
        if 'picked_vertices' in field_dict.keys():
            write_ply(field_dict['picked_vertices'], os.path.join(self.workspace, 'field', 'picked_vertices.ply'))
        if record_rgb:
            print('Rendering patch rgb values...')
            patch_rays = field_dict['patch_rays']
            rgb_path = save_path + '/'
            if not os.path.exists(rgb_path):
                os.makedirs(rgb_path)
            for i in tqdm.tqdm(range(patch_rays.shape[0])):
                rgbd = self.render_rays(patch_rays[i])
                rgb = np.moveaxis(np.array(rgbd[..., :3] * 255, dtype=np.float32), 0, 1)
                rgb = np.clip(rgb, 0, 255)
                rgb = np.array(rgb, dtype=np.uint8)
                imageio.imwrite(rgb_path + '/%05d.png' % i, rgb)
        del field_dict['patch_rays']
        np.savez(save_path, **field_dict)
        print('Save_field Done!')
        return save_path
    
    def load_field(self, load_path=None):
        if load_path is None:
            load_path = os.path.join(self.workspace, 'field', 'texture.npz')
            if not os.path.exists(load_path):
                return load_path + ' file not found.'
        field_dict = np.load(load_path, allow_pickle=True)
        self.model.import_field(field_dict, fp16=self.fp16)
        self.update_optimizer_scheduler()
        return load_path
    
    def load_patch(self, load_path=None):
        if not hasattr(self, 'field_dict_cache') or self.field_dict_cache is None:
            if load_path is None:
                load_path = os.path.join(self.workspace, 'field', self.model.field_name+'.npz')
                if not os.path.exists(load_path):
                    return load_path + ' file not found.'
            field_dict = np.load(load_path, allow_pickle=True)
        else:
            field_dict = self.field_dict_cache
        self.model.import_patch(field_dict, fp16=self.fp16)
        self.update_optimizer_scheduler()
        return load_path
    
    def load_shape(self, load_path=None):
        if load_path is None:
            load_path = os.path.join(self.workspace, 'field', 'target.obj')
            if not os.path.exists(load_path):
                return load_path + ' file not found.'
        self.model.import_shape(load_path, fp16=self.fp16)
        self.update_optimizer_scheduler()
        return load_path

    def load_unhash(self, load_path=None):
        if load_path is None:
            load_path = os.path.join(self.workspace, 'field', 'curved_mesh.npz')
            if not os.path.exists(load_path):
                return load_path + ' file not found'
        self.model.import_unhash(data_path=load_path, fp16=self.fp16)
        self.update_optimizer_scheduler()
        return load_path
    
    def set_uv_utilize_rate(self, rate=1.):
        self.model.set_uv_utilize_rate(rate, fp16=self.fp16)
        self.update_optimizer_scheduler()
        return rate

    def set_k_for_uv(self, k_for_uv=5):
        self.model.set_k_for_uv(k_for_uv, fp16=self.fp16)
        self.update_optimizer_scheduler()
        return k_for_uv
    
    def set_sdf_factor(self, sdf_factor=1.):
        self.model.set_sdf_factor(sdf_factor, fp16=self.fp16)
        self.update_optimizer_scheduler()
        return sdf_factor
    
    def set_sdf_offset(self, sdf_offset=0.):
        self.model.set_sdf_offset(sdf_offset, fp16=self.fp16)
        self.update_optimizer_scheduler()
        return sdf_offset
    
    def set_h_threshold(self, h_threshold=0.):
        self.model.set_h_threshold(h_threshold, fp16=self.fp16)
        self.update_optimizer_scheduler()
        return h_threshold
    
    def visualize_features(self, sv_path=None):
        if sv_path is None:
            sv_path = os.path.join(self.workspace, 'field', 'features_visualization')
        self.model.visualize_features(sv_path=sv_path)
        return sv_path
    
    def update_optimizer_scheduler(self):
        self.optimizer = self.optimizer_func(self.model)
        self.lr_scheduler = self.lr_scheduler_func(self.optimizer)
        if self.ema is not None:
            self.ema = ExponentialMovingAverage(self.model.parameters(), decay=self.ema_decay)

    def save_envmap(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.workspace, 'envmap', 'source')
        if not os.path.exists('/'.join(save_path.split('/')[:-1])):
            os.makedirs('/'.join(save_path.split('/')[:-1]))
        self.model.light_net.save_envmap(save_path)
        return save_path

    def load_envmap(self, load_path=None, log_path=None):
        if load_path is None:
            load_path = os.path.join(self.workspace, 'envmap', 'target')
            log_path = os.path.join(self.workspace, 'envmap', 'opt_logs')
        self.model.light_net.load_envmap(path=load_path, log_path=log_path)
        return load_path

    def train_one_epoch(self, loader):
        self.log(f"==> Start Training Epoch {self.epoch}, lr={self.optimizer.param_groups[0]['lr']:.6f} ...")

        total_loss = 0
        if self.local_rank == 0 and self.report_metric_at_train:
            for metric in self.metrics:
                metric.clear()

        self.model.train()

        # distributedSampler: must call set_epoch() to shuffle indices across multiple epochs
        # ref: https://pytorch.org/docs/stable/data.html
        if self.world_size > 1:
            loader.sampler.set_epoch(self.epoch)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        self.local_step = 0

        for data in loader:
            
            # update grid every 16 steps
            if self.model.cuda_ray and self.global_step % 16 == 0:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()
                        
            if self.update_gridfield:
                self.model.update_gridfield(target_stage=int(self.global_step // self.num_iterations_per_stage))        
                self.update_optimizer_scheduler()
            
            self.local_step += 1
            self.global_step += 1

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)
         
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val

            if self.local_rank == 0:
                if self.report_metric_at_train:
                    for metric in self.metrics:
                        metric.update(preds, truths)
                        
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss_val, self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                if self.scheduler_update_every_step:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f}), lr={self.optimizer.param_groups[0]['lr']:.6f}")
                else:
                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                pbar.update(loader.batch_size)

        if self.ema is not None:
            self.ema.update()

        average_loss = total_loss / self.local_step
        self.stats["loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if self.report_metric_at_train:
                for metric in self.metrics:
                    self.log(metric.report(), style="red")
                    if self.use_tensorboardX:
                        metric.write(self.writer, self.epoch, prefix="train")
                    metric.clear()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        self.log(f"==> Finished Epoch {self.epoch}.")

    def evaluate_one_epoch(self, loader, name=None):
        self.log(f"++> Evaluate at epoch {self.epoch} ...")

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        total_loss = 0
        if self.local_rank == 0:
            for metric in self.metrics:
                metric.clear()

        self.model.eval()

        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}: {percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        with torch.no_grad():
            self.local_step = 0

            # update grid
            if self.model.cuda_ray:
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    self.model.update_extra_state()

            for data in loader:    
                self.local_step += 1

                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, preds_depth, truths, loss = self.eval_step(data)

                # all_gather/reduce the statistics (NCCL only support all_*)
                if self.world_size > 1:
                    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                    loss = loss / self.world_size
                    
                    preds_list = [torch.zeros_like(preds).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_list, preds)
                    preds = torch.cat(preds_list, dim=0)

                    preds_depth_list = [torch.zeros_like(preds_depth).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(preds_depth_list, preds_depth)
                    preds_depth = torch.cat(preds_depth_list, dim=0)

                    truths_list = [torch.zeros_like(truths).to(self.device) for _ in range(self.world_size)] # [[B, ...], [B, ...], ...]
                    dist.all_gather(truths_list, truths)
                    truths = torch.cat(truths_list, dim=0)
                
                loss_val = loss.item()
                total_loss += loss_val

                # only rank = 0 will perform evaluation.
                if self.local_rank == 0:

                    for metric in self.metrics:
                        metric.update(preds, truths)

                    # save image
                    save_path = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}.png')
                    save_path_depth = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_depth.png')
                    #save_path_gt = os.path.join(self.workspace, 'validation', f'{name}_{self.local_step:04d}_gt.png')

                    #self.log(f"==> Saving validation image to {save_path}")
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    if self.opt.color_space == 'linear':
                        preds = linear_to_srgb(preds)

                    pred = preds[0].detach().cpu().numpy()
                    pred_depth = preds_depth[0].detach().cpu().numpy()
                    
                    cv2.imwrite(save_path, cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    cv2.imwrite(save_path_depth, (pred_depth * 255).astype(np.uint8))
                    #cv2.imwrite(save_path_gt, cv2.cvtColor((linear_to_srgb(truths[0].detach().cpu().numpy()) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

                    pbar.set_description(f"loss={loss_val:.4f} ({total_loss/self.local_step:.4f})")
                    pbar.update(loader.batch_size)


        average_loss = total_loss / self.local_step
        self.stats["valid_loss"].append(average_loss)

        if self.local_rank == 0:
            pbar.close()
            if not self.use_loss_as_metric and len(self.metrics) > 0:
                result = self.metrics[0].measure()
                self.stats["results"].append(result if self.best_mode == 'min' else - result) # if max mode, use -result
            else:
                self.stats["results"].append(average_loss) # if no metric, choose best by min loss

            for metric in self.metrics:
                self.log(metric.report(), style="blue")
                if self.use_tensorboardX:
                    metric.write(self.writer, self.epoch, prefix="evaluate")
                metric.clear()

        if self.ema is not None:
            self.ema.restore()

        self.log(f"++> Evaluate epoch {self.epoch} Finished.")

    def save_checkpoint(self, name=None, full=False, best=False, remove_old=True):

        if name is None:
            name = f'{self.name}_ep{self.epoch:04d}'

        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'stats': self.stats,
        }

        if self.model.cuda_ray:
            state['mean_count'] = self.model.mean_count
            state['mean_density'] = self.model.mean_density

        if full:
            state['optimizer'] = self.optimizer.state_dict()
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            state['scaler'] = self.scaler.state_dict()
            if self.ema is not None:
                state['ema'] = self.ema.state_dict()
        
        if not best:

            state['model'] = self.model.state_dict()

            file_path = f"{self.ckpt_path}/{name}.pth"

            if remove_old:
                self.stats["checkpoints"].append(file_path)

                if len(self.stats["checkpoints"]) > self.max_keep_ckpt:
                    old_ckpt = self.stats["checkpoints"].pop(0)
                    if os.path.exists(old_ckpt):
                        os.remove(old_ckpt)

            torch.save(state, file_path)

        else:    
            if len(self.stats["results"]) > 0:
                if self.stats["best_result"] is None or self.stats["results"][-1] < self.stats["best_result"]:
                    self.log(f"[INFO] New best result: {self.stats['best_result']} --> {self.stats['results'][-1]}")
                    self.stats["best_result"] = self.stats["results"][-1]

                    # save ema results 
                    if self.ema is not None:
                        self.ema.store()
                        self.ema.copy_to()

                    state['model'] = self.model.state_dict()

                    if self.ema is not None:
                        self.ema.restore()
                    
                    torch.save(state, self.best_path)
            else:
                self.log(f"[WARN] no evaluated results found, skip saving best checkpoint.")
            
    def load_checkpoint(self, checkpoint=None, model_only=False):
        if checkpoint is None:
            checkpoint_list = sorted(glob.glob(f'{self.ckpt_path}/{self.name}_ep*.pth'))
            if checkpoint_list:
                checkpoint = checkpoint_list[-1]
                self.log(f"[INFO] Latest checkpoint is {checkpoint}")
            else:
                self.log("[WARN] No checkpoint found, model randomly initialized.")
                return

        checkpoint_dict = torch.load(checkpoint, map_location=self.device)
        
        if 'model' not in checkpoint_dict:
            self.model.load_state_dict(checkpoint_dict)
            self.log("[INFO] loaded model.")
            return

        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
        self.log("[INFO] loaded model.")
        if len(missing_keys) > 0:
            self.log(f"[WARN] missing keys: {missing_keys}")
        if len(unexpected_keys) > 0:
            self.log(f"[WARN] unexpected keys: {unexpected_keys}")   

        if self.ema is not None and 'ema' in checkpoint_dict:
            self.ema.load_state_dict(checkpoint_dict['ema'])

        if self.model.cuda_ray:
            if 'mean_count' in checkpoint_dict:
                self.model.mean_count = checkpoint_dict['mean_count']
            if 'mean_density' in checkpoint_dict:
                self.model.mean_density = checkpoint_dict['mean_density']
        
        if model_only:
            return

        self.stats = checkpoint_dict['stats']
        self.epoch = checkpoint_dict['epoch']
        self.global_step = checkpoint_dict['global_step']
        self.log(f"[INFO] load at epoch {self.epoch}, global step {self.global_step}")
        
        if self.optimizer and  'optimizer' in checkpoint_dict:
            try:
                self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
                self.log("[INFO] loaded optimizer.")
            except:
                self.log("[WARN] Failed to load optimizer.")
        
        if self.lr_scheduler and 'lr_scheduler' in checkpoint_dict:
            try:
                self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
                self.log("[INFO] loaded scheduler.")
            except:
                self.log("[WARN] Failed to load scheduler.")
        
        if self.scaler and 'scaler' in checkpoint_dict:
            try:
                self.scaler.load_state_dict(checkpoint_dict['scaler'])
                self.log("[INFO] loaded scaler.")
            except:
                self.log("[WARN] Failed to load scaler.")