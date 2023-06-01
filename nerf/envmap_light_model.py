import os
import cv2
import glob
import torch
import imageio
import numpy as np
from torch import nn
import tinycudann as tcnn
from tools.shape_tools import write_ply_rgb


def isclose(x, val, threshold = 1e-6):
    return torch.abs(x - val) <= threshold


def safe_pow(x, p):
    sqrt_in = torch.relu(torch.where(isclose(x, 0.0), torch.ones_like(x) * 1e-6, x))
    return torch.pow(sqrt_in, p)


def sph2cart(pts_sph):
    """Inverse of :func:`cart2sph`.
    See :func:`cart2sph`.
    """
    pts_sph = np.array(pts_sph)

    # Validate inputs
    is_one_point = False
    if pts_sph.shape == (3,):
        is_one_point = True
        pts_sph = pts_sph.reshape(1, 3)
    elif pts_sph.ndim != 2 or pts_sph.shape[1] != 3:
        raise ValueError("Shape of input must be either (3,) or (n, 3)")
    pts_r_lat_lng = pts_sph

    # Compute x, y and z
    r = pts_r_lat_lng[:, 0]
    lat = pts_r_lat_lng[:, 1]
    lng = pts_r_lat_lng[:, 2]
    z = r * np.sin(lat)
    x = r * np.cos(lat) * np.cos(lng)
    y = r * np.cos(lat) * np.sin(lng)

    # Assemble and return
    pts_cart = np.stack((x, y, z), axis=-1)

    if is_one_point:
        pts_cart = pts_cart.reshape(3)

    return pts_cart


def gen_light_xyz(envmap_h, envmap_w, envmap_radius=1e2):
    """Additionally returns the associated solid angles, for integration.
    """
    # OpenEXR "latlong" format
    # lat = pi/2
    # lng = pi
    #     +--------------------+
    #     |                    |
    #     |                    |
    #     +--------------------+
    #                      lat = -pi/2
    #                      lng = -pi
    lat_step_size = np.pi / (envmap_h + 2)
    lng_step_size = 2 * np.pi / (envmap_w + 2)
    # Try to exclude the problematic polar points
    lats = np.linspace(
        np.pi / 2 - lat_step_size, -np.pi / 2 + lat_step_size, envmap_h)
    lngs = np.linspace(
        np.pi - lng_step_size, -np.pi + lng_step_size, envmap_w)
    lngs, lats = np.meshgrid(lngs, lats)

    # To Cartesian
    rlatlngs = np.dstack((envmap_radius * np.ones_like(lats), lats, lngs))
    rlatlngs = rlatlngs.reshape(-1, 3)
    xyz = sph2cart(rlatlngs)
    xyz = xyz.reshape(envmap_h, envmap_w, 3)

    # Calculate the area of each pixel on the unit sphere (useful for
    # integration over the sphere)
    sin_colat = np.sin(np.pi / 2 - lats)
    areas = 4 * np.pi * sin_colat / np.sum(sin_colat)

    assert 0 not in areas, \
        "There shouldn't be light pixel that doesn't contribute"

    return np.array(xyz, dtype=np.float32), np.array(areas, dtype=np.float32)


class Envmap_EnvmapMaterialNet(nn.Module):
    def __init__(self,
                 input_dim, 
                 env_res=8,
                 white_light=False,
                 use_specular=True):
        super().__init__()

        # Environment Lights
        self.env_res = env_res
        self.light_color_dim = 1 if white_light else 3
        self.white_light = white_light
        self.env_map = nn.Parameter(torch.zeros([self.env_res, self.env_res, self.light_color_dim]), requires_grad=True)
        torch.nn.init.xavier_normal_(self.env_map)
        light_probes, areas = gen_light_xyz(self.env_res, self.env_res, 1e2)
        light_probes = torch.from_numpy(light_probes).float()
        light_probes = light_probes / light_probes.norm(dim=-1, keepdim=True)
        self.light_probes = nn.Parameter(light_probes, requires_grad=False)
        self.light_areas = nn.Parameter(torch.from_numpy(areas).unsqueeze(-1).float(), requires_grad=False)
        self.gamma = 2.4
        self.min_glossiness = 1.

        ############## BRDF Network ############
        self.brdf_layer = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=5,  # albedo[3], specular[1], glossiness[1]
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "Relu",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 3 - 1,
            },
        )
        self.softplus = nn.Softplus()
        self.use_specular = use_specular
        self.name = 'envmap'
        self.envmap_import = None
        self.import_envmap = False

    def forward(self, geo_feat, normals, view_dirs, **kwargs):
        env_map = self.env_map if not self.import_envmap else self.envmap_import
        prefix = geo_feat.shape[:-1]

        # Spatial Attributes
        spatial_output = self.brdf_layer(geo_feat)
        k_diffuse = torch.sigmoid(spatial_output[..., :3])  # N points, 3
        k_specular = torch.sigmoid(spatial_output[..., 3: 4])  # N points, 3
        glossiness = self.softplus(spatial_output[..., -1:]) + self.min_glossiness  # N points, 1
        
        # Light Rotation Editing
        light_probes = self.light_probes.reshape([-1, 3])

        # Integral Coefficients
        r = 2 * (-view_dirs * normals).sum(-1, keepdim=True) * normals + view_dirs  # N, 3
        r, v = r.unsqueeze(1), (- view_dirs).unsqueeze(1)  # N, 1, 3; N, 1, 3
        l = light_probes.unsqueeze(0)  # 1, K, 3
        h = (l + v) / (torch.norm(l + v, dim=-1, keepdim=True) + 1e-5)  # N, K, 3
        nl = torch.clamp((normals.unsqueeze(1) * l).sum(-1, keepdim=True), min=0., max=1.1)  # N points, K lights, 1
        nh = torch.clamp((normals.unsqueeze(1) * h).sum(-1, keepdim=True), min=0., max=1.1)  # N points, K lights, 1

        # Occlusion
        if 'normal_coarse' in kwargs.keys():
            normal_coarse = kwargs['normal_coarse']
            visibility = ((normal_coarse.unsqueeze(1) * l).sum(-1, keepdim=True) > 0).float()
        else:
            visibility = torch.ones_like(nl)

        # Color
        light = (self.softplus(env_map) * self.light_areas).reshape([-1, env_map.shape[-1]])
        diffuse = (k_diffuse.unsqueeze(1) * nl * light.unsqueeze(0) * visibility).sum(1)
        if self.use_specular:
            specular = (k_specular.unsqueeze(1) * torch.pow(nh, glossiness.unsqueeze(1)) * light.unsqueeze(0) * visibility).sum(1)
        else:
            specular = torch.zeros_like(diffuse)
        color = diffuse + specular

        # Tone Mapping
        color = color.clamp(0).reshape([*prefix, -1])
        diffuse = diffuse.clamp(0, 1).reshape([*prefix, -1])
        specular = specular.clamp(0, 1).reshape([*prefix, -1])
        color = safe_pow(color, 1/self.gamma)
        diffuse = safe_pow(diffuse, 1/self.gamma)
        specular = safe_pow(specular, 1/self.gamma)

        return color, specular, diffuse, k_diffuse

    def save_envmap(self, sv_path, H=256, W=512):
        sv_path = self.specific_path(sv_path)
        envmap, viewdirs = envmap2Envmap(self.env_map, light_lobes=self.light_probes, H=H, W=W, clamp=True)
        save_envmap(envmap=self.env_map.detach().cpu().numpy(), Envmap=envmap, viewdirs=viewdirs, path=sv_path)
    
    def load_envmap(self, path, **kwargs):
        if os.path.exists(self.specific_path(path) + '.npy'):
            print('Loading envmap from ', self.specific_path(path) + '.npy')
            self.envSHs_import = nn.Parameter(torch.from_numpy(np.load(self.specific_path(path) + '.npy')).cuda(), requires_grad=False)
        else:
            files = glob.glob(path + '*')
            files = [file for file in files if file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.jpg') or file.endswith('JPEG')]
            if len(files) == 0:
                print('No envmap found: ', path)
                return
            file = files[0]
            envmap = image2envmap(file)
            envmap = cv2.resize(envmap, (self.env_res, self.env_res))
            self.envmap_import = nn.Parameter(torch.from_numpy(envmap).float().to(self.env_map.device), requires_grad=False)
        self.import_envmap = True
        torch.cuda.empty_cache()
        print('Load Envmap Done!')
    
    def specific_path(self, path):
        return path + '_' + self.name


def envmap2Envmap(env_map, light_lobes, H, W, clamp=True):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if clamp:
        env_map = env_map.clamp(0, 1)
    env_map = env_map.detach().cpu().numpy()
    light_lobes = light_lobes / light_lobes.norm(dim=-1, keepdim=True)
    light_lobes = light_lobes.detach().cpu().numpy()
    env_map = cv2.resize(env_map, (W, H))
    viewdirs = cv2.resize(light_lobes, (W, H))
    return env_map, viewdirs

def image2envmap(img_path, force_white=False):
    im = imageio.imread(img_path)[..., :3] / 255.

    # # y = a * exp(b*x)
    # max_e = 10
    # fixed = .5
    # b = 1 / (fixed - 1) * np.log((fixed + 1) / max_e)
    # a = max_e / (np.exp(b))
    # envmap = a * np.exp(b *im) - 1

    im = (im - .5) * 2
    envmap = 10 ** im - 1e-1

    if force_white:
        envmap[..., :] = envmap.mean(axis=-1, keepdims=True)

    return envmap

def save_envmap(envmap, Envmap, viewdirs, path, no_npz=False, no_image=False, no_ply=False):
    im = np.power(Envmap, 1./2.2)
    im = np.clip(im, 0., 1.)
    im = np.uint8(im * 255.)
    if not no_npz:
        np.save(path + '.npy', Envmap)
        np.save(path + '_envmap.npy', envmap)
    if not no_image:
        imageio.imwrite(path + '.png', im)
    if not no_ply:
        if im.shape[-1] != 3:
            im = im.reshape([-1])
            im = np.stack([im, im, im], axis=-1)
        write_ply_rgb(viewdirs.reshape([-1, 3]), im.reshape([-1, 3]), path + '.ply')
