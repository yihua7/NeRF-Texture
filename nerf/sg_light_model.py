import os
import cv2
import glob
import torch
import imageio
import numpy as np
import torch.nn as nn
import tinycudann as tcnn
import torch.nn.functional as F
from tools.shape_tools import write_ply_rgb

TINY_NUMBER = 1e-6


#######################################################################################################
# compute envmap from SG
#######################################################################################################
def compute_envmap(lgtSGs, H, W, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])

    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                           dim=-1)    # [H, W, 3]
    print(viewdirs[0, 0, :], viewdirs[0, W//2, :], viewdirs[0, -1, :])
    print(viewdirs[H//2, 0, :], viewdirs[H//2, W//2, :], viewdirs[H//2, -1, :])
    print(viewdirs[-1, 0, :], viewdirs[-1, W//2, :], viewdirs[-1, -1, :])

    lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, 3))
    return envmap


def compute_envmap_pcd(lgtSGs, N=1000, upper_hemi=False):
    viewdirs = torch.randn((N, 3))
    viewdirs = viewdirs / (torch.norm(viewdirs, dim=-1, keepdim=True) + TINY_NUMBER)

    if upper_hemi:
        # y > 0
        viewdirs = torch.cat((viewdirs[:, 0:1], torch.abs(viewdirs[:, 1:2]), viewdirs[:, 2:3]), dim=-1)

    lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 7]).expand(dots_sh+[M, 7])

    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values

    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]

    return viewdirs.squeeze(-2), rgb

#######################################################################################################
# below are a few utility functions
#######################################################################################################
def prepend_dims(tensor, shape):
    '''
    :param tensor: tensor of shape [a1, a2, ..., an]
    :param shape: shape to prepend, e.g., [b1, b2, ..., bm]
    :return: tensor of shape [b1, b2, ..., bm, a1, a2, ..., an]
    '''
    orig_shape = list(tensor.shape)
    tensor = tensor.view([1] * len(shape) + orig_shape).expand(shape + [-1] * len(orig_shape))
    return tensor


def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER
    # orig impl; might be numerically unstable
    # t = torch.sqrt(lambda_val) * (1.6988 * lambda_val * lambda_val + 10.8438 * lambda_val) / (lambda_val * lambda_val + 6.2201 * lambda_val + 10.2415)

    inv_lambda_val = 1. / lambda_val
    t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

    # orig impl; might be numerically unstable
    # a = torch.exp(t)
    # b = torch.exp(t * cos_beta)
    # s = (a * b - 1.) / ((a - 1.) * (b + 1.))

    ### note: for numeric stability
    inv_a = torch.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
    s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
    b = torch.exp(t * torch.clamp(cos_beta, max=0.))
    s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
    s = mask * s1 + (1. - mask) * s2

    A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
    A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

    return A_b * (1. - s) + A_u * s


def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    # assume lambda1 << lambda2
    ratio = lambda1 / lambda2

    dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = torch.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
    tmp = torch.min(tmp, ratio + 1.)

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / tmp
    lambda2_over_lambda3 = 1. / tmp
    diff = lambda2 * (tmp - ratio - 1.)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
    final_lambdas = lambda3
    final_mus = mu1 * mu2 * torch.exp(diff)

    return final_lobes, final_lambdas, final_mus


#######################################################################################################
# below is the SG renderer
#######################################################################################################
def render_with_sg(lgtSGs, specular_reflectance, roughness, diffuse_albedo, normal, viewdirs, blending_weights=None, diffuse_rgb=None, normal_coarse=None):
    '''
    :param lgtSGs: [M, 7]
    :param specular_reflectance: [K, 3];
    :param roughness: [K, 1]; values must be positive
    :param diffuse_albedo: [..., 3]; values must lie in [0,1]
    :param normal: [..., 3]; ----> camera; must have unit norm
    :param viewdirs: [..., 3]; ----> camera; must have unit norm
    :param blending_weights: [..., K]; values must be positive, and sum to one along last dimension
    :return [..., 3]
    '''
    M = lgtSGs.shape[0]
    K = specular_reflectance.shape[0]
    assert (K == roughness.shape[0])
    dots_shape = list(normal.shape[:-1])

    ########################################
    # specular color
    ########################################
    #### note: sanity
    # normal = normal / (torch.norm(normal, dim=-1, keepdim=True) + TINY_NUMBER)  # [..., 3]; ---> camera
    normal = normal.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])  # [..., M, K, 3]
    if normal_coarse is not None:
        normal_coarse = normal_coarse.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])  # [..., M, K, 3]

    # viewdirs = viewdirs / (torch.norm(viewdirs, dim=-1, keepdim=True) + TINY_NUMBER)  # [..., 3]; ---> camera
    viewdirs = viewdirs.unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, K, 3])  # [..., M, K, 3]

    # light
    lgtSGs = prepend_dims(lgtSGs, dots_shape)  # [..., M, 7]
    lgtSGs = lgtSGs.unsqueeze(-2).expand(dots_shape + [M, K, 7])  # [..., M, K, 7]
    #### note: sanity
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)  # [..., M, 3]
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values

    # NDF
    brdfSGLobes = normal  # use normal as the brdf SG lobes
    inv_roughness_pow4 = 1. / (roughness * roughness * roughness * roughness)  # [K, 1]
    brdfSGLambdas = prepend_dims(2. * inv_roughness_pow4, dots_shape + [M, ])  # [..., M, K, 1]; can be huge
    mu_val = (inv_roughness_pow4 / np.pi).expand([K, 3])  # [K, 1] ---> [K, 3]
    brdfSGMus = prepend_dims(mu_val, dots_shape + [M, ])  # [..., M, K, 3]

    # perform spherical warping
    v_dot_lobe = torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
    warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs
    warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
    # warpBrdfSGLambdas = brdfSGLambdas / (4 * torch.abs(torch.sum(brdfSGLobes * viewdirs, dim=-1, keepdim=True)) + TINY_NUMBER)
    warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)  # can be huge
    warpBrdfSGMus = brdfSGMus  # [..., M, K, 3]

    # add fresnel and geometric terms; apply the smoothness assumption in SG paper
    new_half = warpBrdfSGLobes + viewdirs
    new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)
    v_dot_h = torch.sum(viewdirs * new_half, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_h = torch.clamp(v_dot_h, min=0.)
    specular_reflectance = prepend_dims(specular_reflectance, dots_shape + [M, ])  # [..., M, K, 3]
    F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

    dot1 = torch.sum(warpBrdfSGLobes * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot1 = torch.clamp(dot1, min=0.)
    dot2 = torch.sum(viewdirs * normal, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot2 = torch.clamp(dot2, min=0.)
    k = (roughness + 1.) * (roughness + 1.) / 8.
    G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
    G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
    G = G1 * G2

    Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)
    warpBrdfSGMus = warpBrdfSGMus * Moi

    # multiply with light sg
    final_lobes, final_lambdas, final_mus = lambda_trick(lgtSGLobes, lgtSGLambdas, lgtSGMus,
                                                         warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)

    # now multiply with clamped cosine, and perform hemisphere integral
    mu_cos = 32.7080
    lambda_cos = 0.0315
    alpha_cos = 31.7003
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                      final_lobes, final_lambdas, final_mus)

    dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
    dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)

    # [..., M, K, 3]
    if normal_coarse is None:
        specular_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    else:
        lobe_prime_coarse, lambda_prime_coarse, mu_prime_coarse = lambda_trick(normal_coarse, lambda_cos, mu_cos, lobe_prime, lambda_prime, mu_prime)
        lobe_final_coarse, lambda_final_coarse, mu_final_coarse = lambda_trick(normal_coarse, lambda_cos, mu_cos, final_lobes, final_lambdas, final_mus)

        dotpc = torch.sum(lobe_prime_coarse * normal, dim=-1, keepdim=True)
        dotfc = torch.sum(lobe_final_coarse * normal, dim=-1, keepdim=True)

        specular_rgb = mu_prime_coarse * hemisphere_int(lambda_prime_coarse, dotpc) - alpha_cos * mu_final_coarse * hemisphere_int(lambda_final_coarse, dotfc) - alpha_cos * mu_prime * hemisphere_int(lambda_prime, dot1) + (alpha_cos ** 2) * final_mus * hemisphere_int(final_lambdas, dot2)
    
    if blending_weights is None:     
        specular_rgb = specular_rgb.sum(dim=-2).sum(dim=-2)
    else:
        specular_rgb = (specular_rgb.sum(dim=-3) * blending_weights.unsqueeze(-1)).sum(dim=-2)
    specular_rgb = torch.clamp(specular_rgb, min=0.)

    ########################################
    # per-point hemisphere integral of envmap
    ########################################
    if diffuse_rgb is None:
        diffuse = (diffuse_albedo / np.pi).unsqueeze(-2).unsqueeze(-2).expand(dots_shape + [M, 1, 3])

        # multiply with light sg
        final_lobes = lgtSGLobes.narrow(dim=-2, start=0, length=1)  # [..., M, K, 3] --> [..., M, 1, 3]
        final_mus = lgtSGMus.narrow(dim=-2, start=0, length=1) * diffuse
        final_lambdas = lgtSGLambdas.narrow(dim=-2, start=0, length=1)

        # now multiply with clamped cosine, and perform hemisphere integral
        lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                          final_lobes, final_lambdas, final_mus)

        dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
        dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
        if normal_coarse is None:
            diffuse_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - \
                        final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)    
        else:
            lobe_prime_coarse, lambda_prime_coarse, mu_prime_coarse = lambda_trick(normal_coarse, lambda_cos, mu_cos, lobe_prime, lambda_prime, mu_prime)
            lobe_final_coarse, lambda_final_coarse, mu_final_coarse = lambda_trick(normal_coarse, lambda_cos, mu_cos, final_lobes, final_lambdas, final_mus)

            dotpc = torch.sum(lobe_prime_coarse * normal, dim=-1, keepdim=True)
            dotfc = torch.sum(lobe_final_coarse * normal, dim=-1, keepdim=True)

            diffuse_rgb = mu_prime_coarse * hemisphere_int(lambda_prime_coarse, dotpc) - alpha_cos * mu_final_coarse * hemisphere_int(lambda_final_coarse, dotfc) - alpha_cos * mu_prime * hemisphere_int(lambda_prime, dot1) + (alpha_cos ** 2) * final_mus * hemisphere_int(final_lambdas, dot2)
        
        diffuse_rgb = diffuse_rgb.sum(dim=-2).sum(dim=-2)
        diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

    rgb = specular_rgb + diffuse_rgb
    return rgb, specular_rgb, diffuse_rgb, diffuse_albedo


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


### uniformly distribute points on a sphere
def fibonacci_sphere(samples=1):
    '''
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    :param samples:
    :return:
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


def compute_energy(lgtSGs):
    lgtLambda = torch.abs(lgtSGs[:, 3:4])       # [M, 1]
    lgtMu = torch.abs(lgtSGs[:, 4:])               # [M, 3]
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy


class SG_EnvmapMaterialNet(nn.Module):
    def __init__(self,
                 input_dim,
                 white_specular=False,
                 white_light=False,
                 num_lgt_sgs=32,
                 num_base_materials=1,
                 upper_hemi=False,
                 fix_specular_albedo=False,
                 specular_albedo=[-1.,-1.,-1.]):
        super().__init__()

        ############## spatially-varying diffuse albedo############
        print('Diffuse albedo network size: ', 256)
        diffuse_albedo_layers = tcnn.Network(
            n_input_dims=input_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "Relu",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 3 - 1,
            },
        )
        self.diffuse_albedo_layers = diffuse_albedo_layers

        ##################### specular rgb ########################
        self.numLgtSGs = num_lgt_sgs
        self.numBrdfSGs = num_base_materials
        print('Number of Light SG: ', self.numLgtSGs)
        print('Number of BRDF SG: ', self.numBrdfSGs)
        # by using normal distribution, the lobes are uniformly distributed on a sphere at initialization
        self.white_light = white_light
        if self.white_light:
            print('Using white light!')
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 5), requires_grad=True)   # [M, 5]; lobe + lambda + mu
            # self.lgtSGs.data[:, -1] = torch.clamp(torch.abs(self.lgtSGs.data[:, -1]), max=0.01)
        else:
            self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True)   # [M, 7]; lobe + lambda + mu
            self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))
            # self.lgtSGs.data[:, -3:] = torch.clamp(torch.abs(self.lgtSGs.data[:, -3:]), max=0.01)

        # make sure lambda is not too close to zero
        self.lgtSGs.data[:, 3:4] = 20. + torch.abs(self.lgtSGs.data[:, 3:4] * 100.)
        # make sure total energy is around 1.
        energy = compute_energy(self.lgtSGs.data)
        # print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())
        self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi
        energy = compute_energy(self.lgtSGs.data)
        print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        # deterministicly initialize lobes
        lobes = fibonacci_sphere(self.numLgtSGs).astype(np.float32)
        self.lgtSGs.data[:, :3] = torch.from_numpy(lobes)
        # check if lobes are in upper hemisphere
        self.upper_hemi = upper_hemi
        if self.upper_hemi:
            print('Restricting lobes to upper hemisphere!')
            self.restrict_lobes_upper = lambda lgtSGs: torch.cat((lgtSGs[..., :1], torch.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)
            # limit lobes to upper hemisphere
            self.lgtSGs.data = self.restrict_lobes_upper(self.lgtSGs.data)

        self.white_specular = white_specular
        self.fix_specular_albedo = fix_specular_albedo
        if self.fix_specular_albedo:
            print('Fixing specular albedo: ', specular_albedo)
            specular_albedo = np.array(specular_albedo).astype(np.float32)
            assert(self.numBrdfSGs == 1)
            assert(np.all(np.logical_and(specular_albedo > 0., specular_albedo < 1.)))
            self.specular_reflectance = nn.Parameter(torch.from_numpy(specular_albedo).reshape((self.numBrdfSGs, 3)),
                                                     requires_grad=False)  # [K, 1]
        else:
            if self.white_specular:
                print('Using white specular reflectance!')
                self.specular_reflectance = nn.Parameter(torch.randn(self.numBrdfSGs, 1),
                                                         requires_grad=True)   # [K, 1]
            else:
                self.specular_reflectance = nn.Parameter(torch.randn(self.numBrdfSGs, 3),
                                                         requires_grad=True)   # [K, 3]
            self.specular_reflectance.data = torch.abs(self.specular_reflectance.data)

        # optimize
        # roughness = [np.random.uniform(-1.5, -1.0) for i in range(self.numBrdfSGs)]       # small roughness
        # roughness = [np.random.uniform(1.5, 2.0) for i in range(self.numBrdfSGs)]           # big roughness
        roughness = [np.random.uniform(4e-2, 5e-2) for i in range(self.numBrdfSGs)]           # tiny roughness
        roughness = np.array(roughness).astype(dtype=np.float32).reshape((self.numBrdfSGs, 1))  # [K, 1]
        print('init roughness: ', 1.0 / (1.0 + np.exp(-roughness)))
        self.roughness = nn.Parameter(torch.from_numpy(roughness),
                                      requires_grad=True)

        # blending weights
        if self.numBrdfSGs > 1:
            self.blending_weights_layers = tcnn.Network(
                n_input_dims=input_dim,
                n_output_dims=self.numBrdfSGs,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "Relu",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 3 - 1,
                },
            )

        self.import_envmap = False
        self.lgtSGs_import = None
        self.numLgtSGs_import = None
        self.white_light_import = False
        self.name = 'sg'

    def material(self, geo_feat):

        lgtSGs = self.lgtSGs if not self.import_envmap or self.training else self.lgtSGs_import
        white_light = self.white_light if not self.import_envmap or self.training else self.white_light_import

        if geo_feat is None:
            diffuse_albedo = None
            blending_weights = None
        else:
            diffuse_albedo = torch.sigmoid(self.diffuse_albedo_layers(geo_feat))

            if self.numBrdfSGs > 1:
                blending_weights = F.softmax(self.blending_weights_layers(geo_feat), dim=-1)
            else:
                blending_weights = None

        if self.fix_specular_albedo:
            specular_reflectacne = self.specular_reflectance
        else:
            specular_reflectacne = torch.sigmoid(self.specular_reflectance)
            if self.white_specular:
                specular_reflectacne = specular_reflectacne.expand((-1, 3))     # [K, 3]

        roughness = torch.sigmoid(self.roughness)

        if white_light:
            lgtSGs = torch.cat((lgtSGs, lgtSGs[..., -1:], lgtSGs[..., -1:]), dim=-1)
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        ret = dict([
            ('sg_lgtSGs', lgtSGs),
            ('sg_specular_reflectance', specular_reflectacne),
            ('sg_roughness', roughness),
            ('sg_diffuse_albedo', diffuse_albedo),
            ('sg_blending_weights', blending_weights)
        ])
        return ret

    def forward(self, geo_feat, normals, view_dirs, **kwargs):
        sg_envmap_material = self.material(geo_feat)
        rgb, specular_rgb, diffuse_rgb, diffuse_albedo = render_with_sg(lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                                                        specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                                                        roughness=sg_envmap_material['sg_roughness'],
                                                                        diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                                                                        normal=normals, viewdirs=view_dirs,
                                                                        blending_weights=sg_envmap_material['sg_blending_weights'],
                                                                        normal_coarse=None)
        return rgb, specular_rgb, diffuse_rgb, diffuse_albedo

    def save_envmap(self, sv_path, H=256, W=512):
        sv_path = self.specific_path(sv_path)
        envmap, viewdirs = SG2Envmap(self.lgtSGs, H=H, W=W, upper_hemi=self.upper_hemi)
        envmap = envmap.detach().cpu().numpy()
        if self.white_light:
            envmap = np.concatenate([envmap, envmap, envmap], axis=-1)
        viewdirs = viewdirs.detach().cpu().numpy()
        save_envmap(envmap=envmap, viewdirs=viewdirs, path=sv_path)
    
    def load_envmap(self, path, log_path):
        if os.path.exists(self.specific_path(path) + '.npy'):
            print('Loading envmap from ', self.specific_path(path) + '.npy')
            self.lgtSGs_import = nn.Parameter(torch.from_numpy(np.load(self.specific_path(path) + '.npy')).cuda(), requires_grad=False)
        else:
            files = glob.glob(path + '*')
            files = [file for file in files if file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.jpg') or file.endswith('JPEG')]
            if len(files) == 0:
                print('No envmap found: ', path)
                return
            file = files[0]
            print('Optimizing SG towards ', file, ' ...')
            envmap = image2envmap(file)
            lgtSGs = EnvMap2SG(envmap=envmap, numLgtSGs=self.lgtSGs.shape[0], log_path=log_path, min_loss=5e-2, sv_path=self.specific_path(path))
            self.lgtSGs_import = nn.Parameter(lgtSGs, requires_grad=False)
        self.numLgtSGs_import = self.lgtSGs_import.shape[0]
        self.white_light_import = self.lgtSGs_import.shape[-1] == 5
        self.import_envmap = True
        torch.cuda.empty_cache()
        print('Load Envmap Done!')
    
    def specific_path(self, path):
        return path + '_' + self.name
    
    def switch_envmap_import(self):
        if self.lgtSGs_import is None:
            print('No Imported Light Envmap')
            self.import_envmap = False
        else:
            self.import_envmap = not self.import_envmap
        return self.import_envmap


def SG2Envmap(lgtSGs, H, W, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-0.5*np.pi, 1.5*np.pi, W)])
    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.cos(phi), torch.sin(theta) * torch.sin(phi)],
                           dim=-1)    # [H, W, 3]
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    C = lgtSGs.shape[-1]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, C]).expand(dots_sh+[M, C])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., 4:])  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W, rgb.shape[-1]))
    viewdirs = viewdirs.reshape([*envmap.shape[:-1], -1])
    return envmap, viewdirs


def EnvMap2SG(envmap, numLgtSGs=16, log_path='./logs/', min_loss=5e-2, sv_path='./logs/sg'):
    print('Fitting Environment Map with Spherical Gaussian ...')
    # ground-truth envmap
    gt_envmap = torch.from_numpy(envmap).cuda()
    H, W = gt_envmap.shape[:2]
    os.makedirs(log_path, exist_ok=True)
    assert (os.path.isdir(log_path))
    
    lgtSGs = nn.Parameter(torch.randn(numLgtSGs, 7).cuda())  # lobe + lambda + mu
    lgtSGs.data[..., 3:4] *= 100.
    lgtSGs.requires_grad = True

    optimizer = torch.optim.Adam([lgtSGs,], lr=1e-2)
    N_iter = 20000

    for step in range(N_iter):
        optimizer.zero_grad()
        env_map, viewdirs = SG2Envmap(lgtSGs, H, W)
        loss = torch.mean((env_map - gt_envmap) * (env_map - gt_envmap))
        loss.backward()
        optimizer.step()
        if step % 30 == 0:
            print('step: {}, loss: {}'.format(step, loss.item()))
        if step % 2000 == 0:
            envmap_check = env_map.clone().detach().cpu().numpy()
            gt_envmap_check = gt_envmap.clone().detach().cpu().numpy()
            viewdirs = viewdirs.detach().cpu().numpy()
            save_envmap(envmap_check, viewdirs, os.path.join(log_path, 'log_sg_{}'.format(numLgtSGs)), no_npz=True)
            save_envmap(gt_envmap_check, viewdirs, os.path.join(log_path, 'log_sg_{}_gt'.format(numLgtSGs)), no_npz=True)
            np.save(os.path.join(log_path, 'sg_{}.npy'.format(numLgtSGs)), lgtSGs.clone().detach().cpu().numpy())
        if loss.item() < min_loss:
            break
    np.save(sv_path+'.npy', lgtSGs.clone().detach().cpu().numpy())
    env_map, viewdirs = SG2Envmap(lgtSGs, H, W)
    envmap_check = env_map.clone().detach().cpu().numpy()
    viewdirs = viewdirs.detach().cpu().numpy()
    save_envmap(envmap_check, viewdirs, sv_path, no_npz=True, no_image=True)
    return lgtSGs


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

def save_envmap(envmap, viewdirs, path, no_npz=False, no_image=False, no_ply=False):
    im = np.power(envmap, 1./2.2)
    im = np.clip(im, 0., 1.)
    im = np.uint8(im * 255.)
    if not no_npz:
        np.save(path + '.npy', envmap)
    if not no_image:
        imageio.imwrite(path + '.png', im)
    if not no_ply:
        write_ply_rgb(viewdirs.reshape([-1, 3]), im.reshape([-1, 3]), path + '.ply')
